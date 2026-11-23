from maix import camera, display, image, time, app, uart, sys, pinmap, err
from gui import GUI
from ring_digits import digit_templates, read_ring_digit
import cv2
import numpy as np
import json
import os
from math import hypot
from typing import NamedTuple

class ValueMode:
    Min = 0
    Max = 1


class Circle(NamedTuple):
    center: tuple
    ellipse: tuple
    area: float


gui = GUI()

_btn_mode = -1
_btn_ch1 = -1
_btn_ch2 = -1
_btn_screen = -1
_btn_binary = -1
_btn_wb = -1
_slider_id = -1
_wb_requested = False
_wb_status = ''
_wb_detail = ''
_wb_profile = None
WB_ROI = (128, 88, 64, 64)  # 白纸采样框，坐标基于320×240取景图
WB_DEFAULT_GAINS = {
    'maixcam': [0.134, 0.0625, 0.0625, 0.1239],
    'maixcam2': [0.0682, 0.0, 0.0, 0.04897],
}
SOURCE_VERSION = '1.0.9'
DEBUG_OUTPUT = True
DISPLAY_OFF_COMMAND = 0x07
DISPLAY_ON_COMMAND = 0x08
RING_IDS = (1, 2, 3)
_display_enabled = True
DETECT_SCALE = 0.75  # 240×180检测，输出坐标换算回320×240
_perf = {'capture': 0, 'process': 0, 'gui': 0}
_preview = np.zeros((240, 320, 3), dtype=np.uint8)  # 保存图像底图，供文字逐帧重绘
_screen = np.zeros((320, 320, 3), dtype=np.uint8)
_screen_image = None  # 保持共享显示缓冲区存活
_process_buffers = None  # 首帧按检测尺寸分配，后续复用OpenCV输出数组
TEXT_REFRESH_INTERVAL_MS = 500  # 关闭预览时的文字刷新间隔
_last_text_refresh_ms = None
COLOR_NAMES = ('未知', '红', '绿', '黄', '黑', '蓝', '浅蓝')
LIGHT_BLUE = 6
_color_state = {'candidate': (), 'frames': 0, 'label': '未检测'}


# 默认值、下限、上限、网页输入步长。
DETECTION_PARAMETERS = {
    'canny_low': (50, 0, 255, 1),
    'canny_high': (150, 0, 255, 1),
    'min_circularity': (0.7, 0, 1, 0.01),
    'min_area': (1200, 0, 76800, 1),
    'min_perimeter': (120, 1, 2000, 1),
    'max_perimeter': (555, 1, 2000, 1),
    'saturation_min': (70, 0, 255, 1),
    'brightness_min': (40, 0, 255, 1),
    'black_ratio': (0.25, 0, 1, 0.01),
    'min_color_fraction': (0.18, 0.01, 1, 0.01),
    'red_low_end': (10, 0, 179, 1),
    'red_high_start': (170, 0, 179, 1),
    'green_start': (35, 0, 179, 1),
    'green_end': (85, 0, 179, 1),
    'yellow_start': (18, 0, 179, 1),
    'yellow_end': (34, 0, 179, 1),
    'blue_start': (90, 0, 179, 1),
    'blue_end': (135, 0, 179, 1),
    'light_blue_start': (85, 0, 179, 1),
    'light_blue_end': (135, 0, 179, 1),
    'light_blue_s_min': (30, 1, 255, 1),
    'light_blue_s_max': (170, 1, 255, 1),
    'light_blue_v_min': (155, 0, 255, 1),
}
_detection = {name: limits[0] for name, limits in DETECTION_PARAMETERS.items()}
CANNY_KEYS = ('canny_low', 'canny_high')

_context = {
    'value_mode': ValueMode.Min,
    'disp_binary': True,

}


def valid_wb_gains(gains):
    # MaixCAM2的两个绿色通道增益允许为0。
    return (gains is not None and len(gains) == 4
            and all(0 <= float(value) <= 1 for value in gains)
            and float(gains[0]) > 0 and float(gains[3]) > 0)


def default_wb_gains():
    return WB_DEFAULT_GAINS.get(sys.device_id(), WB_DEFAULT_GAINS['maixcam'])[:]


def apply_white_balance(cam, gains):
    if not valid_wb_gains(gains):
        raise ValueError('白平衡目标增益无效')
    result = cam.set_wb_gain(gains)
    # 官方MaixCAM的回读转换使用整数除法，成功时可能回读四个0。
    if result is None or len(result) != 4:
        raise ValueError('增益写入返回:' + repr(result))


def black_brightness_threshold():
    white_level = _wb_profile['white_level'] if _wb_profile is not None else 200
    return max(30, min(80, white_level * _detection['black_ratio']))


def light_blue_hsv_limits(black_level):
    return (
        (_detection['light_blue_start'], _detection['light_blue_s_min'],
         max(_detection['light_blue_v_min'], int(np.ceil(black_level)))),
        (_detection['light_blue_end'], _detection['light_blue_s_max'], 255),
    )


def detect_circle_color(frame, ellipse, frame_hsv=None):
    center, axes, angle = ellipse
    x, y, width, height = cv2.boundingRect(cv2.boxPoints(ellipse))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, frame.shape[1]), min(y + height, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.ellipse(mask, ((center[0] - x0, center[1] - y0),
                      (axes[0] * 0.97, axes[1] * 0.97), angle), 255, -1)
    sample_frame = frame if frame_hsv is None else frame_hsv
    pixels = sample_frame[y0:y1:2, x0:x1:2][mask[::2, ::2] != 0]
    if len(pixels) < 20:
        return 0
    # 色相区分彩色物料，浅蓝额外要求较高亮度与较低饱和度。
    hsv = (cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
           if frame_hsv is None else pixels)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    black_level = black_brightness_threshold()
    (h_min, s_min, v_min), (h_max, s_max, _) = light_blue_hsv_limits(black_level)
    light_blue = (h >= h_min) & (h <= h_max) & (s >= s_min) & (s <= s_max) & (v >= v_min)
    colored = ((s >= _detection['saturation_min'])
               & (v >= max(_detection['brightness_min'], black_level)) & ~light_blue)
    votes = [
        np.count_nonzero(colored & ((h <= _detection['red_low_end']) | (h >= _detection['red_high_start']))),
        np.count_nonzero(colored & (h >= _detection['green_start']) & (h <= _detection['green_end'])),
        np.count_nonzero(colored & (h >= _detection['yellow_start']) & (h <= _detection['yellow_end'])),
        np.count_nonzero(v < black_level),
        np.count_nonzero(colored & (h >= _detection['blue_start']) & (h <= _detection['blue_end'])),
        np.count_nonzero(light_blue),
    ]
    winner = int(np.argmax(votes))
    return winner + 1 if votes[winner] >= len(pixels) * _detection['min_color_fraction'] else 0


def distinct_circles(circles):
    # Canny 的内外两侧可能为同一个目标产生两个近重合轮廓。
    unique = []
    for circle in sorted(circles, key=lambda candidate: candidate.area, reverse=True):
        x, y = circle.center
        radius = min(circle.ellipse[1]) / (2 * DETECT_SCALE)
        if any(hypot(x - other.center[0], y - other.center[1])
               <= max(2, min(radius, min(other.ellipse[1]) / (2 * DETECT_SCALE)) * 0.25)
               for other in unique):
            continue
        unique.append(circle)
    return unique


def find_ring_shapes(contours, hierarchy, circles, min_circularity):
    if not circles or hierarchy is None:
        return [], set()

    parents = hierarchy[0, :, 3]
    holes = {}
    ring_edges = set()
    for index, contour in enumerate(contours):
        ancestor = parents[index]
        if ancestor < 0 or len(contour) < 5:
            continue
        ancestors = []
        while ancestor >= 0:
            if ancestor in circles:
                ancestors.append(ancestor)
            ancestor = parents[ancestor]
        if not ancestors:
            continue

        area = cv2.contourArea(contour)
        possible = [outer for outer in ancestors if 0.08 <= area / circles[outer].area <= 0.81]
        if not possible:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or 4 * np.pi * area / perimeter ** 2 <= min_circularity:
            continue
        hole = cv2.fitEllipse(contour)
        inner_center, inner_axes, _ = hole
        for outer in possible:
            outer_center, outer_axes, _ = circles[outer].ellipse
            distance = hypot(inner_center[0] - outer_center[0], inner_center[1] - outer_center[1])
            if min(outer_axes) <= 0 or distance > max(2, min(outer_axes) * 0.10):
                continue
            ratios = [inner / outer_axis for inner, outer_axis in zip(sorted(inner_axes), sorted(outer_axes))]
            if min(ratios) < 0.28 or max(ratios) > 0.90 or abs(ratios[0] - ratios[1]) > 0.20:
                continue
            if outer not in holes or area > holes[outer][0]:
                holes[outer] = (area, hole)
            ring_edges.update((outer, index))

    ring_holes = {circles[index]: holes[index][1] for index in sorted(holes)}
    return [(circle, ring_holes[circle]) for circle in distinct_circles(ring_holes)], ring_edges


def identify_rings(gray, shapes):
    readings = [(read_ring_digit(gray, hole), tuple(map(int, circle.center)))
                for circle, hole in shapes]
    ids = [ring_id for ring_id, _ in readings]
    return [(ring_id if ids.count(ring_id) == 1 else 0, center) for ring_id, center in readings]


def summarize_materials(materials):
    if not materials:
        return None, None, ()

    highest = min(circle.center[1] for circle, _ in materials)
    lower = [index for index, (circle, _) in enumerate(materials) if circle.center[1] - highest > 25]
    if len(lower) == 1:
        materials = [material for index, material in enumerate(materials) if index != lower[0]]
    centers = [circle.center for circle, _ in materials]
    colors = [color for _, color in materials]
    center = tuple(int(sum(axis) / len(axis)) for axis in zip(*centers))
    counts = [colors.count(color) for color in range(1, LIGHT_BLUE)]
    color = counts.index(max(counts)) + 1 if any(counts) else 0
    return center, color, colors


def update_color_label(colors):
    candidate = tuple(sorted(set(colors) - {0})) or ((0,) if colors else ())
    if not candidate:
        _color_state.update(candidate=(), frames=0, label='未检测')
        return
    frames = _color_state['frames'] + 1 if candidate == _color_state['candidate'] else 1
    label = ''.join(COLOR_NAMES[value] for value in candidate) if frames >= 3 else '确认中'
    _color_state.update(candidate=candidate, frames=min(frames, 3), label=label)


def sample_white_reference(frame):
    x, y, width, height = WB_ROI
    patch = frame[y:y + height, x:x + width]
    if patch.shape[:2] != (height, width):
        raise ValueError('白纸采样框超出画面')
    # 使用亮度有效的像素，允许框内存在少量高光和暗点。
    valid = (np.min(patch, axis=2) >= 20) & (np.max(patch, axis=2) < 250)
    if np.mean(valid) < 0.5:
        if np.median(np.max(patch, axis=2)) >= 250:
            raise ValueError('白纸过曝，请减弱光照')
        if np.median(np.min(patch, axis=2)) < 20:
            raise ValueError('白纸过暗，请增加光照')
        raise ValueError('请用均匀白纸覆盖中心框')
    pixels = patch[valid]
    if np.max(np.std(pixels, axis=0)) > 30:
        raise ValueError('白纸明暗不均或含图案')
    return np.median(pixels, axis=0)


def load_white_balance(cam):
    global _wb_profile, _wb_status, _wb_detail
    try:
        path = os.path.join(app.get_app_data_path(), 'white_balance.json')
        with open(path, 'r', encoding='utf-8') as config:
            profile = json.load(config)
        if profile['version'] != 1 or not valid_wb_gains(profile['gains']):
            raise ValueError('白平衡参数无效')
        apply_white_balance(cam, profile['gains'])
        _wb_profile = profile
        _wb_status = ''
        _wb_detail = ''
    except FileNotFoundError:
        pass
    except Exception as error:
        _wb_status = '白平衡:加载失败'
        _wb_detail = str(error)
        print('白平衡加载失败:', error)


def calibrate_white_balance(cam):
    global _wb_profile, _wb_status, _wb_detail
    previous_gains = None
    previous_mode = None
    stage = '相机接口'
    try:
        previous_mode = cam.awb_mode()
        previous_gains = list(cam.set_wb_gain())
        print('白平衡初始读数:', previous_gains)
        if not valid_wb_gains(previous_gains) and _wb_profile is not None:
            previous_gains = _wb_profile['gains'][:]
        # 自动模式下的手动增益字段可能尚未初始化，使用官方初值启动。
        gains = (previous_gains if valid_wb_gains(previous_gains) else default_wb_gains())[:]
        cam.awb_mode(camera.AwbMode.Manual)
        # 手动保持增益，分轮测量白纸并使红、蓝通道向绿色通道收敛。
        for attempt in range(8):
            stage = '写入增益'
            apply_white_balance(cam, gains)
            cam.skip_frames(8)
            stage = '白纸采样'
            samples = []
            for _ in range(3):
                frame = image.image2cv(cam.read(), True, True)
                samples.append(sample_white_reference(frame))
            b, g, r = np.mean(samples, axis=0)
            print('白平衡采样:', attempt + 1, 'BGR:', (b, g, r), '增益:', gains)
            if max(b, g, r) / min(b, g, r) <= 1.08:
                break
            gains[0] *= float(g / r) ** 0.7
            gains[3] *= float(g / b) ** 0.7
            if not valid_wb_gains(gains):
                raise ValueError('白平衡校准增益超出范围')
        else:
            raise ValueError('白平衡未收敛，请检查白纸和光照')

        profile = {'version': 1, 'gains': gains, 'white_level': float((b + g + r) / 3)}
        stage = '文件保存'
        directory = app.get_app_data_path()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'white_balance.json')
        # 临时文件落盘后替换，保存完成后同步设备存储。
        with open(path + '.tmp', 'w', encoding='utf-8') as config:
            json.dump(profile, config)
            config.flush()
            os.fsync(config.fileno())
        os.replace(path + '.tmp', path)
        if hasattr(os, 'sync'):
            os.sync()
        _wb_profile = profile
        _wb_status = '白平衡:已保存'
        _wb_detail = ''
    except Exception as error:
        _wb_status = '白平衡:' + stage + '失败'
        _wb_detail = ('固件缺少白平衡接口' if isinstance(error, AttributeError)
                      and stage in ('相机接口', '写入增益') else str(error))
        print('白平衡校准失败:', stage, repr(error))
        try:
            if previous_gains is not None and valid_wb_gains(previous_gains):
                cam.set_wb_gain(previous_gains)
            if previous_mode is not None:
                cam.awb_mode(previous_mode)
        except Exception as restore_error:
            print('白平衡恢复失败:', restore_error)


class Comm:  # 串口类
    def __init__(self):
        if sys.device_id() == 'maixcam2':
            for pin, function in (('A21', 'UART4_TX'), ('A22', 'UART4_RX')):
                err.check_raise(pinmap.set_pin_function(pin, function),
                                '设置{}为{}失败'.format(pin, function))
            serial_device = '/dev/ttyS4'
        else:
            serial_device = '/dev/ttyS0'
        serial = uart.UART(serial_device, 115200)
        time.sleep_ms(100)  # wait uart ready
        self.uart = serial
        self.Data_Temp = [0, 0]

    def _send_position(self, frame_type, center, code):
        x, y = center
        data = bytes((0xb3, frame_type, max(0, x - 255), min(x, 255),
                      max(0, y - 255), min(y, 255), code, 0x5b))
        self.uart.write(data)
        if DEBUG_OUTPUT:
            print("send_data:", data)

    def send_detect(self, center, color):
        self._send_position(0xb3, center, color)

    def send_ring(self, center, ring_id):
        self._send_position(0xb4, center, ring_id)

    def read_commands(self):
        # 屏显采用最后一个指令，每个圆环请求各返回一帧。
        enabled = None
        ring_requests = []
        for value in self.uart.read() or b'':
            if value == DISPLAY_OFF_COMMAND:
                enabled = False
            elif value == DISPLAY_ON_COMMAND:
                enabled = True
            elif value in RING_IDS:
                ring_requests.append(value)
        return enabled, ring_requests


def camera_Init():
    cam = camera.Camera(320, 240, format=image.Format.FMT_BGR888)
    cam.skip_frames(30)  # 等待摄像头稳定
    return cam


def draw_preview(target, frame, edges, show_edges, circles, average_center, marker_color):
    if show_edges:
        resized = cv2.resize(edges, (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR, dst=target)
    else:
        target[:] = frame
    for center, (_, axes, angle), _ in circles:
        scaled_axes = (axes[0] / DETECT_SCALE, axes[1] / DETECT_SCALE)
        cv2.ellipse(target, (center, scaled_axes, angle), (255, 0, 0), 3)
    if average_center is not None:
        cv2.drawMarker(target, average_center, marker_color, cv2.MARKER_CROSS, 20, 3)


def draw_target_labels(preview, pale_centers, rings):
    for center in pale_centers:
        cv2.drawMarker(preview, center, (255, 235, 60), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(preview, '6', (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 235, 60), 2)
    for ring_id, center in rings:
        cv2.drawMarker(preview, center, (0, 165, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(preview, f'R{ring_id}', (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)


def encode_preview(frame):
    try:
        encoded, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if encoded:
            return data.tobytes()
    except (AttributeError, cv2.error):
        pass
    jpeg = image.cv2image(frame, True, False).to_jpeg(75)
    return bytes(jpeg.to_bytes())


def process_frame(img, comm, kernel, gui, fps, tuning=None, ring_requests=()):
    global _screen_image, _last_text_refresh_ms, _process_buffers
    process_start = time.ticks_ms()
    gui.poll()  # 每帧读取触摸，保证关闭预览后按钮和滑条继续响应
    render = _display_enabled
    # 相机直接输出BGR时共享取景缓冲区；其生命周期由当前img保持。
    img_cv = image.image2cv(img, False, False)
    if img.format() != image.Format.FMT_BGR888:
        img_cv = image.image2cv(img, True, True)
    if _process_buffers is None:
        detect_frame = cv2.resize(img_cv, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                                  interpolation=cv2.INTER_AREA)
        _process_buffers = (
            detect_frame, np.empty_like(detect_frame),
            *(np.empty(detect_frame.shape[:2], dtype=np.uint8) for _ in range(6))
        )
    else:
        cv2.resize(img_cv, None, dst=_process_buffers[0], fx=DETECT_SCALE,
                   fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
    detect_frame, frame_hsv, gray, blurred, edged, saturation, color_edges, ceroded = _process_buffers

    # 转灰度
    cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY, dst=gray)
    # 高斯模糊去噪声
    cv2.GaussianBlur(gray, (5, 5), 0, dst=blurred)
    # 边缘检测
    canny_low = _detection['canny_low']
    canny_high = _detection['canny_high']
    cv2.Canny(blurred, canny_low, canny_high, edges=edged)
    # 饱和度边缘补充亮色圆形的边界。
    cv2.cvtColor(detect_frame, cv2.COLOR_BGR2HSV, dst=frame_hsv)
    cv2.extractChannel(frame_hsv, 1, dst=saturation)
    cv2.GaussianBlur(saturation, (5, 5), 0, dst=blurred)
    cv2.Canny(blurred, canny_low, canny_high, edges=color_edges)
    cv2.bitwise_or(edged, color_edges, dst=edged)
    # 浅蓝与白底的灰度、饱和度差可能较小，用颜色掩膜补充边界。
    lower, upper = light_blue_hsv_limits(black_brightness_threshold())
    cv2.inRange(frame_hsv, lower, upper, dst=blurred)
    cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, dst=blurred)
    if cv2.countNonZero(blurred) >= 20:
        cv2.Canny(blurred, canny_low, canny_high, edges=color_edges)
        cv2.bitwise_or(edged, color_edges, dst=edged)
    # 小范围连接边缘，并保留原始边缘像素。
    cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, dst=ceroded, iterations=1)
    cv2.bitwise_or(ceroded, edged, dst=ceroded)
    # 找轮廓
    contours, hierarchy = cv2.findContours(ceroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 圆度阈值
    min_circularity = _detection['min_circularity']

    # 存储所有符合条件的圆形中心
    circles = {}
    min_area = _detection['min_area'] * DETECT_SCALE ** 2
    min_perimeter = _detection['min_perimeter'] * DETECT_SCALE
    max_perimeter = _detection['max_perimeter'] * DETECT_SCALE
    for index, cnt in enumerate(contours):
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)

        # 过滤小轮廓和周长为0的异常轮廓
        if perimeter < min_perimeter or perimeter > max_perimeter:
            continue

        # 计算圆度
        circularity = 4 * np.pi * area / (perimeter **2)
        if circularity > min_circularity:
            # 拟合椭圆，获取中心坐标
            ellipse = cv2.fitEllipse(cnt)
            center = ellipse[0]
            center = (center[0] / DETECT_SCALE, center[1] / DETECT_SCALE)
            if not (0 <= center[0] < img_cv.shape[1] and 0 <= center[1] < img_cv.shape[0]):
                continue
            circles[index] = Circle(center, ellipse, area)

    ring_shapes, ring_edges = find_ring_shapes(contours, hierarchy, circles, min_circularity)
    rings = identify_rings(gray, ring_shapes) if ring_requests or tuning is not None else []
    confirmed_rings = {ring_id: center for ring_id, center in rings if ring_id}
    for requested_id in ring_requests:
        if requested_id in confirmed_rings:
            comm.send_ring(confirmed_rings[requested_id], requested_id)
        else:
            comm.send_ring((0, 0), 0)
    materials = [(circle, detect_circle_color(detect_frame, circle.ellipse, frame_hsv))
                 for index, circle in circles.items() if index not in ring_edges]
    ordinary = [(circle, color) for circle, color in materials if color != LIGHT_BLUE]
    pale = distinct_circles(circle for circle, color in materials if color == LIGHT_BLUE)
    pale_centers = [tuple(map(int, circle.center)) for circle in pale]
    average_center, material_color, colors = summarize_materials(ordinary)
    update_color_label([*colors, LIGHT_BLUE] if pale_centers else colors)
    if average_center is not None:
        comm.send_detect(average_center, material_color)
    for center in pale_centers:
        comm.send_detect(center, LIGHT_BLUE)

    display_center = average_center or (pale_centers[0] if len(pale_centers) == 1 else None)
    display_color = COLOR_NAMES[material_color] if material_color is not None else None
    if pale_centers:
        pale_name = COLOR_NAMES[LIGHT_BLUE]
        display_color = f'{display_color} / {pale_name}' if display_color else pale_name
    show_edges = _context['disp_binary']
    if render:
        draw_preview(_preview, img_cv, ceroded, show_edges, circles.values(),
                     display_center, (255, 0, 0))
        draw_target_labels(_preview, pale_centers, rings)
        x, y, width, height = WB_ROI
        cv2.rectangle(_preview, (x, y), (x + width, y + height), (0, 255, 255), 1)
    if tuning is not None:
        tuning.publish(_detection, {
            'fps': fps,
            'contours': len(circles),
            'center': display_center,
            'color': display_color,
            'light_blue_centers': pale_centers,
            'rings': [{'ring_id': ring_id, 'center': center} for ring_id, center in rings],
        })
        for mode in tuning.preview_modes():
            preview = np.empty_like(img_cv)
            draw_preview(preview, img_cv, ceroded, mode == 'edges', circles.values(),
                         display_center, (0, 255, 0))
            draw_target_labels(preview, pale_centers, rings)
            try:
                tuning.publish_preview(mode, encode_preview(preview))
            except Exception as error:
                print('网页预览编码失败:', error)
                tuning.disable_preview()
                break
    render_start = time.ticks_ms()
    _perf['process'] = render_start - process_start
    if (not render and _screen_image is not None and not gui.dirty
            and _last_text_refresh_ms is not None
            and render_start - _last_text_refresh_ms < TEXT_REFRESH_INTERVAL_MS):
        _perf['gui'] = 0
        return
    # 下方80像素用于触摸控件，FPS按文字宽度贴右下角排列。
    screen = _screen
    # 从干净底图恢复文字背景；关闭图像刷新时使用最后一张底图。
    screen[:240] = _preview
    screen[240:] = 0
    cv2.putText(screen, 'v' + SOURCE_VERSION, (246, 233),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    fps_text = f"FPS:{fps}"
    (text_width, text_height), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(screen, fps_text, (screen.shape[1] - text_width - 5, screen.shape[0] - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    if _screen_image is None:
        _screen_image = image.cv2image(screen, True, False)
    img_show = _screen_image
    # 信息按固定行排列，颜色位于取景区左上角。
    img_show.draw_string(65, 2, '颜色:' + _color_state['label'], image.COLOR_GREEN)
    img_show.draw_string(65, 22, '阈值:{},{}'.format(canny_low, canny_high), image.COLOR_GREEN)
    coordinates = 'X:{} Y:{}'.format(*display_center) if display_center is not None else 'X:-- Y:--'
    img_show.draw_string(65, 42, coordinates, image.COLOR_GREEN)
    img_show.draw_string(65, 62, '轮廓数:{}'.format(len(circles)), image.COLOR_GREEN)
    img_show.draw_string(65, 82, '采{} 算{} 显{}ms'.format(
        _perf['capture'], _perf['process'], _perf['gui']), image.COLOR_GREEN)
    for line in range(3):
        detail = _wb_detail[line * 15:(line + 1) * 15]
        if detail:
            img_show.draw_string(65, 166 + line * 20, detail, image.COLOR_GREEN)
    if _wb_status:
        img_show.draw_string(65, 242, _wb_status, image.COLOR_GREEN)
    gui.run(img_show, poll_touch=False)
    _perf['gui'] = time.ticks_ms() - render_start
    _last_text_refresh_ms = render_start


comm = Comm()  # 串口初始化

###########################################  主函数  ################################################


def set_display_enabled(enabled):
    global _display_enabled, _last_text_refresh_ms
    if _display_enabled != enabled:
        _display_enabled = enabled
        _last_text_refresh_ms = None
        gui.setButtonLabel(_btn_screen, '屏显开' if enabled else '屏显关')


def sync_canny_slider():
    key = CANNY_KEYS[_context['value_mode']]
    gui.setSliderValue(_slider_id, int(_detection[key] / 255 * 100))


def btn_pressed(btn_id, state):
    global _wb_requested, _wb_status, _wb_detail
    if state == 1:
        return
    if btn_id == _btn_screen:
        set_display_enabled(not _display_enabled)
    elif btn_id == _btn_ch1:
        _context['value_mode'] = (ValueMode.Max if _context['value_mode'] == ValueMode.Min
                                  else ValueMode.Min)
        index = _context['value_mode']
        label = 'C Max' if index == ValueMode.Max else 'C Min'
        gui.setButtonLabel(_btn_ch1, label)
        gui.setSliderLabel(_slider_id, label)
        sync_canny_slider()
    elif btn_id == _btn_binary:
        _context['disp_binary'] = not _context['disp_binary']
    elif btn_id == _btn_wb:
        _wb_requested = True
        _wb_status = ''
        _wb_detail = ''


def slider_changed(slider_id, value):
    key = CANNY_KEYS[_context['value_mode']]
    _detection[key] = int(value / 100 * 255)


def start_tuning_server():
    if os.environ.get('MAIXVISION') != '1':
        return None
    try:
        from tuning import TuningServer
        tuning = TuningServer(DETECTION_PARAMETERS)
    except (ImportError, OSError, RuntimeError) as error:
        print('调参网页启动失败:', error)
        return None

    try:
        addresses = sys.ip_address().values()
    except Exception as error:
        print('读取设备 IP 失败:', error)
        addresses = ()
    addresses = sorted({ip for ip in addresses if '.' in ip and not ip.startswith('127.')})
    for ip in addresses or ['<设备 IP>']:
        print('调参网页:', tuning.url(ip), flush=True)
    return tuning


def main():
    global _btn_mode, _btn_ch1, _btn_ch2, _btn_screen, _btn_binary, _btn_wb, _slider_id, _wb_requested
    # 单核平台减少OpenCV线程调度；MaixVision传输使用官方JPEG画质接口。
    cv2.setNumThreads(1)
    digit_templates()
    if hasattr(display, 'set_trans_image_quality'):
        display.set_trans_image_quality(70)
    cam = camera_Init()
    load_white_balance(cam)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    _btn_mode = gui.createButton(0, 0, 60, 64, '圆心')
    _btn_ch1 = gui.createButton(0, 64, 60, 64, 'C Min')
    gui.setButtonCallback(_btn_ch1, btn_pressed)
    _btn_ch2 = gui.createButton(0, 128, 60, 64, '模式3')
    _btn_screen = gui.createButton(0, 192, 60, 64, '屏显开')
    gui.setButtonCallback(_btn_screen, btn_pressed)
    _btn_binary = gui.createButton(0, 256, 60, 64, '显示')
    gui.setButtonCallback(_btn_binary, btn_pressed)
    _btn_wb = gui.createButton(210, 263, 104, 28, '白平衡')
    gui.setButtonCallback(_btn_wb, btn_pressed)
    _slider_id = gui.createSlider(80, 294, 140, 18, label='C Min')
    gui.setSliderCallback(_slider_id, slider_changed)
    sync_canny_slider()
    fps = 0
    tuning = start_tuning_server()
    try:
        while not app.need_exit():
            if _wb_requested:
                _wb_requested = False
                calibrate_white_balance(cam)
            frame_start = time.ticks_ms()
            if tuning is not None:
                changes = tuning.take_changes()
                _detection.update(changes)
                if any(key in changes for key in CANNY_KEYS):
                    sync_canny_slider()
            command, ring_requests = comm.read_commands()
            if command is not None:
                set_display_enabled(command)
            capture_start = time.ticks_ms()
            img = cam.read()
            _perf['capture'] = time.ticks_ms() - capture_start
            process_frame(img, comm, kernel, gui, fps, tuning, ring_requests)
            fps = int(1000.0 / max(time.ticks_ms() - frame_start, 1))
            if DEBUG_OUTPUT:
                print('FPS:', fps, '阶段耗时:', _perf)
    finally:
        if tuning is not None:
            tuning.close()


if __name__ == "__main__":
    main()
