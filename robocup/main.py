from maix import camera, display, image, time, app
from maix.v1.machine import UART
from gui import GUI
import cv2
import numpy as np
import json
import os

class ValueMode:
    Min = 0
    Max = 1

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
WB_DEFAULT_GAINS = [0.134, 0.0625, 0.0625, 0.1239]  # 官方MaixCAM初始增益
SOURCE_VERSION = '1.0.9'
DEBUG_OUTPUT = True
DISPLAY_OFF_COMMAND = 0x07
DISPLAY_ON_COMMAND = 0x08
_display_enabled = True
DETECT_SCALE = 0.75  # 240×180检测，输出坐标换算回320×240
_perf = {'capture': 0, 'process': 0, 'gui': 0}
_preview = np.zeros((240, 320, 3), dtype=np.uint8)  # 保存图像底图，供文字逐帧重绘
_screen = np.zeros((320, 320, 3), dtype=np.uint8)
_screen_image = None  # 保持共享显示缓冲区存活
_process_buffers = None  # 首帧按检测尺寸分配，后续复用OpenCV输出数组
TEXT_REFRESH_INTERVAL_MS = 500  # 关闭预览时的文字刷新间隔
_last_text_refresh_ms = None
COLOR_NAMES = ('未知', '红', '绿', '黄', '黑', '蓝')
_color_state = {'candidate': (), 'frames': 0, 'label': '未检测'}
CIRCLE_MIN_AREA = 1200
CIRCLE_MIN_PERIMETER = 120
Yuanxy = [0, 0, 0, 0, 0]


_context = {
    'value_mode': ValueMode.Min,

    'disp_binary': True,

    'threshold_circle':[50,150]
}


def valid_wb_gains(gains):
    # MaixCAM2的两个绿色通道增益允许为0。
    return (gains is not None and len(gains) == 4
            and all(0 <= float(value) <= 1 for value in gains)
            and float(gains[0]) > 0 and float(gains[3]) > 0)


def apply_white_balance(cam, gains):
    if not valid_wb_gains(gains):
        raise ValueError('白平衡目标增益无效')
    result = cam.set_wb_gain(gains)
    # 官方MaixCAM的回读转换使用整数除法，成功时可能回读四个0。
    if result is None or len(result) != 4:
        raise ValueError('增益写入返回:' + repr(result))


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
    # 色相区分红绿黄蓝，黑色使用相对白纸参考亮度。
    hsv = (cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
           if frame_hsv is None else pixels)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    white_level = _wb_profile['white_level'] if _wb_profile is not None else 200
    black_level = max(30, min(80, white_level * 0.25))
    colored = (s >= 70) & (v >= max(40, black_level))
    votes = [
        np.count_nonzero(colored & ((h <= 10) | (h >= 170))),
        np.count_nonzero(colored & (h >= 35) & (h <= 85)),
        np.count_nonzero(colored & (h >= 18) & (h <= 34)),
        np.count_nonzero(v < black_level),
        np.count_nonzero(colored & (h >= 90) & (h <= 135)),
    ]
    winner = int(np.argmax(votes))
    return winner + 1 if votes[winner] >= len(pixels) * 0.18 else 0


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
        gains = (previous_gains if valid_wb_gains(previous_gains) else WB_DEFAULT_GAINS)[:]
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
        uart = UART("/dev/ttyS0", 115200)
        time.sleep_ms(100)  # wait uart ready
        self.uart = uart
        self.Data_Temp = [0, 0]

    def send_detect(self, Mode, Data):  # 发送数据包
        data = None
        if Mode == 1:
            data = bytes([0xb3, 0xb3, 0x01, int(Data[0]), int(Data[1]), int(Data[2]), int(Data[3]), 0, 0x5b])
            self.uart.write(data)
        if Mode == 2:
            data = bytes([0xb3, 0xb3, 0x02, int(Data[0]), int(Data[1]), int(Data[2]), int(Data[3]), 0, 0x5b])
            self.uart.write(data)
        if Mode == 3:
            data = bytes([0xb3, 0xb3, 0x03, int(Data[0]), int(Data[1]), int(Data[2]), int(Data[3]), int(Data[4]), 0x5b])
            self.uart.write(data)
        if Mode == 4:
            data = bytes([0xb3, 0xb3, 0x04, int(Data), 0, 0, 0, 0, 0x5b])
            self.uart.write(data)
        if Mode == 5:
            data = bytes([0xb3, 0xb3, 0x04, 6, 6, 6, 6, 6, 0x5b])
            self.uart.write(data)
        if data is not None and DEBUG_OUTPUT:
            print("send_data:", data)

    def read_display_command(self):
        # 非阻塞读取；连续收到多个开关指令时采用最后一个。
        enabled = None
        for value in self.uart.read() or b'':
            if value == DISPLAY_OFF_COMMAND:
                enabled = False
            elif value == DISPLAY_ON_COMMAND:
                enabled = True
        return enabled


def camera_Init():
    cam = camera.Camera(320, 240, format=image.Format.FMT_BGR888)
    cam.skip_frames(30)  # 等待摄像头稳定
    return cam


def process_frame(img, comm, kernel, gui, fps):
    global Yuanxy, _screen_image, _last_text_refresh_ms, _process_buffers
    process_start = time.ticks_ms()
    gui.poll()  # 每帧读取触摸，保证关闭预览后按钮和滑条继续响应
    render = _display_enabled
    circle_ellipses = []
    detected_colors = []
    avg_center_x = 0  # 平均中心X坐标
    avg_center_y = 0  # 平均中心Y坐标
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
    cv2.Canny(blurred, *_context['threshold_circle'], edges=edged)
    # 饱和度边缘补充亮色圆形的边界。
    cv2.cvtColor(detect_frame, cv2.COLOR_BGR2HSV, dst=frame_hsv)
    cv2.extractChannel(frame_hsv, 1, dst=saturation)
    cv2.GaussianBlur(saturation, (5, 5), 0, dst=blurred)
    cv2.Canny(blurred, *_context['threshold_circle'], edges=color_edges)
    cv2.bitwise_or(edged, color_edges, dst=edged)
    # 小范围连接边缘，并保留原始边缘像素。
    cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, dst=ceroded, iterations=1)
    cv2.bitwise_or(ceroded, edged, dst=ceroded)
    # 将单通道的ceroded转换为三通道BGR格式（便于显示和后续叠加彩色标记）
    ceroded_rgb = None
    if render:
        ceroded_rgb = _preview
        if _context['disp_binary']:
            edges_show = cv2.resize(ceroded, (img_cv.shape[1], img_cv.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            cv2.cvtColor(edges_show, cv2.COLOR_GRAY2BGR, dst=ceroded_rgb)
        else:
            ceroded_rgb[:] = img_cv
    # 找轮廓
    contours, _ = cv2.findContours(ceroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 圆度阈值
    min_circularity = 0.7

    # 存储所有符合条件的圆形中心
    circle_centers = []
    min_area = CIRCLE_MIN_AREA * DETECT_SCALE ** 2
    min_perimeter = CIRCLE_MIN_PERIMETER * DETECT_SCALE
    max_perimeter = 555 * DETECT_SCALE
    for cnt in contours:
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
            center, axes, angle = cv2.fitEllipse(cnt)
            circle_ellipses.append((center, axes, angle))
            center = (center[0] / DETECT_SCALE, center[1] / DETECT_SCALE)
            axes = (axes[0] / DETECT_SCALE, axes[1] / DETECT_SCALE)
            circle_centers.append(center)  # 原始320×240图像坐标

            # 绘制所有符合条件的圆（便于调试）
            if render:
                cv2.ellipse(ceroded_rgb, (center, axes, angle), (255, 0, 0), 3)

    if circle_centers:
        # 提取所有圆心的y坐标
        ys = [c[1] for c in circle_centers]
        # 最上方圆的y坐标（最小值，因为y越小越靠上）
        min_y = min(ys)
        # 计算每个圆心y坐标与最上方圆心y坐标的差值
        y_diffs = [y - min_y for y in ys]

        # 找出差值大于25像素的圆心（可能的错误圆）
        error_indices = [i for i, diff in enumerate(y_diffs) if diff > 25]

        # 仅当存在且仅存在1个错误圆时，剔除该圆
        if len(error_indices) == 1:
            filtered_centers = [c for i, c in enumerate(circle_centers) if i not in error_indices]
        else:
            filtered_centers = circle_centers  # 无错误圆或不符合条件，保留所有圆

        # 计算过滤后圆心的平均坐标
        xs_filtered = [c[0] for c in filtered_centers]
        ys_filtered = [c[1] for c in filtered_centers]
        avg_center_x = sum(xs_filtered) / len(xs_filtered)
        avg_center_y = sum(ys_filtered) / len(ys_filtered)

        # 在处理后的图像上标记平均中心
        if render:
            cv2.drawMarker(ceroded_rgb, (int(avg_center_x), int(avg_center_y)),
                        (255, 0, 0), cv2.MARKER_CROSS, 20, 3)

    if circle_centers:  # 存在圆形时才更新数据
        # 保存平均中心坐标
        Yuanxy = [0, 0, 0, 0, 0]
        Yuanxy[1] = int(avg_center_x)
        Yuanxy[3] = int(avg_center_y)
        if Yuanxy[1] > 255:
            Yuanxy[0] = Yuanxy[1] - 255
            Yuanxy[1] = 255
        if Yuanxy[3] > 255:
            Yuanxy[2] = Yuanxy[3] - 255
            Yuanxy[3] = 255

        comm.send_detect(0x02, Yuanxy)

    if circle_centers:
        detected_colors = [detect_circle_color(detect_frame, ellipse, frame_hsv)
                           for center, ellipse in zip(circle_centers, circle_ellipses)
                           if center in filtered_centers]
    update_color_label(detected_colors)
    render_start = time.ticks_ms()
    _perf['process'] = render_start - process_start
    if (not render and _screen_image is not None and not gui.dirty
            and _last_text_refresh_ms is not None
            and render_start - _last_text_refresh_ms < TEXT_REFRESH_INTERVAL_MS):
        _perf['gui'] = 0
        return
    if render:
        x, y, width, height = WB_ROI
        cv2.rectangle(ceroded_rgb, (x, y), (x + width, y + height), (0, 255, 255), 1)
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
    img_show.draw_string(65, 22, '阈值:{},{}'.format(*_context['threshold_circle']), image.COLOR_GREEN)
    coordinates = 'X:{} Y:{}'.format(int(avg_center_x), int(avg_center_y)) if circle_centers else 'X:-- Y:--'
    img_show.draw_string(65, 42, coordinates, image.COLOR_GREEN)
    img_show.draw_string(65, 62, '轮廓数:{}'.format(len(circle_centers)), image.COLOR_GREEN)
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
        gui.setSliderValue(_slider_id, int(_context['threshold_circle'][index] / 255 * 100))
    elif btn_id == _btn_binary:
        _context['disp_binary'] = not _context['disp_binary']
    elif btn_id == _btn_wb:
        _wb_requested = True
        _wb_status = ''
        _wb_detail = ''


def slider_changed(slider_id, value):
    _context['threshold_circle'][_context['value_mode']] = int(value / 100 * 255)


def main():
    global _btn_mode, _btn_ch1, _btn_ch2, _btn_screen, _btn_binary, _btn_wb, _slider_id, _wb_requested
    # 单核平台减少OpenCV线程调度；MaixVision传输使用官方JPEG画质接口。
    cv2.setNumThreads(1)
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
    gui.setSliderValue(_slider_id, int(_context['threshold_circle'][0] / 255 * 100))
    fps = 0
    while not app.need_exit():
        if _wb_requested:
            _wb_requested = False
            calibrate_white_balance(cam)
        frame_start = time.ticks_ms()
        command = comm.read_display_command()
        if command is not None:
            set_display_enabled(command)
        capture_start = time.ticks_ms()
        img = cam.read()
        _perf['capture'] = time.ticks_ms() - capture_start
        process_frame(img, comm, kernel, gui, fps)
        fps = int(1000.0 / max(time.ticks_ms() - frame_start, 1))
        if DEBUG_OUTPUT:
            print('FPS:', fps, '阶段耗时:', _perf)


if __name__ == "__main__":
    main()
