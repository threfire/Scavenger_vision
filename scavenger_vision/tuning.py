import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from threading import Lock, Thread
from time import monotonic


PREVIEW_MODES = ('camera', 'edges')
PREVIEW_INTERVAL = 0.2
PREVIEW_KEEPALIVE = 2


ORDERED_PARAMETERS = (
    ('canny_low', 'canny_high', 'Canny 下限不能大于上限'),
    ('min_perimeter', 'max_perimeter', '最小周长不能大于最大周长'),
    ('red_low_end', 'red_high_start', '红色低段上限不能大于高段下限'),
    ('green_start', 'green_end', '绿色 H 下限不能大于上限'),
    ('yellow_start', 'yellow_end', '黄色 H 下限不能大于上限'),
    ('blue_start', 'blue_end', '蓝色 H 下限不能大于上限'),
    ('light_blue_start', 'light_blue_end', '浅蓝 H 下限不能大于上限'),
    ('light_blue_s_min', 'light_blue_s_max', '浅蓝 S 下限不能大于上限'),
)


class TuningHandler(BaseHTTPRequestHandler):
    timeout = 2

    def log_message(self, format, *args):
        pass

    def reply(self, status, content_type, body):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def reply_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.reply(status, 'application/json; charset=utf-8', body)

    def authorized(self):
        authorization = self.headers.get('Authorization', '').encode('latin-1')
        if hmac.compare_digest(authorization, self.server.authorization):
            return True
        self.reply_json(403, {'error': '请打开本次 MaixVision 控制台里打印的完整链接'})
        return False

    def do_GET(self):
        if self.path in ('/', self.server.page_path):
            self.reply(200, 'text/html; charset=utf-8', self.server.page)
        elif self.path == '/api/state':
            if self.authorized():
                self.reply_json(200, self.server.snapshot())
        elif self.path.startswith('/api/preview/'):
            mode = self.path.rsplit('/', 1)[-1]
            if mode not in PREVIEW_MODES:
                self.reply_json(404, {'error': '预览模式不存在'})
            elif self.authorized():
                try:
                    frame = self.server.get_preview(mode)
                except RuntimeError:
                    self.reply_json(503, {'error': '设备无法编码预览画面'})
                    return
                self.reply(200 if frame else 204, 'image/jpeg', frame or b'')
        else:
            self.reply_json(404, {'error': '页面不存在'})

    def do_POST(self):
        if self.path != '/api/settings':
            self.reply_json(404, {'error': '接口不存在'})
            return
        if not self.authorized():
            return
        if self.headers.get_content_type() != 'application/json':
            self.reply_json(415, {'error': '请发送 JSON 参数'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2048:
                raise ValueError
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError
            changes = json.loads(body.decode('utf-8'))
        except (OSError, ValueError, RecursionError):
            self.reply_json(400, {'error': '参数格式无效'})
            return
        try:
            self.server.submit(changes)
        except ValueError as error:
            self.reply_json(400, {'error': str(error)})
            return
        self.reply_json(202, {'accepted': True})


class TuningServer(ThreadingHTTPServer):
    def __init__(self, parameters, host='0.0.0.0', port=8765):
        self.page = Path(__file__).with_suffix('.html').read_bytes()
        self.parameters = parameters
        self._key = token_urlsafe(18)
        self.page_path = '/tune/' + self._key
        self.authorization = ('Bearer ' + self._key).encode('ascii')
        self._settings = {name: limits[0] for name, limits in parameters.items()}
        self._pending = {}
        self._result = {}
        self._preview_requests = {}
        self._preview_frames = {}
        self._next_preview = 0
        self._preview_disabled = False
        self._lock = Lock()
        super().__init__((host, port), TuningHandler)
        self._thread = Thread(target=self.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True)
        try:
            self._thread.start()
        except RuntimeError:
            self.server_close()
            raise

    def url(self, ip):
        return f'http://{ip}:{self.server_port}{self.page_path}'

    def close(self):
        self.shutdown()
        self.server_close()
        self._thread.join()

    def snapshot(self):
        with self._lock:
            return {
                'parameters': self.parameters,
                'settings': {**self._settings, **self._pending},
                'result': self._result,
            }

    def submit(self, changes):
        if not isinstance(changes, dict) or not changes or changes.keys() - self.parameters.keys():
            raise ValueError('参数名称无效')
        for name, value in changes.items():
            default, low, high, _ = self.parameters[name]
            allowed_types = (int,) if type(default) is int else (int, float)
            if type(value) not in allowed_types or not low <= value <= high:
                raise ValueError(f'{name} 的范围为 {low}–{high}')

        with self._lock:
            updated = {**self._settings, **self._pending, **changes}
            for low, high, message in ORDERED_PARAMETERS:
                if (low in changes or high in changes) and updated[low] > updated[high]:
                    raise ValueError(message)
            self._pending.update(changes)

    def take_changes(self):
        with self._lock:
            changes = self._pending
            self._pending = {}
            self._settings.update(changes)
            return changes

    def publish(self, settings, result):
        with self._lock:
            self._settings.update(settings)
            self._result = result

    def get_preview(self, mode):
        with self._lock:
            if self._preview_disabled:
                raise RuntimeError('预览已停用')
            now = monotonic()
            self._preview_requests[mode] = now
            timestamp, frame = self._preview_frames.get(mode, (0, None))
            if now - timestamp < PREVIEW_KEEPALIVE:
                return frame
            return None

    def preview_modes(self):
        now = monotonic()
        if now < self._next_preview:
            return ()
        self._next_preview = now + PREVIEW_INTERVAL
        with self._lock:
            if self._preview_disabled:
                return ()
            return tuple(mode for mode, last_request in self._preview_requests.items()
                         if now - last_request < PREVIEW_KEEPALIVE)

    def publish_preview(self, mode, frame):
        with self._lock:
            self._preview_frames[mode] = (monotonic(), frame)

    def disable_preview(self):
        with self._lock:
            self._preview_disabled = True
            self._preview_frames.clear()
