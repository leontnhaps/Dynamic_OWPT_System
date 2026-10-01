"""M1-3 detection-only preview. Never issues actuator commands."""
import csv
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
from Tx.Controller.live_core import target_from_boxes, fresh
from common.model_paths import DEFAULT_YOLO_PATH, resolve_yolo_path


class PVDetector:
    def __init__(self, path, device):
        path = resolve_yolo_path(path)
        if not Path(path).is_file():
            raise ValueError(f'PV YOLO 모델 파일이 없습니다: {path}')
        from ultralytics import YOLO
        self.model = YOLO(path)
        self.names = self.model.names
        self.device = device

    def infer(self, jpeg, confidence, class_id):
        image = Image.open(io.BytesIO(jpeg)).convert('RGB')
        started = time.perf_counter()
        prediction = self.model.predict(source=image, conf=confidence,
                                        classes=[class_id], device=self.device, verbose=False)[0]
        boxes = prediction.boxes.data.detach().cpu().numpy().tolist() if prediction.boxes is not None else []
        target, count = target_from_boxes(boxes, class_id, confidence, image.size)
        return dict(image=image, target=target, count=count,
                    inference_ms=(time.perf_counter()-started)*1000)


class DetectionPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=6)
        self.app = app
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.model = None
        self.running = False
        self.generation = 0
        self.last_frame = None
        self.latest = None
        self.display = None; self.rendered_frame = None
        self.log = None
        self.widgets = []
        self.path = tk.StringVar(value=str(DEFAULT_YOLO_PATH))
        self.device = tk.StringVar(value='cuda')
        self.conf = tk.StringVar(value='0.5')
        self.class_id = tk.StringVar(value='0')
        row = ttk.Frame(self); row.pack(fill='x')
        ttk.Label(row, text='YOLO .pt').pack(side='left')
        entry = ttk.Entry(row, textvariable=self.path, width=70); entry.pack(side='left', fill='x', expand=True)
        button = ttk.Button(row, text='선택', command=self.browse); button.pack(side='left')
        self.widgets.extend([entry, button])
        row = ttk.Frame(self); row.pack(fill='x', pady=4)
        for label, var in [('장치 cpu/cuda', self.device), ('신뢰도', self.conf), ('PV class ID', self.class_id)]:
            ttk.Label(row, text=label).pack(side='left')
            entry = ttk.Entry(row, textvariable=var, width=9); entry.pack(side='left', padx=4)
            self.widgets.append(entry)
        row = ttk.Frame(self); row.pack(fill='x')
        for label, fn in [('모델 불러오기', self.load), ('검출 시작', self.start), ('검출 정지', self.stop),
                          ('기록 시작', self.start_log), ('기록 종료', self.end_log), ('이미지 저장', self.save)]:
            ttk.Button(row, text=label, command=lambda f=fn: self.guard(f)).pack(side='left', padx=3)
        self.status = tk.StringVar(value='기본 PV 모델 경로 확인 → 모델 불러오기 → 검출 시작')
        ttk.Label(self, textvariable=self.status, wraplength=1100).pack(anchor='w')
        ttk.Label(self, text='원본 영상 좌표 · 최고 confidence PV 하나 선택 · 저장: captures/M1-3/').pack(anchor='w')
        self.canvas = ttk.Label(self, anchor='center'); self.canvas.pack()

    def guard(self, fn):
        try:
            fn()
        except Exception as exc:
            self.stop()
            self.status.set('오류: '+str(exc))
            self.app.record(dict(event='detection_error', message=str(exc)))
            messagebox.showerror('M1-3 PV 검출', str(exc))

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[('YOLO weights', '*.pt')])
        if path: self.path.set(path)

    def lock(self, locked):
        for widget in self.widgets: widget.configure(state='disabled' if locked else 'normal')

    def load(self):
        if self.future is not None: raise ValueError('진행 중인 작업이 끝난 뒤 다시 시도하세요.')
        self.stop()
        device = self.device.get().strip()
        if device not in ('cpu', 'cuda'): raise ValueError('장치는 cpu 또는 cuda를 입력하세요.')
        self.model = None
        self.loaded = dict(path=resolve_yolo_path(self.path.get()), device=device)
        self.lock(True)
        self.job = ('load', self.generation)
        self.future = self.executor.submit(PVDetector, self.loaded['path'], device)
        self.status.set('YOLO 모델 불러오는 중…')

    def start(self):
        timing=getattr(self.app,"timing",None)
        if timing is not None and timing.profile.running:
            raise ValueError("처리 시간 측정을 먼저 종료하세요.")
        if self.model is None: raise ValueError('모델부터 불러오세요.')
        if self.future is not None: raise ValueError('진행 중인 작업이 끝난 뒤 시작하세요.')
        self.stop()
        confidence = float(self.conf.get()); class_id = int(self.class_id.get())
        if not 0 < confidence <= 1 or class_id not in self.model.names:
            raise ValueError(f'신뢰도는 0 초과 1 이하, class ID 확인: {self.model.names}')
        if not self.app.current or not fresh(self.app.current[2], time.monotonic(), 2):
            raise ValueError('M1-1에서 카메라 영상을 먼저 켜세요.')
        # Stop any existing live tracking/learning before a detection-only test.
        self.app.live.stop('M1-3 검출 시험')
        if self.app.live.pending or self.app.servo.pending:
            raise ValueError('이미 전송한 서보 명령의 응답을 기다리세요.')
        self.settings = (confidence, class_id)
        self.running = True; self.last_frame = None
        self.processed = 0; self.started = time.monotonic()
        self.lock(True)
        self.status.set('검출 중 · 서보 제어 없음')

    def stop(self):
        self.running = False; self.generation += 1; self.latest = None
        self.display = None; self.rendered_frame = None
        self.canvas.configure(image=''); self.canvas.image = None
        self.end_log()
        if self.future is None: self.lock(False)
        self.status.set('검출 정지')

    def start_log(self):
        if not self.running: raise ValueError('검출부터 시작하세요.')
        if self.log is not None: return
        folder = self.app.stage_dir('M1-3') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        folder.mkdir()
        settings = dict(model=self.loaded, confidence=self.settings[0], class_id=self.settings[1],
                        coordinate_system='original image pixels; origin top-left',
                        note='Latest-frame sampling; unprocessed camera frames are not detection misses.')
        (folder/'session.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
        self.log = (folder/'detections.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.log, fieldnames=['receive_unix_ns', 'seq', 'simulated', 'width', 'height',
            'status', 'count', 'confidence', 'u', 'v', 'x1', 'y1', 'x2', 'y2', 'inference_ms',
            'result_age_s', 'receive_fps', 'detection_fps'])
        self.writer.writeheader(); self.log.flush()
        self.app.record(dict(event='detection_record_start', folder=str(folder)))

    def end_log(self):
        if self.log is not None:
            self.log.close(); self.log = None
            self.app.record(dict(event='detection_record_stop'))

    def poll(self):
        now = time.monotonic()
        if self.future is not None and self.future.done():
            future = self.future; self.future = None
            if self.job[0] == 'load':
                try: self.model = future.result()
                finally: self.lock(False)
                self.status.set(f'모델 준비 완료: {self.model.names} | {self.loaded["device"]}')
            elif self.job[1] == self.generation:
                self.consume(future.result(), self.job[2], now)
            else:
                self.lock(False)
        if not self.running: return
        frame = self.app.current
        if not frame or not fresh(frame[2], now, 2):
            self.latest = None
            self.display = None
            self.canvas.configure(image=''); self.canvas.image = None
            self.status.set('영상 수신 대기 / 오래된 좌표 무효')
            return
        if self.future is None and frame[2] != self.last_frame:
            self.last_frame = frame[2]
            self.job = ('infer', self.generation, frame)
            self.future = self.executor.submit(self.model.infer, frame[0], *self.settings)

    def consume(self, result, frame, now):
        self.processed += 1
        fps = self.processed / max(now-self.started, .001)
        image = result['image']
        target = result['target']
        stale = not fresh(frame[2], now, 2)
        row = dict(receive_unix_ns=frame[3], seq=frame[1].get('seq'), simulated=frame[1].get('simulated'),
                   width=image.width, height=image.height, status='stale' if stale else ('detected' if target else 'missing'),
                   count=result['count'], inference_ms=result['inference_ms'], result_age_s=now-frame[2],
                   receive_fps=self.app.fps, detection_fps=fps)
        if target and not stale:
            row.update(confidence=target['confidence'], u=target['center'][0], v=target['center'][1])
            row.update(zip(('x1', 'y1', 'x2', 'y2'), target['box']))
            u, v = target['center']
            detail = f'PV ({u:.1f}, {v:.1f}) px | confidence {target["confidence"]:.3f} | 후보 {result["count"]}'
        else:
            detail = '오래된 결과 · 좌표 무효' if stale else 'PV 미검출 · 좌표 없음'
        if self.log is not None: self.writer.writerow(row); self.log.flush()
        self.latest = None if stale else (image, frame, row)
        self.display = (image, frame, row)
        self.status.set(f'{detail} | 검출 {fps:.1f} FPS | 추론 {result["inference_ms"]:.0f} ms | 기록 {"ON" if self.log else "OFF"}')

    @staticmethod
    def annotated(image,row):
        image=image.copy()
        if row.get('status')=='detected':
            draw=ImageDraw.Draw(image)
            draw.rectangle(tuple(row[k] for k in ('x1','y1','x2','y2')),outline='lime',width=3)
            u,v=row['u'],row['v']
            draw.line((u-10,v,u+10,v),fill='lime',width=3)
            draw.line((u,v-10,u,v+10),fill='lime',width=3)
        return image

    def render(self):
        if self.display is None or self.display[1][2]==self.rendered_frame:return
        image,frame,row=self.display
        image=self.annotated(image,row);image.thumbnail((1000,420))
        photo=ImageTk.PhotoImage(image);self.canvas.configure(image=photo);self.canvas.image=photo
        self.rendered_frame=frame[2]

    def save(self):
        if not self.latest or not fresh(self.latest[1][2], time.monotonic(), 2):
            raise ValueError('저장할 최근 검출 결과가 없습니다.')
        image, frame, row = self.latest
        path = self.app.stage_dir('M1-3') / ('frame_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        path.with_suffix('.jpg').write_bytes(frame[0])
        self.annotated(image,row).save(str(path)+'_detection.jpg')
        path.with_suffix('.json').write_text(json.dumps(dict(row, model=self.loaded,
            confidence_threshold=self.settings[0], class_id=self.settings[1], frame_metadata=frame[1]),
            ensure_ascii=False, indent=2), encoding='utf-8')
        self.app.record(dict(event='detection_saved', file=str(path)))

    def close(self):
        self.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
