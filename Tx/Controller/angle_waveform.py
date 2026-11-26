"""Save an armed optical angle attempt and open its comparison from the controller."""
import json
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from uuid import uuid4

from PIL import Image, ImageTk

from common.angle_demo import DECODE_SAMPLE_MS, payload_from_frame, validate_angle_bit_ms
from Tx.Controller.pi_ook_rx import PiCapture


ANGLE_DECODER_VERSION = 'ook-angle-postcapture-v1'


class AngleWaveform(PiCapture):
    """Use the speed-test CSV format. PC arrival times are shared by a batch."""

    def __init__(self, stage_dir, token, bit_ms, frame_bits, *, context=None):
        self.bit_ms = validate_angle_bit_ms(bit_ms)
        self.token = token
        self.end_received = self.closed = False
        self.end_count = None
        self.frame_check = None
        self.gaps_over_bit = 0
        context = dict(context or {})
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        folder = Path(stage_dir) / f'angle_ook_{stamp}_{uuid4().hex[:8]}'
        settings = dict(bits=frame_bits, payload_bits=payload_from_frame(frame_bits),
                        bit_ms=bit_ms, bitrate_bps=1000 / bit_ms, repeats=1)
        folder.mkdir(parents=True)
        super().__init__(folder, settings, context.get('rx_ssh_target'), token, DECODE_SAMPLE_MS,
                         dict(kind='angle_ook', decoder=ANGLE_DECODER_VERSION,
                              context=context, pc_start_unix_ns=time.time_ns(),
                              pc_receive_timing='Time at receipt of this sample batch, common to its rows.'))

    def tx_event(self, event):
        if self.closed or self.token not in (event.get('token'), event.get('request_id')):
            return
        if isinstance(event.get('frame_check'), dict):
            self.frame_check = event['frame_check']
        with (self.folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event, pc_recorded_unix_ns=time.time_ns()),
                                    ensure_ascii=False, allow_nan=False) + '\n')
        if event.get('kind') == 'armed':
            if event.get('bit_ms') != self.bit_ms or event.get('sample_ms') != self.sample_ms:
                raise ValueError('Rx가 확인한 비트 길이 또는 표본 간격이 전송 설정과 다릅니다.')
            self.confirm(event)

    def samples(self, event):
        if self.closed or self.token != event.get('token') or self.end_received:
            raise ValueError('각도 파형 표본의 요청 ID 또는 도착 순서를 확인하세요.')
        for seq, ns, raw in event['rows']:
            previous_ns = self.last_ns
            # A saturated first sample may terminate the arm before a ready ACK.
            # Retain that diagnostic data with interval_confirmed still false.
            self._write_raw_sample(dict(seq=seq, rx_monotonic_ns=ns, bus_raw=raw,
                                        pc_receive_monotonic_ns=event.get('pc_receive_monotonic_ns'),
                                        pc_receive_unix_ns=event.get('pc_receive_unix_ns')))
            if previous_ns is not None and ns - previous_ns > self.bit_ms * 1_000_000:
                self.gaps_over_bit += 1
        self.file.flush()

    def end_samples(self, event):
        count = event.get('count')
        if (self.closed or self.end_received or event.get('token') != self.token
                or type(count) is not int or count < 0):
            raise ValueError('각도 파형 종료 응답을 확인할 수 없습니다.')
        self.end_received = True
        self.end_count = count
        if count != self.rows:
            raise ValueError(f'각도 파형 표본 수 불일치 · Rx {count}개, PC {self.rows}개')

    def finish(self, success, reason):
        if self.closed:
            return
        self.file.close()
        # A decode timeout can still have a complete, useful waveform.
        complete = self.end_received and self.end_count == self.rows and not self.invalid_lines
        summary = dict(schema_version=2, decoder=ANGLE_DECODER_VERSION, frame_check=self.frame_check,
                       token=self.token, bit_ms=self.bit_ms, sample_ms=self.sample_ms,
                       angle_verified=bool(success), end_reason=reason, waveform_complete=bool(complete),
                       interval_confirmed=self.config['interval_confirmed'],
                       pc_rows=self.rows, rx_reported_rows=self.end_count, missing_sample_slots=self.missing_records,
                       gaps_longer_than_one_bit=self.gaps_over_bit, max_sample_dt_ms=self.max_sample_dt_ms,
                       clipped_rows=self.clipped_rows, invalid_lines=self.invalid_lines,
                       pc_end_unix_ns=time.time_ns())
        tmp = self.folder / 'summary.tmp'
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        tmp.replace(self.folder / 'summary.json')
        self.closed = True


def _compare(folder):
    from Tx.Controller.angle_waveform_compare import compare_angle_waveform

    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    return compare_angle_waveform(folder.parent, folder, summary['bit_ms'], summary['sample_ms'])


class WaveformReview(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='angle-waveform')
        self.folder = self.future = self.png = None
        self.button = ttk.Button(self, text='기본 성공 파형과 비교', command=self.show, state='disabled')
        self.button.pack(side='left')
        self.note = tk.StringVar(self, value='')
        ttk.Label(self, textvariable=self.note, wraplength=740).pack(side='left', padx=5)

    def recording(self, folder):
        self.folder = Path(folder)
        self.future = self.png = None
        self.button.configure(state='disabled')
        self.note.set('각도 OOK 파형 기록 중')

    def analyze(self, folder):
        job = self.executor.submit(_compare, Path(folder))
        if self.folder == Path(folder):
            self.future = job
            self.note.set('전압 기록 저장 · 같은 속도 시험과 비교 중…')

    def poll(self):
        if self.future is None or not self.future.done():
            return
        job, self.future = self.future, None
        try:
            self.png, note = job.result()
            self.note.set(note)
            self.button.configure(state='normal')
        except Exception as exc:
            self.note.set(f'파형 비교 오류 · {exc} · 원시 기록 {self.folder.name}')

    def show(self):
        if self.png is None:
            return
        try:
            with Image.open(self.png) as plot:
                plot.thumbnail((int(self.winfo_screenwidth() * .9), int(self.winfo_screenheight() * .84)),
                               Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(plot, master=self)
        except OSError as exc:
            self.note.set(f'비교 그림을 열 수 없습니다 · {exc}')
            return
        view = tk.Toplevel(self)
        view.title('각도 OOK · 기본 시험 파형 비교')
        label = ttk.Label(view, image=photo)
        label.image = photo
        label.pack()

    def close(self):
        self.executor.shutdown(wait=False)
