"""Batch controls for the existing OOK tab; laser operations stay in OokPanel."""
import json
import queue
import threading
import time
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog
from common.ook_experiment import make_plan, number_list, RULES


class OokExperimentMixin:
    def init_experiment(self):
        self.experiment = None
        self.experiment_root = None
        self.last_experiment = None
        self.batch_index = 0
        self.batch_tx_failures = 0
        self.next_trial_at = None
        self.active_spec = None
        self.analysis_worker = None
        self.analysis_events = queue.Queue()
        self.sample_ms = tk.StringVar(value='5')
        self.sweep_bits = tk.StringVar(value='100,50,30,20,15,10,5')
        self.sweep_samples = tk.StringVar(value='5,10')
        self.sweep_count = tk.StringVar(value='10')
        self.sweep_seed = tk.StringVar(value='20261028')
        self.sweep_pattern = tk.StringVar(value='난수32bit')
        self.condition_note = tk.StringVar(value='5m; ambient= ; alignment= ')
        self.experiment_status = tk.StringVar(value='조건별 10회는 탐색용 · 경계 재측정 후 새 데이터로 검증')

    def build_experiment_controls(self):
        box = ttk.LabelFrame(self, text='반복 실험 · 현재 조건을 고정하고 실행', padding=4)
        box.pack(fill='x', pady=4)
        row = ttk.Frame(box); row.pack(fill='x')
        for label, variable, width in [('비트(ms)', self.sweep_bits, 25),
                                      ('측정(ms)', self.sweep_samples, 8), ('조건별 횟수', self.sweep_count, 5),
                                      ('Seed', self.sweep_seed, 10)]:
            ttk.Label(row, text=label).pack(side='left', padx=3)
            ttk.Entry(row, textvariable=variable, width=width).pack(side='left')
        row = ttk.Frame(box); row.pack(fill='x', pady=3)
        ttk.Combobox(row, textvariable=self.sweep_pattern, values=['난수32bit', '1010 반복', '00001111 반복'],
                     state='readonly', width=16).pack(side='left')
        for text, callback in [('① 1초 ON/OFF 반복', lambda: self.start_experiment('step')),
                               ('② 속도 비교 시작', lambda: self.start_experiment('sweep')),
                               ('결과 폴더 분석', self.choose_analysis)]:
            ttk.Button(row, text=text, command=lambda fn=callback:self.guard(fn)).pack(side='left', padx=3)
        row = ttk.Frame(box); row.pack(fill='x')
        ttk.Label(row, text='조건 메모').pack(side='left', padx=3)
        ttk.Entry(row, textvariable=self.condition_note).pack(side='left', fill='x', expand=True)
        ttk.Label(box, textvariable=self.experiment_status, wraplength=1100).pack(anchor='w')

    def start_experiment(self, mode):
        self.check_ready()
        if self.arduino is None or not self.arduino.connected:
            raise ValueError('Arduino 연결이 필요합니다.')
        if self.hardware.get('simulated'):
            raise ValueError('실제 장비로 연결하세요. 시뮬레이션 결과는 광학 실험에 사용할 수 없습니다.')
        if 'laser_pattern' not in self.hardware.get('capabilities', []):
            raise ValueError('Pi를 M4-4로 업데이트하고 재시작하세요.')
        if self.analysis_worker and self.analysis_worker.is_alive():
            raise ValueError('진행 중인 결과 분석이 끝난 뒤 시작하세요.')
        # Fail before turning on the laser if plotting/analysis dependencies are missing.
        try:
            import numpy
            import matplotlib
        except ImportError as exc:
            raise ValueError('requirements-m4-4.txt의 Python 패키지를 설치하세요.') from exc
        plan = make_plan(number_list(self.sweep_bits.get()) if mode == 'sweep' else [],
                         number_list(self.sweep_samples.get(), integer=True),
                         int(self.sweep_count.get()), int(self.sweep_seed.get()),
                         {'난수32bit':'random', '1010 반복':'alternating', '00001111 반복':'runs'}[self.sweep_pattern.get()],
                         mode=mode, note=self.condition_note.get())
        root = self.app.stage_dir('M4-4') / ('experiment_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:6])
        root.mkdir()
        (root/'experiment.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        self.experiment, self.experiment_root, self.last_experiment = plan, root, root
        self.batch_index = 0
        self.batch_tx_failures = 0
        self.next_trial_at = time.monotonic()
        self.rx_save.set(True)
        self.experiment_status.set(f'총 {len(plan["trials"])}회 준비 · 조건 순서 무작위 · 종료 후 자동 분석')

    def cancel_experiment(self, reason):
        plan, self.experiment = self.experiment, None
        self.next_trial_at = None
        if plan:
            self.experiment_status.set(f'반복 실험 종료: {reason} · 송신 시간 초과 {self.batch_tx_failures}회 · 저장 폴더: {self.experiment_root}')
            (self.experiment_root/'run.json').write_text(json.dumps(dict(schema_version=2, reason=reason,
                finished_trials=self.batch_index, planned_trials=len(plan['trials']),
                completed_trials=self.batch_index-self.batch_tx_failures,
                timing_overrun_trials=self.batch_tx_failures), indent=2), encoding='utf-8')

    def trial_finished(self, reason):
        if not self.experiment:
            return
        if reason not in ('completed', 'tx_timing_overrun'):
            self.cancel_experiment(reason)
            return
        # Only a Pi terminal timing_overrun with confirmed OFF is allowed here.
        # Retain this invalid TX attempt; never count it as an optical failure,
        # retry it invisibly, or relax the Pi's timing deadline.
        if reason == 'tx_timing_overrun':
            self.batch_tx_failures += 1
        self.batch_index += 1
        if self.batch_index == len(self.experiment['trials']):
            root = self.experiment_root
            self.cancel_experiment('completed')
            self.start_analysis(root)
        else:
            self.next_trial_at = time.monotonic() + .25

    def tick_experiment(self):
        while True:
            try:
                ok, result = self.analysis_events.get_nowait()
            except queue.Empty:
                break
            self.experiment_status.set(('분석 저장: ' if ok else '분석 실패: ') + str(result))
        if not self.experiment or self.next_trial_at is None or time.monotonic() < self.next_trial_at:
            return
        if self.capture or self.pending:
            return
        self.next_trial_at = None
        try:
            self.check_ready(for_experiment=True)
            spec = self.experiment['trials'][self.batch_index]
            self.active_spec = dict(spec, rules_version=RULES['version'])
            folder = self.experiment_root / f'trial_{self.batch_index+1:05d}'
            self.experiment_status.set(f'{self.batch_index+1}/{len(self.experiment["trials"])} · '
                f'비트 {spec["settings"]["bit_ms"]} ms · 측정 {spec["sample_ms"]} ms · '
                f'송신 시간 초과 {self.batch_tx_failures}회')
            self.begin_trial(spec['settings'], spec['sample_ms'], folder=folder, experiment=self.active_spec)
        except (ValueError, OSError) as exc:
            self.abort_capture('experiment_error')
            self.experiment_status.set('실험 중단: '+str(exc))

    def choose_analysis(self):
        if self.busy:
            raise ValueError('실험을 마치거나 중지한 후 분석하세요.')
        folder = filedialog.askdirectory(title='experiment.json이 있는 실험 폴더 선택',
                                         initialdir=str(self.last_experiment or self.app.stage_dir('M4-4')))
        if folder:
            self.start_analysis(folder)

    def start_analysis(self, folder):
        if self.analysis_worker and self.analysis_worker.is_alive():
            raise ValueError('분석 중입니다.')
        def run():
            try:
                from Tx.Controller.ook_analysis import analyze_experiment
                self.analysis_events.put((True, analyze_experiment(folder)))
            except Exception as exc:
                self.analysis_events.put((False, str(exc)))
        self.experiment_status.set('원시 데이터 분석 중…')
        self.analysis_worker = threading.Thread(target=run, daemon=True)
        self.analysis_worker.start()
