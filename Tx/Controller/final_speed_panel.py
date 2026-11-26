"""Pi speed tests in the final tab, using the same SSH connection and OOK trial engine."""
import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

from common.ook_experiment import PI_LIMIT_PERIODS, PI_LIMIT_SAMPLE_MS, PI_RECEIVER, PI_TRIALS_PER_RATE
from Tx.Controller.final_speed_results import SPEED_COLUMNS, load_speed_results


class FinalSpeedPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text='Pi + INA219 · OOK 수신 속도', padding=4)
        self.panel = parent
        self.engine = parent.app.ook
        self.running = False
        self.last_root = self.shown = None
        self.periods = tk.StringVar(self, value=','.join(map(str, PI_LIMIT_PERIODS)))
        self.sample_ms = tk.StringVar(self, value=str(PI_LIMIT_SAMPLE_MS))
        row = ttk.Frame(self)
        row.pack(anchor='w')
        ttk.Button(row, text='기존 7속도 × 30회', command=lambda: self.start('reference')).pack(side='left')
        ttk.Label(row, text='한계 탐색 Tx (ms/bit)').pack(side='left', padx=(10, 3))
        ttk.Entry(row, textvariable=self.periods, width=17).pack(side='left')
        ttk.Label(row, text='Rx 표본 간격 (ms)').pack(side='left', padx=(6, 3))
        ttk.Entry(row, textvariable=self.sample_ms, width=4).pack(side='left')
        ttk.Button(row, text=f'한계 측정 × {PI_TRIALS_PER_RATE}회',
                   command=lambda: self.start('limit')).pack(side='left', padx=3)
        ttk.Button(row, text='속도 측정 중지', command=self.stop).pack(side='left')
        ttk.Button(row, text='저장 결과 열기', command=self.choose).pack(side='left', padx=3)
        self.graph_path = None
        self.graph_button = ttk.Button(row, text='그래프 보기', state='disabled', command=self.graph)
        self.graph_button.pack(side='left')
        self.progress = tk.StringVar(self, value='기존 7속도는 Rx 5 ms 고정 · 한계 측정은 입력한 Tx/Rx 간격 사용')
        ttk.Label(self, textvariable=self.progress, wraplength=1080).pack(anchor='w')
        self.headline = tk.StringVar(self)
        ttk.Label(self, textvariable=self.headline, wraplength=1080).pack(anchor='w')
        self.grid_frame = ttk.Frame(self)
        self.table = ttk.Treeview(self.grid_frame, columns=SPEED_COLUMNS, show='headings', height=4)
        for column in SPEED_COLUMNS:
            self.table.heading(column, text=column)
            self.table.column(column, width=90, stretch=True, anchor='center')
        scroll = ttk.Scrollbar(self.grid_frame, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.table.pack(side='left', expand=True, fill='x')

    def start(self, profile):
        p, e = self.panel, self.engine
        controls = p.tx_controls
        if (self.running or p.phase or p.off_token or p.method or p.wave is not None or e.busy or e.pending
                or (controls and (controls.busy or controls.owns_output))):
            self.progress.set('현재 동작 또는 OFF 확인이 끝난 후 속도 측정을 시작하세요.')
            return
        try:
            p.ready_rx()
            e.check_ready()
            if e.hardware.get('simulated') is not False:
                raise ValueError('실제 Tx가 연결되어야 합니다.')
            if e.arduino is not None and e.arduino is not p.rx and e.arduino.worker.is_alive():
                raise ValueError('이전 수신기 연결 해제 중입니다. 잠시 후 다시 시작하세요.')
            previous = e.arduino
            # Only one consumer reads this SSH queue. While lent to OokPanel,
            # PointComparisonPanel skips its per-tick demo/voltage receiver poll.
            e.arduino = p.rx
            try:
                note = f'최종 탭 · {p.station["point_id"]}' if p.station else '최종 탭 · 독립 속도 측정'
                if profile == 'reference':
                    e.start_pi_comparison(note=note)
                else:
                    e.start_pi_limit(self.periods.get(), self.sample_ms.get(), note=note)
            except Exception:
                e.arduino = previous
                raise
        except (ValueError, OSError) as exc:
            self.progress.set(str(exc))
            return
        self.running = True
        self.last_root = e.experiment_root
        p.phase = 'speed'
        self.headline.set('')
        self.grid_frame.pack_forget()
        self.graph_button.configure(state='disabled')
        self.progress.set(e.experiment_status.get())

    def release(self):
        if self.running:
            if self.engine.arduino is self.panel.rx:
                self.engine.arduino = None
            self.running = False
            self.panel.phase = None

    def stop(self):
        if not self.running:
            return
        self.engine.off()
        self.progress.set(self.engine.experiment_status.get())

    def tick(self):
        e = self.engine
        if self.running:
            msg = e.experiment_status.get()
            if self.progress.get() != msg:
                self.progress.set(msg)
            if e.experiment is None and e.capture is None and e.pending is None:
                self.release()
                if e.output_requested:
                    # An interrupted legacy run can time out before OFF is confirmed.
                    self.panel.off_token = 'm6-speed-' + self.last_root.name + '-stop'
                    self.panel.stop('속도 측정 종료 · Tx OFF 재확인')
        if self.last_root is None:
            return
        analysis = e.last_analysis
        if analysis is not None and analysis.parent == self.last_root and analysis != self.shown:
            self.shown = analysis
            self.display(analysis)
        elif not self.running and self.last_root in (e.last_experiment, getattr(e, 'analysis_root', None)) and analysis is None:
            # Analysis happens after the last acquisition, including when it fails.
            msg = e.experiment_status.get()
            if self.progress.get() != msg:
                self.progress.set(msg)

    def display(self, directory):
        try:
            headline, rows = load_speed_results(directory)
        except (OSError, ValueError) as exc:
            self.headline.set(str(exc))
            self.grid_frame.pack_forget()
            self.graph_button.configure(state='disabled')
            return
        self.table.delete(*self.table.get_children())
        for row in rows:
            self.table.insert('', 'end', values=row)
        self.headline.set(headline)
        self.progress.set(f'분석 결과 · {directory}')
        self.grid_frame.pack(fill='x')
        self.graph_path = Path(directory) / 'ook_bitrate_results.png'
        self.graph_button.configure(state='normal' if self.graph_path.is_file() else 'disabled')

    def graph(self):
        try:
            with Image.open(self.graph_path) as chart:
                chart.thumbnail((self.winfo_screenwidth() * 0.85, self.winfo_screenheight() * 0.8),
                                Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(chart, master=self)
        except OSError as exc:
            self.headline.set(str(exc))
            return
        view = tk.Toplevel(self)
        view.title('Pi OOK 속도 측정 결과')
        label = ttk.Label(view, image=photo)
        label.image = photo
        label.pack()

    def choose(self):
        folder = filedialog.askdirectory(parent=self, title='Pi 속도 실험 또는 분석 폴더 선택',
                                         initialdir=self.last_root or self.panel.app.stage_dir('M6-4'))
        if not folder:
            return
        path = Path(folder)
        if (path / 'experiment.json').is_file():
            analysis = sorted(p for p in path.glob('analysis_*') if (p/'summary.csv').is_file())
            if not analysis:
                if self.running or self.panel.phase:
                    self.headline.set('현재 측정이 끝난 후 저장된 실험을 분석하세요.')
                    return
                try:
                    manifest = json.loads((path / 'experiment.json').read_text(encoding='utf-8-sig'))
                    if not isinstance(manifest, dict) or manifest.get('receiver') != PI_RECEIVER:
                        raise ValueError('Pi + INA219 속도 실험 폴더를 선택하세요.')
                    self.engine.start_analysis(path)
                except (OSError, ValueError) as exc:
                    self.headline.set(str(exc))
                    return
                self.last_root, self.shown = path, None
                self.progress.set('저장된 실험 분석 중…')
                return
            path = analysis[-1]
        self.display(path)
