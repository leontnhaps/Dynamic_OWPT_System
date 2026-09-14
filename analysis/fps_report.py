"""Offline M1/M2 JSONL receive-statistics report (no camera/network required)."""
import argparse
import csv
import json
import math
import uuid
from pathlib import Path


def analyze(events, warmup=5):
    if not math.isfinite(warmup) or warmup < 0:
        raise ValueError('warmup must be finite and nonnegative')
    results=[]
    cfg=None
    start=None
    previous=None
    intervals=[]
    def finish():
        nonlocal intervals
        if intervals:
            seconds=sum(x[0] for x in intervals)
            frames=sum(x[1] for x in intervals)
            rates=[n/t for t,n in intervals]
            results.append(dict(segment=len(results)+1, **{k:(cfg or {}).get(k) for k in
                ('width','height','quality','fps','shutter_speed','analogue_gain')},
                measured_seconds=seconds, received_frames=frames, mean_receive_fps=frames/seconds,
                min_interval_fps=min(rates), max_interval_fps=max(rates),
                zero_frame_seconds=sum(t for t,n in intervals if n==0), intervals=len(intervals)))
        intervals=[]
    for e in events:
        kind=e.get('event')
        t=e.get('gui_unix_ns')
        if not isinstance(t,(int,float)) or not math.isfinite(t):
            continue
        t=t/1e9
        if kind=='command' and e.get('command',{}).get('cmd') in ('preview','snap','outputs_off'):
            finish();cfg=None;start=None;previous=None
        elif kind=='preview':
            finish();previous=None
            cfg=e.get('requested') if e.get('state')=='started' else None
            start=t if cfg else None
        elif kind in ('ready','network','agent','hello'):
            # Never count disconnected time or cross a reconnect in one interval.
            finish();previous=None;cfg=None;start=None
        elif kind=='error' and e.get('operation') in ('preview','snap'):
            finish();previous=None;cfg=None;start=None
        elif kind=='receive_stats' and cfg and start is not None:
            count=e.get('total')
            if not isinstance(count,int) or isinstance(count,bool) or count<0:
                finish();previous=None;continue
            if previous:
                pt,pc=previous
                if t<=pt or count<pc:
                    finish();start=t
                elif pt>=start+warmup:
                    intervals.append((t-pt,count-pc))
            previous=(t,count)
    finish()
    return results


def report(path, warmup=5):
    events=[];bad=0
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        try:
            e=json.loads(line)
            if not isinstance(e,dict):raise ValueError('not an object')
            events.append(e)
        except ValueError:bad+=1
    rows=analyze(events,warmup)
    if not rows:
        raise ValueError('분석 가능한 구간이 없습니다. preview started와 충분한 receive_stats가 있는 events JSONL을 선택하세요.')
    out=Path(path).parent/'fps_reports'/uuid.uuid4().hex
    out.mkdir(parents=True)
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(source=str(Path(path).resolve()),warmup_seconds=warmup,
        malformed_lines=bad, method='sum counter differences / sum GUI wall-clock interval durations; settings are requested values',
        segments=rows),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=[f"구간 {r['segment']}: {r['width']}×{r['height']}, quality {r['quality']}, 목표 {r['fps']} FPS\n"
           f"  평균 {r['mean_receive_fps']:.2f} FPS / {r['measured_seconds']:.1f}초 / {r['received_frames']}장\n"
           f"  구간 최저~최고 {r['min_interval_fps']:.2f}~{r['max_interval_fps']:.2f} FPS / 수신 0장 구간 {r['zero_frame_seconds']:.1f}초"
           for r in rows]
    return '\n\n'.join(lines)+f'\n\n잘못된 로그 줄: {bad}\n저장: {out}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',nargs='?',help='events_*.jsonl; omit to open file chooser')
    parser.add_argument('--warmup',type=float,default=5,help='seconds excluded after preview start (default 5)')
    args=parser.parse_args()
    if args.log:
        try:print(report(args.log,args.warmup))
        except (OSError,ValueError) as exc:parser.error(str(exc))
        return
    import tkinter as tk
    from tkinter import filedialog,messagebox
    from tkinter.scrolledtext import ScrolledText
    root=tk.Tk();root.withdraw()
    path=filedialog.askopenfilename(title='FPS 분석할 events 로그 선택',initialdir='captures/m2',filetypes=[('이벤트 로그','*.jsonl')])
    if not path:root.destroy();return
    try:result=report(path,args.warmup)
    except (OSError,ValueError) as exc:
        messagebox.showerror('FPS 분석',str(exc));root.destroy();return
    root.deiconify();root.title('수신 FPS 분석');root.geometry('850x550')
    text=ScrolledText(root,wrap='word');text.pack(fill='both',expand=True)
    text.insert('1.0',result);text.configure(state='disabled');root.mainloop()


if __name__=='__main__':main()
