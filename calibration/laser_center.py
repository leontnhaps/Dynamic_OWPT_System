"""Review saved manual laser annotations and summarize selected samples by condition."""
import csv
import json
import math
import statistics
import uuid
from pathlib import Path


def load_samples(directory):
    rows, skipped = [], []
    for marker in sorted(Path(directory).glob('*/complete.json')):
        try:
            meta = json.loads(marker.with_name('metadata.json').read_text(encoding='utf-8'))
            ann = json.loads(marker.with_name('annotations.json').read_text(encoding='utf-8'))
            point = ann['points']['laser']
            size = meta['actual_image_size']
            if list(ann['original_size']) != list(size):
                raise ValueError('annotation image size mismatch')
            u, v = float(point['u']), float(point['v'])
            if not (math.isfinite(u) and math.isfinite(v) and 0 <= u < size[0] and 0 <= v < size[1]):
                raise ValueError('invalid pixel coordinates')
            if point.get('method') != 'manual_click':
                raise ValueError('not a manual annotation')
            m = meta['measurement']
            if m['kind'] != 'laser_reference' or meta.get('simulated') is not False:
                raise ValueError('requires real laser_reference sample')
            if meta.get('gpio_changed_during_capture') or meta.get('laser_gpio_level') != 1:
                raise ValueError('laser state unsuitable')
            angles = meta['servo_at_capture_start']['commanded']
            distance, pan, tilt = [float(x) for x in (m['distance_m'], angles['pan'], angles['tilt'])]
            if not all(map(math.isfinite, (distance, pan, tilt))) or distance <= 0:
                raise ValueError('invalid distance/angles')
            rows.append(dict(sample=marker.parent.name, session=m.get('session_label',''),
                             label=m.get('sample_label',''), distance_m=distance,
                             width=size[0], height=size[1], pan=pan, tilt=tilt,
                             screen=m.get('screen_orientation',''), ir=meta.get('ir_gpio_level'),
                             u=u, v=v))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            skipped.append(f'{marker.parent.name}: {exc}')
    return rows, skipped


def summarize(rows):
    if not rows:
        raise ValueError('평균에 사용할 사진을 선택하세요.')
    groups = {}
    fields = ('session','distance_m','width','height','pan','tilt','screen','ir')
    for row in rows:
        groups.setdefault(tuple(row[k] for k in fields), []).append(row)
    results = []
    for key, samples in groups.items():
        us, vs = [r['u'] for r in samples], [r['v'] for r in samples]
        results.append(dict(zip(fields,key), count=len(samples), u_mean=statistics.mean(us),
                            v_mean=statistics.mean(vs),
                            u_std=statistics.stdev(us) if len(us)>1 else None,
                            v_std=statistics.stdev(vs) if len(vs)>1 else None,
                            sample_ids=[r['sample'] for r in samples]))
    return results


def save_results(directory, rows):
    results = summarize(rows)
    out = Path(directory)/'laser_center_results'/uuid.uuid4().hex
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(dict(
        coordinate_system='original image pixels: u right, v down',
        std_definition='sample standard deviation (ddof=1); null for n=1',
        groups=results, selected_samples=rows), ensure_ascii=False, indent=2, allow_nan=False),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(dict(r,sample_ids=';'.join(r['sample_ids'])) for r in results)
    return out, results


def open_window(parent, directory):
    import tkinter as tk
    from tkinter import ttk, messagebox
    win=tk.Toplevel(parent)
    win.title('레이저 중심 평균 — 저장된 수동 좌표')
    win.geometry('1050x600')
    ttk.Label(win,text='Ctrl/Shift로 사진 선택 → 선택 사진 평균·저장. 거리·해상도·각도·세션·판 기준·IR별로 분리 계산합니다.').pack()
    columns=('label','distance_m','width','height','pan','tilt','u','v','session')
    tree=ttk.Treeview(win,columns=columns,show='headings',selectmode='extended')
    for c in columns:
        tree.heading(c,text=c);tree.column(c,width=100)
    tree.pack(fill='both',expand=True)
    info=tk.StringVar();ttk.Label(win,textvariable=info,wraplength=1000).pack()
    output=tk.Text(win,height=8);output.pack(fill='x')
    rows=[]
    def refresh():
        nonlocal rows
        rows, skipped=load_samples(directory)
        tree.delete(*tree.get_children())
        for i,r in enumerate(rows):tree.insert('', 'end',iid=str(i),values=[r[c] for c in columns])
        info.set(f'사용 가능 {len(rows)}장 / 제외 {len(skipped)}장. 좌표를 수정했다면 목록 새로고침을 누르세요.')
        output.delete('1.0','end');output.insert('end','\n'.join(skipped))
    def save():
        try:
            ids={rows[int(i)]['sample'] for i in tree.selection()}
            fresh,_=load_samples(directory)
            chosen=[r for r in fresh if r['sample'] in ids]
            if len(chosen)!=len(ids):raise ValueError('선택 사진이 변경되었습니다. 목록을 새로고침하세요.')
            path,results=save_results(directory,chosen)
            output.delete('1.0','end')
            for r in results:
                output.insert('end',f"Z={r['distance_m']}m, Pan/Tilt={r['pan']}/{r['tilt']}, n={r['count']}: 평균 ({r['u_mean']:.3f}, {r['v_mean']:.3f}), 표준편차 ({r['u_std']}, {r['v_std']})\n")
            info.set(f'저장 완료: {path}')
        except (ValueError,OSError) as exc:messagebox.showerror('중심 평균',str(exc),parent=win)
    buttons=ttk.Frame(win);buttons.pack()
    ttk.Button(buttons,text='목록 새로고침',command=refresh).pack(side='left')
    ttk.Button(buttons,text='전체 선택',command=lambda:tree.selection_set(tree.get_children())).pack(side='left')
    ttk.Button(buttons,text='선택 사진 평균·저장',command=save).pack(side='left')
    refresh()


if __name__=='__main__':
    import argparse
    import tkinter as tk
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='captures/m2')
    args=parser.parse_args()
    root=tk.Tk();root.withdraw()
    window=open_window(root,args.input)
    # Closing the sole analysis window also ends the standalone process.
    for child in root.winfo_children():child.protocol('WM_DELETE_WINDOW',root.destroy)
    root.mainloop()
