"""Static calibration acquisition and manual full-image pixel annotations."""
import io
import json
import queue
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from common.capture import capture_request, save_capture, append_index, export_catalog


class CapturePanel(ttk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,padding=8)
        self.app=app
        self.pending=None
        self.last_path=None
        self.vars={}
        self.state=tk.StringVar(value='촬영: 대기')
        self.hardware=tk.StringVar(value='IR / Laser: 상태 조회 필요')
        self.awb=tk.BooleanVar(value=False)
        rows=[
            [('width','Width','1920'),('height','Height','1080'),('quality','JPEG quality','95'),
             ('shutter_speed','Exposure µs','1000'),('analogue_gain','Gain','1')],
            [('distance_m','정면 Z 거리(m)',''),('target_x_m','표적 X(m), 우+','0'),
             ('target_y_m','표적 Y(m), 위+','0'),('session_label','세션 이름','calibration'),('sample_label','표본/쌍 ID','')],
            [('screen_orientation','판 방향/기준',''),('note','메모','')]]
        for r,fields in enumerate(rows):
            row=ttk.Frame(self);row.pack(fill='x',pady=2)
            for key,label,value in fields:
                frame=ttk.Frame(row);frame.pack(side='left',padx=3)
                ttk.Label(frame,text=label).pack(anchor='w')
                var=tk.StringVar(value=value);self.vars[key]=var
                ttk.Entry(frame,textvariable=var,width=18 if key!='note' else 50).pack()
        row=ttk.Frame(self);row.pack(fill='x',pady=4)
        self.kind=tk.StringVar(value='laser_reference')
        ttk.Label(row,text='사진 용도').pack(side='left')
        ttk.Combobox(row,textvariable=self.kind,state='readonly',width=18,
                     values=['laser_reference','background','target']).pack(side='left',padx=4)
        ttk.Checkbutton(row,text='자동 화이트밸런스 (OFF: gains 1,1)',variable=self.awb).pack(side='left')
        ttk.Label(row,text='노출·gain 둘 다 공란: 자동 노출').pack(side='left',padx=10)
        row=ttk.Frame(self);row.pack(fill='x',pady=4)
        ttk.Button(row,text='이 설정으로 미리보기',command=lambda:self.guard(self.preview)).pack(side='left',padx=3)
        self.button=ttk.Button(row,text='Pi에서 새 사진 촬영',command=lambda:self.guard(self.snap));self.button.pack(side='left')
        ttk.Button(row,text='마지막 촬영 사진 · 중심 표시',command=lambda:self.guard(self.annotate)).pack(side='left',padx=4)
        ttk.Button(row,text='기존 촬영 사진 열기',command=lambda:self.guard(self.open_image)).pack(side='left',padx=3)
        ttk.Button(row,text='좌표 포함 CSV 내보내기',command=lambda:self.guard(self.export)).pack(side='left',padx=3)
        ttk.Label(self,textvariable=self.state).pack(anchor='w')
        row=ttk.Frame(self);row.pack(fill='x',pady=4)
        for label,cmd in [('IR 통과 ON (night)',dict(cmd='ir_cut',level=1)),
                          ('IR 통과 OFF (day)',dict(cmd='ir_cut',level=0)),
                          ('Laser ON',dict(cmd='laser',value=1)),('Laser OFF',dict(cmd='laser',value=0)),
                          ('상태 조회',dict(cmd='hardware_status'))]:
            ttk.Button(row,text=label,command=lambda c=cmd:self.app.send(dict(c,request_id=uuid.uuid4().hex))).pack(side='left',padx=3)
        ttk.Label(self,textvariable=self.hardware).pack(anchor='w')
        ttk.Label(self,text='IR 버튼은 IR-CUT 필터 전환이며 조명 LED와 별개입니다. GPIO 상태는 광출력 측정값이 아닙니다.\n'
                  'Pan/Tilt 탭에서 운용 범위 적용 → 각도 이동 후 촬영. 사진 중 변경은 제한되며 Laser OFF는 항상 가능합니다.\n'
                  '거리/좌표는 기준 자세에서 고정한 Tx 좌표계의 수동 측정값. 원본 JPEG와 촬영 조건 저장.').pack(anchor='w')

    def guard(self,fn):
        try: fn()
        except (ValueError,TypeError,OSError,KeyError) as exc:
            messagebox.showerror('M2 Capture',str(exc))

    def preview(self):
        from common.capture import camera_config
        values={k:v.get().strip() for k,v in self.vars.items()}
        cfg=camera_config(dict(values,shutter_speed=values['shutter_speed'] or None,
                               analogue_gain=values['analogue_gain'] or None,awb=self.awb.get()))
        self.app.send(dict(cmd='preview',enable=True,**cfg))

    def open_image(self):
        path=filedialog.askopenfilename(initialdir=str(self.app.out),filetypes=[('JPEG','*.jpg')])
        if path:
            path=Path(path)
            if not path.with_name('complete.json').exists():
                raise ValueError('M2 촬영 폴더의 image.jpg를 선택하세요.')
            AnnotationWindow(self.app.root,path)

    def export(self):
        path,count=export_catalog(self.app.out)
        self.state.set(f'{count}개 표본 CSV 저장: {path}')

    def snap(self):
        if self.pending: raise ValueError('이전 촬영의 수신을 기다리고 있습니다.')
        if not self.app.servo.online: raise ValueError('Pi 연결을 확인하세요.')
        if self.app.links.get('7601')!='connected': raise ValueError('영상 채널 연결을 확인하세요.')
        if self.app.servo.pending: raise ValueError('Pan/Tilt 응답을 기다리세요.')
        values={k:v.get().strip() for k,v in self.vars.items()}
        exposure,gain=values['shutter_speed'],values['analogue_gain']
        token=uuid.uuid4().hex
        cmd=dict(cmd='snap',request_id=token,width=values['width'],height=values['height'],quality=values['quality'],
                 shutter_speed=exposure or None,analogue_gain=gain or None,awb=self.awb.get(),
                 measurement=dict(values,kind=self.kind.get()))
        capture_request(cmd)
        if self.app.send(cmd):
            self.pending=(token,time.monotonic())
            self.button.state(['disabled'])
            self.state.set('촬영·전송 대기…')

    def event(self,event):
        kind=event.get('event')
        if kind in ('ready','hello') or (kind=='agent' and event.get('state')=='connected'):
            self.app.send(dict(cmd='hardware_status'))
        if kind in ('hardware','laser','ir_cut','outputs_off'):
            self.hardware.set(('SIMULATION | ' if event.get('simulated') else '')+
                              f"IR GPIO: {event.get('ir_gpio_level')} | Laser GPIO: {event.get('laser_gpio_level')} (명령 상태)")
        if kind in ('network','agent') and event.get('state')=='disconnected':
            self.hardware.set('연결 끊김 · 상태 미확인')
        if kind=='error' and self.pending and event.get('request_id')==self.pending[0]:
            self.finish('촬영 실패: '+event.get('message',''))

    def finish(self,text):
        self.pending=None
        self.button.state(['!disabled'])
        self.state.set(text)

    def poll(self):
        if self.pending and time.monotonic()-self.pending[1]>30:
            self.finish('응답 시간 초과 · Pi 원본 확인 (자동 재촬영 없음)')
        try:
            for _ in range(8):
                data,meta,received,wall=self.app.net.captures.get_nowait()
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        size=list(image.size)
                    meta=dict(meta,actual_image_size=size,gui_receive_unix_ns=wall)
                    path,meta=save_capture(self.app.out,data,meta)
                    self.last_path=path
                    # Index failure must not misreport an already committed sample as lost.
                    try: append_index(self.app.out,path,meta)
                    except OSError as exc: self.app.record(dict(event='index_error',file=str(path),message=str(exc)))
                    self.app.record(dict(event='capture_saved',request_id=meta['request_id'],file=str(path),actual_image_size=size))
                    if self.pending is None or self.pending[0]==meta['request_id']:
                        self.finish(f'저장 완료: {meta["request_id"][:8]} | {size[0]}×{size[1]}')
                except (OSError,ValueError) as exc:
                    self.app.record(dict(event='capture_save_error',request_id=meta.get('request_id'),message=str(exc)))
                    if self.pending is None or self.pending[0]==meta.get('request_id'):
                        self.finish('저장 실패 · Pi 원본 확인: '+str(exc))
        except queue.Empty: pass

    def annotate(self):
        if not self.last_path: raise ValueError('먼저 사진을 촬영하세요.')
        AnnotationWindow(self.app.root,self.last_path)


class AnnotationWindow(tk.Toplevel):
    """Annotations belong to a frozen captured image, never the live preview."""
    def __init__(self,parent,path):
        super().__init__(parent)
        self.path=Path(path)
        self.title('촬영 사진 중심 좌표 — 수동 클릭 / 원본 pixel')
        self.points={}
        self.kind=tk.StringVar(value='laser')
        self.info=tk.StringVar(value='표시할 종류 선택 후 사진을 클릭하세요. 좌측 상단 원점, u 우+, v 아래+.')
        row=ttk.Frame(self);row.pack(fill='x')
        for key,label in [('laser','레이저 중심'),('target','표적 중심')]:
            ttk.Radiobutton(row,text=label,value=key,variable=self.kind).pack(side='left')
        ttk.Button(row,text='좌표 저장',command=self.save).pack(side='left')
        ttk.Label(self,textvariable=self.info).pack()
        with Image.open(path) as original:
            self.original_size=original.size
            image=original.copy()
        image.thumbnail((1000,650),Image.Resampling.LANCZOS)
        self.display_size=image.size
        self.photo=ImageTk.PhotoImage(image)
        self.canvas=tk.Canvas(self,width=image.width,height=image.height,highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.canvas.bind('<Button-1>',self.click_point)
        annotations=self.path.with_name('annotations.json')
        if annotations.exists():
            self.points=json.loads(annotations.read_text(encoding='utf-8')).get('points',{})
            self.info.set(str(self.points))

    def click_point(self,event):
        dw,dh=self.display_size
        if not (0<=event.x<dw and 0<=event.y<dh): return
        w,h=self.original_size
        # Convert display pixel centers, preserving the original image's coordinates.
        u=max(0,min(w-1,(event.x+.5)*w/dw-.5))
        v=max(0,min(h-1,(event.y+.5)*h/dh-.5))
        kind=self.kind.get()
        self.points[kind]=dict(u=round(u,3),v=round(v,3),method='manual_click')
        self.canvas.delete(kind)
        self.canvas.create_oval(event.x-5,event.y-5,event.x+5,event.y+5,outline='red' if kind=='laser' else 'lime',tags=kind)
        self.info.set(str(self.points))

    def save(self):
        if not self.points: return
        path=self.path.with_name('annotations.json')
        data=dict(image_file='image.jpg',original_size=self.original_size,points=self.points,
                  coordinate_system='full_image_pixels_top_left_origin',saved_unix_ns=time.time_ns())
        try:
            temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(path)
            self.info.set('annotations.json 저장 완료 · '+str(self.points))
        except OSError as exc: messagebox.showerror('저장 실패',str(exc))
