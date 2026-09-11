"""Dynamic OWPT M1-1: evolving laptop GUI derived from DLC preview responsibilities."""
import argparse
import io
import json
import queue
import time
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk,messagebox
from PIL import Image,ImageTk
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Tx.Controller.network_client import Network
from Tx.Controller.servo_panel import ServoPanel

class App:
    def __init__(self,root,args):
        self.root=root;self.out=Path(args.output);self.current=None
        self.net=Network(args.server);self.links={};self.pings={}
        self.mark=time.monotonic();self.previous=0;self.fps=0;self.size=None
        self.out.mkdir(parents=True,exist_ok=True)
        self.logpath=self.out/('events_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
        root.title('Dynamic OWPT — M1-1 / M1-2');root.geometry('1150x850')
        root.protocol('WM_DELETE_WINDOW',self.close)
        tabs=ttk.Notebook(root);tabs.pack(fill='x')
        camera_tab=ttk.Frame(tabs);tabs.add(camera_tab,text='M1-1 Camera')
        self.servo=ServoPanel(tabs,self);tabs.add(self.servo,text='M1-2 Pan / Tilt')
        row=ttk.Frame(camera_tab,padding=10);row.pack(fill='x');self.values={}
        for i,(name,value) in enumerate([('width','640'),('height','480'),('fps','10'),('quality','80'),('shutter_speed',''),('analogue_gain','')]):
            ttk.Label(row,text=name).grid(row=0,column=i)
            var=tk.StringVar(value=value);self.values[name]=var
            ttk.Entry(row,textvariable=var,width=14).grid(row=1,column=i)
        ttk.Label(camera_tab,text='노출(µs)·gain 모두 공란 = 자동 / fps = 전송 목표 상한').pack()
        row=ttk.Frame(camera_tab,padding=8);row.pack(fill='x')
        for label,fn in [('Start / Apply',self.start),('Stop',lambda:self.send(dict(cmd='preview',enable=False))),
                         ('Save JPEG + JSON',self.save),('IR LOW',lambda:self.send(dict(cmd='ir_cut',level=0))),
                         ('IR HIGH',lambda:self.send(dict(cmd='ir_cut',level=1))),('Ping RTT',self.ping)]:
            ttk.Button(row,text=label,command=fn).pack(side='left',padx=3)
        self.connection=tk.StringVar(value='Connecting…');ttk.Label(root,textvariable=self.connection).pack()
        self.status=tk.StringVar(value='Waiting for frames');ttk.Label(root,textvariable=self.status).pack()
        logs=ttk.Panedwindow(root,orient='horizontal')
        logs.pack(side='bottom',fill='x',padx=6,pady=6)
        self.event_log=self.make_log(logs,'명령 · 응답 · 연결 · 오류')
        self.stats_log=self.make_log(logs,'실시간 수신 통계 (receive_stats)')
        self.preview=ttk.Label(root,anchor='center');self.preview.pack(expand=True,fill='both')
        root.after(30,self.poll)
    def make_log(self,parent,title):
        panel=ttk.LabelFrame(parent,text=title,padding=4)
        parent.add(panel,weight=1)
        follow=tk.BooleanVar(value=True)
        ttk.Checkbutton(panel,text='자동 스크롤',variable=follow).pack(anchor='w')
        body=ttk.Frame(panel);body.pack(fill='both',expand=True)
        widget=tk.Text(body,height=8,width=45,wrap='word',state='disabled')
        scroll=ttk.Scrollbar(body,orient='vertical',command=widget.yview)
        widget.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');widget.pack(fill='both',expand=True)
        return widget,follow

    def record(self,event):
        event=dict(event,gui_unix_ns=time.time_ns())
        line=json.dumps(event,ensure_ascii=False)+'\n'
        # Preserve the complete chronological evidence file, regardless of UI routing.
        with self.logpath.open('a',encoding='utf-8') as f:f.write(line)
        widget,follow=self.stats_log if event.get('event')=='receive_stats' else self.event_log
        widget.configure(state='normal')
        widget.insert('end',line)
        if int(widget.index('end-1c').split('.')[0])>500:widget.delete('1.0','101.0')
        if follow.get():widget.see('end')
        widget.configure(state='disabled')
    def send(self,cmd):
        try:
            self.net.send(cmd);self.record(dict(event='command',command=cmd));return True
        except OSError as exc:
            self.record(dict(event='send_error',command=cmd,message=str(exc)))
            messagebox.showerror('Connection',str(exc));return False
    def start(self):
        try:
            cfg={k:int(self.values[k].get()) for k in ('width','height','fps','quality')}
            exposure=self.values['shutter_speed'].get().strip();gain=self.values['analogue_gain'].get().strip()
            if bool(exposure)!=bool(gain):raise ValueError('수동 노출은 exposure와 gain 모두 입력')
            cfg.update(shutter_speed=int(exposure) if exposure else None,analogue_gain=float(gain) if gain else None)
            self.send(dict(cmd='preview',enable=True,**cfg))
        except ValueError as exc:messagebox.showerror('Settings',str(exc))
    def ping(self):
        token=str(time.monotonic_ns());self.pings={token:time.monotonic()}
        self.send(dict(cmd='ping',token=token))
    def save(self):
        if not self.current or time.monotonic()-self.current[2]>2:
            messagebox.showerror('Save','최근 수신 프레임이 없습니다.');return
        data,meta,received,wall=self.current
        name='frame_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path=self.out/(name+'.jpg');path.write_bytes(data)
        meta=dict(meta,actual_image_size=self.size,gui_receive_unix_ns=wall,gui_receive_fps=self.fps,
                  m1_2=self.servo.context(),
                  note='requested controls and servo commands are not measurements; clocks are not synchronized')
        path.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        self.record(dict(event='saved',file=str(path),m1_2=self.servo.context()))
    def poll(self):
        try:
            for _ in range(100):
                event=self.net.events.get_nowait();kind=event.get('event')
                self.servo.event(event)
                if kind=='network':self.links[str(event['port'])]=event['state']
                if kind in ('hello','agent'):self.links['Pi']=event.get('agent_state',event.get('state'))
                if kind=='pong':
                    mark=self.pings.pop(event.get('token'),None)
                    if mark is not None:event['app_roundtrip_ms']=(time.monotonic()-mark)*1000
                self.connection.set(' | '.join(f'{k}: {v}' for k,v in self.links.items()))
                self.record(event)
        except queue.Empty:pass
        self.servo.tick()
        frame,count=self.net.pop();now=time.monotonic()
        if now-self.mark>=1:
            self.fps=(count-self.previous)/(now-self.mark);self.previous=count;self.mark=now
            self.record(dict(event='receive_stats',fps=self.fps,total=count))
        if frame:
            try:
                image=Image.open(io.BytesIO(frame[0]));self.size=image.size
                image.thumbnail((1000,480),Image.Resampling.LANCZOS)
                photo=ImageTk.PhotoImage(image);self.preview.configure(image=photo);self.preview.image=photo
                self.current=frame
            except (ValueError,OSError) as exc:self.record(dict(event='decode_error',message=str(exc)))
        if self.current:
            age=now-self.current[2];meta=self.current[1]
            self.status.set(f'{"SIMULATION | " if meta["simulated"] else ""}Frame {meta["seq"]} | {self.size} | receive {self.fps:.1f} fps | last receive {age:.2f}s ago'+(' — STALE / STOPPED' if age>2 else ''))
        self.root.after(30,self.poll)
    def close(self):
        try:self.net.send(dict(cmd='preview',enable=False))
        except OSError:pass
        self.net.close();self.root.destroy()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--server',default='127.0.0.1');parser.add_argument('--output',default='captures/m1_2')
    args=parser.parse_args();root=tk.Tk();App(root,args);root.mainloop()
if __name__=='__main__':main()
