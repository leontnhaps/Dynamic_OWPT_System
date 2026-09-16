"""YOLO + frozen SAC live test tab, asynchronous inference and explicit tracking."""
import csv
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, ImageDraw
from Tx.Controller.live_core import LiveModels, fresh, command_inside


class LivePanel(ttk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,padding=6)
        self.app=app
        self.executor=ThreadPoolExecutor(max_workers=1)
        self.future=None;self.models=None;self.running=False;self.detecting=False
        self.generation=0;self.pending=None;self.previous_error=None
        self.last_frame=None;self.last_submit=0.;self.not_before=0.;self.log=None
        self.paths={};self.widgets=[]
        for row,(name,label) in enumerate([('yolo','YOLO .pt'),('sac','SAC .zip'),('config','SAC config.json')]):
            ttk.Label(self,text=label).grid(row=row,column=0,sticky='w')
            var=tk.StringVar();self.paths[name]=var
            entry=ttk.Entry(self,textvariable=var,width=70);entry.grid(row=row,column=1,columnspan=5,sticky='ew')
            button=ttk.Button(self,text='선택',command=lambda n=name:self.browse(n));button.grid(row=row,column=6)
            self.widgets.extend([entry,button])
        options=ttk.Frame(self);options.grid(row=3,column=0,columnspan=7,sticky='w')
        self.device=tk.StringVar(value='cuda');self.conf=tk.StringVar(value='0.5');self.class_id=tk.StringVar(value='0')
        for label,var in [('장치 cpu/cuda',self.device),('신뢰도',self.conf),('PV class ID',self.class_id)]:
            ttk.Label(options,text=label).pack(side='left');e=ttk.Entry(options,textvariable=var,width=9);e.pack(side='left',padx=4);self.widgets.append(e)
        buttons=ttk.Frame(self);buttons.grid(row=4,column=0,columnspan=7,sticky='w')
        for label,fn in [('모델 불러오기',self.load),('검출 + SAC 출력 보기',self.preview),('추적 시작',self.start),('정지',self.stop)]:
            ttk.Button(buttons,text=label,command=lambda f=fn:self.guard(f)).pack(side='left',padx=4)
        self.status=tk.StringVar(value='모델을 선택하세요. 추적 시작 전에는 이동 명령을 보내지 않습니다.')
        ttk.Label(self,textvariable=self.status,wraplength=1000).grid(row=5,column=0,columnspan=7,sticky='w')
        ttk.Label(self,text='해상도·레이저 기준점은 선택한 config 사용 | 미검출·다중 PV·0.5초 이상 지난 영상은 추적 정지 | 레이저 자동 ON 없음').grid(row=6,column=0,columnspan=7,sticky='w')
        self.canvas=ttk.Label(self);self.canvas.grid(row=7,column=0,columnspan=7)
        self.detail=tk.StringVar();ttk.Label(self,textvariable=self.detail).grid(row=8,column=0,columnspan=7,sticky='w')

    def guard(self,fn):
        try:fn()
        except Exception as exc:
            self.stop(str(exc));messagebox.showerror('실전 테스트',str(exc))

    def browse(self,name):
        path=filedialog.askopenfilename(filetypes=[('Model / config','*.pt *.zip *.json')])
        if path:
            self.paths[name].set(path)
            if name=='sac':
                config=Path(path).with_name('config.json')
                if not config.is_file():config=Path(path).parent.parent/'config.json'
                if config.is_file():self.paths['config'].set(str(config))

    def load(self):
        self.stop()
        if self.future is not None:
            raise ValueError('진행 중인 추론/모델 로딩이 끝난 뒤 다시 시도하세요.')
        self.models=None
        self.load_paths={k:v.get() for k,v in self.paths.items()}
        device=self.device.get().strip()
        if device not in ('cpu','cuda'):raise ValueError('장치는 cpu 또는 cuda')
        for w in self.widgets:w.configure(state='disabled')
        self.future=self.executor.submit(LiveModels,self.load_paths['yolo'],self.load_paths['sac'],self.load_paths['config'],device)
        self.job=('load',self.generation)
        self.status.set('모델 불러오는 중…')

    def settings(self):
        confidence=float(self.conf.get());class_id=int(self.class_id.get())
        if not 0 < confidence <= 1 or class_id < 0:raise ValueError('신뢰도는 0 초과 1 이하, class ID는 0 이상')
        if class_id not in self.models.names:raise ValueError(f'존재하지 않는 class ID: {self.models.names}')
        return confidence,class_id

    def preview(self):
        if self.models is None:raise ValueError('모델부터 불러오세요.')
        self.stop();self.settings();self.detecting=True
        self.status.set('검출 + SAC 출력 확인 중 · 이동 명령 전송 안 함')

    def start(self):
        if self.models is None:raise ValueError('모델부터 불러오세요.')
        self.settings()
        s=self.app.servo
        if not s.online or s.pending or self.pending or s.last is None or not s.limits:
            raise ValueError('Pan / Tilt 탭에서 상태 조회, 운용 범위 적용, 시작 각도 명령을 먼저 완료하세요.')
        c=self.models.cfg
        command=[s.last[a] for a in ('pan','tilt')]
        if not command_inside(command,s.limits) or not all(lo<=v<=hi for lo,v,hi in zip(c.angle_low,command,c.angle_high)):
            raise ValueError('직전 명령이 운용/모델 범위 밖입니다.')
        if any(abs(v-round(v))>1e-6 for v in command):raise ValueError('시작 각도는 1° 단위로 설정하세요.')
        frame=self.app.current
        if not frame or not fresh(frame[2],time.monotonic()):raise ValueError('최근 영상을 먼저 수신하세요.')
        if self.app.size!=(c.width,c.height):raise ValueError('Camera 해상도를 모델 config에 맞추세요.')
        self.stop();self.running=True;self.detecting=True;self.not_before=time.monotonic()
        self.run_settings=self.settings()
        folder=Path('captures/M4/live')/datetime.now().strftime('%Y%m%d_%H%M%S_%f');folder.mkdir(parents=True)
        self.log=(folder/'frames.csv').open('w',newline='',encoding='utf-8')
        self.writer=csv.DictWriter(self.log,fieldnames=['unix_ns','frame_seq','frame_age_s','inference_s','status','error_u','error_v','raw_pan','raw_tilt','delta_pan','delta_tilt','command_pan','command_tilt','request_id'])
        self.writer.writeheader()
        (folder/'session.json').write_text(json.dumps(dict(models=self.load_paths,config=vars(c),servo=s.context(),confidence=self.run_settings[0],class_id=self.run_settings[1],actual_angles_measured=False),ensure_ascii=False,indent=2),encoding='utf-8')
        self.app.record(dict(event='live_start',folder=str(folder)))
        self.status.set('추적 중 · 정지 버튼으로 추가 이동 명령 중단')

    def stop(self,reason='사용자 정지'):
        active=self.running
        self.running=False;self.detecting=False;self.generation+=1;self.previous_error=None
        if self.log:self.log.close();self.log=None
        self.status.set(reason+' · 추가 이동 명령 중단')
        if active:self.app.record(dict(event='live_stop',reason=reason))
        # Already transmitted move may finish; firmware has no verified motion-stop API.

    def event(self,event):
        if self.running and event.get('event')=='servo' and event.get('operation') in ('move','servo_config') and (not self.pending or event.get('request_id')!=self.pending[0]):
            self.stop('다른 제어 명령 감지')
        if self.pending and event.get('request_id')==self.pending[0] and event.get('event') in ('servo','error'):
            self.pending=None
            if event['event']=='error':self.stop(event.get('message','서보 오류'));return
            self.app.servo.last=event.get('commanded')
            self.app.servo.limits=event.get('limits')
            self.not_before=time.monotonic()
            if self.app.servo.last:
                for a in ('pan','tilt'):self.app.servo.vars[a].set(str(self.app.servo.last[a]))
        if self.running and (event.get('event') in ('hello','agent','ready') or
            (event.get('event')=='network' and event.get('state')!='connected')):
            self.stop('연결 상태 변경: 상태 확인 후 다시 시작하세요.')

    def poll(self):
        now=time.monotonic()
        if self.running and (not self.app.current or not fresh(self.app.current[2],now)):
            self.stop('영상 수신 지연')
        if self.pending and now-self.pending[1]>2:
            self.pending=None;self.app.servo.invalidate('실전 명령 응답 시간 초과');self.stop('서보 응답 시간 초과')
        if self.future is not None and self.future.done():
            future=self.future;self.future=None
            try:
                result=future.result()
                if self.job[0]=='load':
                    self.models=result
                    for w in self.widgets:w.configure(state='normal')
                    c=result.cfg
                    self.status.set(f'불러옴: {result.names} | {c.width}×{c.height} | 레이저 ({c.laser_u}, {c.laser_v}) | Δ ±{c.delta_limit_deg}°')
                elif self.job[1]==self.generation:
                    self.consume(result,self.job[2],self.job[3],now)
            except Exception as exc:
                for w in self.widgets:w.configure(state='normal')
                self.stop('실행 오류: '+str(exc))
                self.app.record(dict(event='live_error',message=str(exc)))
        if not self.detecting or self.future is not None or self.pending or self.app.servo.pending:return
        frame=self.app.current
        if not frame or frame[2]==self.last_frame or frame[2]<self.not_before or not fresh(frame[2],now):return
        if now-self.last_submit<self.models.cfg.dt:return
        confidence,class_id=self.run_settings if self.running else self.settings()
        last=self.app.servo.last
        command=[last[a] for a in ('pan','tilt')] if last else [0.,0.]
        self.last_frame=frame[2];self.last_submit=now
        self.job=('infer',self.generation,frame,now)
        self.future=self.executor.submit(self.models.infer,frame[0],command,self.previous_error,confidence,class_id)

    def consume(self,result,frame,started,now):
        if not fresh(frame[2],now):
            self.previous_error=None
            if self.running:self.stop('추론 결과가 0.5초 이상 지연됨')
            else:self.status.set('추론 지연: 오래된 결과 폐기')
            return
        image=result['image'];draw=ImageDraw.Draw(image);c=self.models.cfg
        u,v=c.laser_u,c.laser_v
        draw.line((u-12,v,u+12,v),fill='red',width=3);draw.line((u,v-12,u,v+12),fill='red',width=3)
        target=result['target']
        if target:
            draw.rectangle(target['box'],outline='lime',width=3)
            x,y=target['center'];draw.ellipse((x-4,y-4,x+4,y+4),fill='lime')
        image.thumbnail((800,300));photo=ImageTk.PhotoImage(image);self.canvas.configure(image=photo);self.canvas.image=photo
        if target is None:
            self.previous_error=None
            self.detail.set(f'PV 검출 {result["count"]}개: 단일 PV가 필요합니다.')
            self.write_row(result,frame,started,now,'missing_or_ambiguous','')
            if self.running:self.stop('PV 미검출 또는 다중 검출')
            return
        self.previous_error=result['error']
        self.detail.set(f'오차 {tuple(round(x,1) for x in result["error"])} px | 원래 Δ {tuple(round(x,2) for x in result["raw_delta"])}° | 적용 Δ {result["applied_delta"]}° | 목표 {result["command"]}° | 추론 {(now-started)*1000:.0f} ms')
        token='';status='preview'
        if self.running:
            s=self.app.servo
            if not s.online or s.last is None or not command_inside(result['command'],s.limits):
                self.stop('운용 범위 또는 서보 상태 확인 필요');return
            if any(x!=0 for x in result['applied_delta']):
                token='live-'+uuid.uuid4().hex
                from common.servo import move_from
                move=move_from(dict(s.values(),pan=result['command'][0],tilt=result['command'][1]),s.limits)
                if self.app.send(dict(cmd='move',request_id=token,**move),tracking=True):
                    self.pending=(token,now);status='sent'
                else:self.stop('명령 전송 실패');return
            else:status='hold'
        self.write_row(result,frame,started,now,status,token)

    def write_row(self,r,frame,started,now,status,token):
        if self.log is None:return
        row=dict(unix_ns=time.time_ns(),frame_seq=frame[1].get('seq'),frame_age_s=now-frame[2],inference_s=now-started,status=status,request_id=token)
        for key,fields in [('error',['error_u','error_v']),('raw_delta',['raw_pan','raw_tilt']),('applied_delta',['delta_pan','delta_tilt']),('command',['command_pan','command_tilt'])]:
            if key in r:row.update(zip(fields,r[key]))
        self.writer.writerow(row);self.log.flush()

    def close(self):
        self.stop('창 닫힘');self.executor.shutdown(wait=False,cancel_futures=True)
