"""M1-5 visual response measurement using the existing PV detector."""
import csv
import json
import math
import statistics
import time
import uuid
from datetime import datetime
from tkinter import ttk
import tkinter as tk
from Tx.Controller.detection_panel import DetectionPanel
from common.servo import move_from


def analyze(samples, command_time, baseline, hold, floor):
    before = [s for s in samples if command_time-baseline <= s['receive'] < command_time]
    after = [s for s in samples if s['receive'] >= command_time]
    invalid = lambda s: s['status'] != 'detected' or s['count'] != 1
    if len(before) < 5 or len(after) < 5 or any(invalid(s) for s in before+after):
        return {'status': 'invalid_detection', 'onset_s': None, 'settled_s': None}
    center = [statistics.median(s[a] for s in before) for a in ('u', 'v')]
    noise = max(math.hypot(s['u']-center[0], s['v']-center[1]) for s in before)
    threshold = max(floor, 3*noise)
    times = [s['receive'] for s in before+after]
    gap = max(b-a for a,b in zip(times,times[1:]))
    endpoint = [s for s in after if s['receive'] >= after[-1]['receive']-hold]
    final = [statistics.median(s[a] for s in endpoint) for a in ('u','v')]
    displacement = math.hypot(final[0]-center[0], final[1]-center[1])
    result = dict(status='unresolved', onset_s=None, settled_s=None,
                  threshold_px=threshold, displacement_px=displacement,
                  max_sample_gap_s=gap, delta_u=final[0]-center[0], delta_v=final[1]-center[1])
    if gap > hold/2:
        result['status']='insufficient_sampling'; return result
    if displacement <= 2*threshold:
        result['status']='movement_too_small'; return result
    for i,s in enumerate(after[:-1]):
        if all(math.hypot(x['u']-center[0],x['v']-center[1]) > threshold for x in after[i:i+2]):
            result['onset_s']=s['receive']-command_time
            result['onset_previous_s']=(after[i-1]['receive'] if i else before[-1]['receive'])-command_time
            break
    for i,s in enumerate(after):
        tail=after[i:]
        if tail[-1]['receive']-s['receive'] >= hold and all(
                math.hypot(x['u']-final[0],x['v']-final[1]) <= threshold for x in tail):
            result['settled_s']=s['receive']-command_time; break
    if result['onset_s'] is not None and result['settled_s'] is not None:
        result['status']='valid'
    return result


class TimingPanel(DetectionPanel):
    def __init__(self,parent,app):
        super().__init__(parent,app)
        self.preparing=None; self.auto_measure=False; self.setup_pending=None
        self.active=False; self.samples=[]; self.results=[]; self.pending=None
        self.fields={}
        row=ttk.Frame(self); row.pack(fill='x',before=self.canvas)
        for label,key,value in [('이동각°','steps','1,3,5'),('반복','repeats','10'),
                ('기준 기록초','baseline','2'),('이동 관측초','window','3'),
                ('안정 유지초','hold','0.5'),('최소 문턱px','floor','2')]:
            # Labels are localized below for the existing Korean GUI.
            labels={'steps':'이동각°','repeats':'반복','baseline':'기준 기록초',
                    'window':'이동 관측초','hold':'안정 유지초','floor':'최소 문턱px'}
            ttk.Label(row,text=labels[key]).pack(side='left')
            var=tk.StringVar(value=value);self.fields[key]=var
            ttk.Entry(row,textvariable=var,width=7).pack(side='left')
        row=ttk.Frame(self);row.pack(fill='x',before=self.canvas)
        ttk.Button(row,text='자동 측정 시작',command=lambda:self.guard(self.begin)).pack(side='left')
        ttk.Button(row,text='측정 중단',command=self.abort).pack(side='left')
        self.measure_status=tk.StringVar(value='검출 시작 / 자동 측정 시작 → 카메라·서보·모델 자동 준비')
        ttk.Label(row,textvariable=self.measure_status,wraplength=1000).pack(side='left')

    def start(self):
        self.prepare(False)

    def prepare(self, auto_measure):
        if self.active or self.preparing:
            raise ValueError('측정 또는 자동 준비가 진행 중입니다.')
        self.app.live.stop('M1-5 자동 준비')
        self.app.detection.stop()
        if self.app.live.pending or self.app.servo.pending:
            raise ValueError('기존 명령 응답을 기다린 뒤 시작하세요.')
        if not self.app.servo.online:
            raise ValueError('Pi 제어 연결이 없습니다. 서버·Pi 실행 상태를 확인하세요.')
        if self.model is None and self.future is None:
            self.load()
        elif self.future is not None and self.job[0] != 'load':
            raise ValueError('검출 작업이 끝난 뒤 시작하세요.')
        self.auto_measure=auto_measure
        self.preparing='model'; self.setup_pending=None
        self.setup_deadline=time.monotonic()+120
        self.status.set('M1-5 자동 준비: 모델 로딩 대기')

    def setup_command(self, command, next_phase):
        token=uuid.uuid4().hex
        if not self.app.send(dict(command, request_id=token),tracking=True):
            raise ValueError('자동 준비 명령 전송 실패')
        self.setup_pending=token;self.preparing=next_phase
        self.setup_deadline=time.monotonic()+5

    def prepare_poll(self):
        now=time.monotonic()
        if now>self.setup_deadline:
            raise ValueError('자동 준비 시간 초과: '+self.preparing)
        if self.preparing=='model':
            if self.model is None:return
            # Apply the settings established in M1-1 and M1-2.
            camera=dict(width=1296,height=972,fps=30,quality=80,shutter_speed=None,analogue_gain=None)
            for key,value in camera.items():self.app.values[key].set('' if value is None else str(value))
            if not self.app.send(dict(cmd='preview',enable=True,**camera),tracking=True):
                raise ValueError('카메라 시작 실패')
            self.setup_command(dict(cmd='servo_config',pan_min=-180,pan_max=180,tilt_min=-15,tilt_max=40),'limits')
            self.status.set('M1-5 자동 준비: 1296×972 / 30 FPS / quality 80 / 자동 노출·gain')
        elif self.preparing=='move' and self.setup_pending is None:
            self.setup_command(dict(cmd='move',pan=0,tilt=0,speed=100,acc=1),'arrival')
        elif self.preparing=='arrival' and self.setup_pending is None:
            self.preparing='frames';self.wait_until=now+2;self.setup_deadline=now+15
        elif self.preparing=='frames' and now>=self.wait_until:
            frame=self.app.current
            if not frame or now-frame[2]>.5 or frame[2]<self.wait_until:return
            requested=frame[1].get('requested',{})
            if self.app.size!=(1296,972) or any(requested.get(k)!=v for k,v in dict(width=1296,height=972,fps=30,quality=80).items()):return
            if frame[1].get('simulated'):raise ValueError('실제 카메라 영상이 필요합니다.')
            self.preparing=None
            auto=self.auto_measure
            DetectionPanel.start(self)
            self.auto_measure=auto
            self.measure_status.set('자동 설정 완료 · PV 검출 확인 중')

    def guard(self, fn):
        try:
            fn()
        except Exception as exc:
            import traceback
            from tkinter import messagebox
            phase=self.preparing or ('inference' if self.running else 'start/load')
            self.preparing=None;self.auto_measure=False
            self.stop()
            message=str(exc)
            self.status.set('M1-5 '+phase+' 오류: '+message)
            self.app.record(dict(event='timing_error',phase=phase,message=message,traceback=traceback.format_exc()))
            messagebox.showerror('M1-5 '+phase, message)

    def start_log(self):
        if not self.active: raise ValueError('M1-5는 자동 측정 시작 버튼으로 기록합니다.')

    def save(self):
        if not self.latest: raise ValueError('최근 검출 결과 없음')
        image,frame,row=self.latest
        if time.monotonic()-frame[2]>2:raise ValueError('오래된 영상')
        folder=self.folder if hasattr(self,'folder') else self.app.stage_dir('M1-5')
        path=folder/('frame_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        path.with_suffix('.jpg').write_bytes(frame[0]);image.save(str(path)+'_detection.jpg')
        path.with_suffix('.json').write_text(json.dumps(dict(row,frame_metadata=frame[1]),ensure_ascii=False,indent=2),encoding='utf-8')

    def begin(self):
        if self.active:raise ValueError('이미 측정 중')
        if not self.running:
            self.prepare(True);return
        servo=self.app.servo
        if not self.running or not self.latest:raise ValueError('모델 로딩 후 검출부터 시작하세요.')
        if not servo.online or servo.pending or self.app.live.pending or not servo.last or servo.simulated is not False:
            raise ValueError('실제 Pi 연결·범위 적용·초기 자세 이동을 먼저 확인하세요.')
        if self.latest[2]['count']!=1 or self.latest[2]['status']!='detected' or time.monotonic()-self.latest[1][2]>0.5:
            raise ValueError('최신 영상에 고정 PV 하나가 검출되어야 합니다.')
        cfg={k:float(self.fields[k].get()) for k in ('baseline','window','hold','floor')}
        steps=[float(s) for s in self.fields['steps'].get().split(',')]
        repeats=int(self.fields['repeats'].get())
        if not all(math.isfinite(x) and x>0 for x in cfg.values()) or cfg['hold']>=cfg['window'] or cfg['baseline']<1:
            raise ValueError('양의 유한 설정값, 기준 기록 ≥1초, 안정 유지 < 이동 관측 필요')
        if not steps or not all(math.isfinite(x) and x>=1 and x.is_integer() for x in steps) or not 1<=repeats<=100:
            raise ValueError('이동각은 1° 이상 정수, 반복은 1~100')
        origin={a:servo.last[a] for a in ('pan','tilt')}
        plan=[]
        for axis in ('pan','tilt'):
            for step in steps:
                for sign in (1,-1):
                    target=dict(origin);target[axis]+=sign*step
                    values=dict(servo.values(),**target)
                    move_from(values,servo.limits)
                    for _ in range(repeats):
                        plan.extend([(axis,sign*step,target),(axis,-sign*step,dict(origin))])
        self.cfg=cfg;self.plan=plan;self.origin=origin;self.index=0
        self.samples=[];self.results=[];self.pending=None
        self.folder=self.app.stage_dir('M1-5')/datetime.now().strftime('%Y%m%d_%H%M%S_%f');self.folder.mkdir()
        session=dict(cfg,steps=steps,repeats=repeats,origin=origin,servo=servo.context(),
                     model=self.loaded,confidence=self.settings[0],class_id=self.settings[1],
                     camera={k:v.get() for k,v in self.app.values.items()},
                     note='Laptop monotonic clock. Visual response includes command/video transport; not pure servo latency. Fixed single PV required. No actual angle feedback.')
        (self.folder/'session.json').write_text(json.dumps(session,ensure_ascii=False,indent=2),encoding='utf-8')
        self.sample_file=(self.folder/'detections.csv').open('w',newline='',encoding='utf-8')
        self.sample_writer=csv.DictWriter(self.sample_file,fieldnames=['trial','receive','processed','receive_unix_ns','seq','status','count','u','v','confidence','inference_ms','result_age_s'])
        self.sample_writer.writeheader()
        self.event_file=(self.folder/'commands.jsonl').open('w',encoding='utf-8')
        self.active=True;self.phase='baseline';self.baseline_start=time.monotonic();self.deadline=self.baseline_start+cfg['baseline']
        self.app.detection.stop()
        self.measure_status.set(f'측정 시작 · {len(plan)}회 이동 · {self.folder}')

    def event(self,event):
        if self.preparing:
            if event.get('request_id')==self.setup_pending:
                if event.get('event')=='error':
                    self.guard(lambda: (_ for _ in ()).throw(ValueError(event.get('message','자동 설정 실패'))));return
                if event.get('event')=='servo':
                    self.setup_pending=None
                    if event.get('simulated') or not event.get('available'):
                        self.guard(lambda: (_ for _ in ()).throw(ValueError('실제 서보 연결이 필요합니다.')));return
                    servo=self.app.servo
                    servo.limits=event.get('limits');servo.simulated=event.get('simulated')
                    if self.preparing=='limits':
                        for key,value in servo.limits.items():servo.vars[key].set(str(value))
                        self.preparing='move'
                    if event.get('commanded'):
                        servo.last=event['commanded']
                        for key in ('pan','tilt','speed','acc'):servo.vars[key].set(str(servo.last[key]))
            return
        if not self.active:return
        if event.get('event') in ('network','agent') and (event.get('state')=='disconnected' or event.get('agent_state')=='disconnected'):
            self.abort('연결 끊김');return
        if event.get('request_id')==self.pending:
            self.event_file.write(json.dumps(dict(event,gui_monotonic=time.monotonic()),ensure_ascii=False)+'\n');self.event_file.flush()
            if event.get('event')=='error':self.abort('명령 오류');return
            if event.get('event')=='servo':
                self.pending=None
                commanded=event.get('commanded')
                if commanded:
                    self.app.servo.last=commanded
                    for axis in ('pan','tilt'):self.app.servo.vars[axis].set(str(commanded[axis]))

    def consume(self,result,frame,now):
        super().consume(result,frame,now)
        if not self.active or frame[2]<self.baseline_start:return
        target=result['target'];status='detected' if target else 'missing'
        if now-frame[2]>0.5 or frame[1].get('simulated'):status='stale'
        row=dict(trial=self.index,receive=frame[2],processed=now,receive_unix_ns=frame[3],seq=frame[1].get('seq'),
                 status=status,count=result['count'],u=target['center'][0] if target else None,
                 v=target['center'][1] if target else None,confidence=target['confidence'] if target else None,
                 inference_ms=result['inference_ms'],result_age_s=now-frame[2])
        self.samples.append(row);self.sample_writer.writerow(row);self.sample_file.flush()

    def poll(self):
        super().poll()
        if self.preparing:
            self.prepare_poll();return
        if self.auto_measure and self.running and self.latest:
            self.auto_measure=False
            self.begin()
        if not self.active:return
        now=time.monotonic()
        if not self.app.current or now-self.app.current[2]>0.5:
            self.abort('영상 지연 / 중단');return
        if self.pending and now-self.command_time>5:self.abort('명령 응답 시간 초과');return
        if now<self.deadline:return
        if self.phase=='baseline':
            recent=[s for s in self.samples if s['receive']>=now-self.cfg['baseline']]
            if len(recent)<5 or any(s['status']!='detected' or s['count']!=1 for s in recent):
                self.abort('기준 구간 검출 부족');return
            axis,delta,target=self.plan[self.index]
            cmd=dict(cmd='move',request_id=uuid.uuid4().hex,**move_from(dict(self.app.servo.values(),**target),self.app.servo.limits))
            self.pending=cmd['request_id'];self.command_time=now
            if not self.app.send(cmd,tracking=True):self.abort('전송 실패');return
            self.event_file.write(json.dumps(dict(command=cmd,gui_monotonic=now,trial=self.index),ensure_ascii=False)+'\n');self.event_file.flush()
            self.phase='response';self.deadline=now+self.cfg['window']
            self.measure_status.set(f'{self.index+1}/{len(self.plan)} · {axis} {delta:+g}° 응답 기록')
        else:
            if self.pending:return
            axis,delta,target=self.plan[self.index]
            result=analyze(self.samples,self.command_time,self.cfg['baseline'],self.cfg['hold'],self.cfg['floor'])
            result.update(trial=self.index,axis=axis,delta_deg=delta,target=target)
            self.results.append(result)
            if result['status']!='valid':self.abort('응답 판정 실패: '+result['status']);return
            self.index+=1
            if self.index>=len(self.plan):self.abort('측정 완료');return
            self.samples=[];self.phase='baseline';self.baseline_start=now;self.deadline=now+self.cfg['baseline']

    def abort(self,reason='사용자 중단'):
        if self.preparing:
            self.preparing=None;self.auto_measure=False
            self.measure_status.set(reason+' · 자동 준비 중단 (이미 전송된 명령은 취소되지 않음)')
        if not self.active:return
        self.active=False
        self.sample_file.close();self.event_file.close()
        summary=dict(reason=reason,trials=self.results,planned=len(self.plan),completed=len(self.results),
                     interrupted_trial=self.index if self.index<len(self.plan) else None,
                     pending_request_id=self.pending,
                     note='No automatic return on abort. Check M1-2 before further motion. Timing is visual end-to-end; sample gaps bound resolution.')
        valid=[r for r in self.results if r['status']=='valid']
        summary['statistics']={}
        for axis in ('pan','tilt'):
            for key in ('onset_s','settled_s','max_sample_gap_s'):
                values=sorted(r[key] for r in valid if r['axis']==axis)
                if values:summary['statistics'][axis+'_'+key]=dict(count=len(values),median=statistics.median(values),p95=values[math.ceil(.95*len(values))-1],maximum=max(values))
        (self.folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        self.measure_status.set(reason+' · summary.json 저장 · 중단은 이미 전송된 움직임을 취소하지 않음')
        self.app.record(dict(event='timing_measurement_end',reason=reason,folder=str(self.folder)))

    def stop(self):
        self.preparing=None
        if getattr(self,'active',False):self.abort('검출 정지')
        super().stop()
