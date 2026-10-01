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
        self.measure_status=tk.StringVar(value='M1-2 범위 적용·초기 자세 이동 → 모델 로딩·검출 시작 → 자동 측정')
        ttk.Label(row,textvariable=self.measure_status,wraplength=1000).pack(side='left')

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
        if getattr(self,'active',False):self.abort('검출 정지')
        super().stop()
