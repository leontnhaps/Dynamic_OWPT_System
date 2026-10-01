"""Stationary A/B processing profile; all laptop durations use monotonic time."""
import csv
import io
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from PIL import Image, ImageTk
from Tx.Controller.detection_panel import PVDetector


def describe(values):
    values=sorted(values)
    if not values:return dict(count=0)
    return dict(count=len(values),median=statistics.median(values),
                p95=values[math.ceil(.95*len(values))-1],maximum=max(values))


def timed_infer(model, frame, settings, submitted):
    started=time.monotonic()
    result=model.infer(frame[0],*settings)
    return result,dict(receive=frame[2],submitted=submitted,worker_started=started,
                       worker_finished=time.monotonic())


class ProcessingProfile:
    def __init__(self,panel):
        self.panel=panel;self.app=panel.app;self.running=False
        self.pool=ThreadPoolExecutor(max_workers=1);self.future=None
        self.rows=[];self.frames=[];self.events=[]
        self.display=None;self.rendered_frame=None;self.receipts=[]

    def start(self):
        p=self.panel
        if self.running:raise ValueError('처리 시간 측정이 진행 중입니다.')
        if self.future is not None and not self.future.done():raise ValueError('이전 처리 작업 완료 후 다시 시작하세요.')
        if p.active or p.preparing or p.future is not None:raise ValueError('기존 측정·모델 작업을 종료한 뒤 시작하세요.')
        self.app.live.stop('처리 시간 측정');self.app.detection.stop();p.stop()
        if self.app.live.pending or self.app.servo.pending:raise ValueError('이전 서보 응답을 기다리세요.')
        cfg=dict(width=1296,height=972,fps=30,quality=80,shutter_speed=None,analogue_gain=None)
        for key,value in cfg.items():self.app.values[key].set('' if value is None else str(value))
        if not self.app.send(dict(cmd='preview',enable=True,**cfg)):raise ValueError('카메라 시작 실패')
        confidence=float(p.conf.get());class_id=int(p.class_id.get());device=p.device.get().strip()
        if not 0<confidence<=1 or device not in ('cpu','cuda'):raise ValueError('모델 설정을 확인하세요.')
        self.settings=(confidence,class_id);self.model=None
        self.folder=self.app.stage_dir('M1-5')/('processing_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));self.folder.mkdir()
        self.config=dict(camera=cfg,model_path=p.path.get(),device=device,confidence=confidence,class_id=class_id,
            phase_seconds=30,warmup_seconds=2,repetitions=3,schema_version=2,
            pipeline='inference_first_visible_render',poll_interval_ms=5,render_interval_ms=33,
            clock_resolution_s=time.get_clock_info('monotonic').resolution,
            note='Stationary test. No servo commands. Laptop monotonic clock only. Pi clocks remain separate. Frames CSV contains actual visible renders (YOLO mode reuses decoded inference images). Processing CSV includes all in-window results, with blank render fields for undisplayed results. Render timestamps mark widget update return, not monitor scanout.')
        (self.folder/'session.json').write_text(json.dumps(self.config,ensure_ascii=False,indent=2),encoding='utf-8')
        self.rows=[];self.frames=[];self.events=[];self.receipts=[];self.phases=[];self.index=-1;self.last=None
        self.display=None;self.rendered_frame=None
        self.running=True;self.state='loading';self.deadline=time.monotonic()+120
        self.future=self.pool.submit(PVDetector,p.path.get(),device)
        p.measure_status.set('처리 시간 측정: 모델 준비 중 · 서보 이동 없음')

    def next_phase(self,now):
        self.index+=1
        if self.index==6:self.finish('측정 완료');return
        self.mode='video' if self.index%2==0 else 'yolo'
        self.state='warmup';self.phase_start=now;self.measure_start=now+2;self.deadline=self.measure_start+30
        self.last=None
        self.display=None;self.rendered_frame=None
        self.phases.append(dict(index=self.index,mode=self.mode,repetition=self.index//2+1,start=now,measure_start=self.measure_start,end=self.deadline))

    def tick(self):
        if not self.running:return
        try:self._tick()
        except Exception as exc:
            self.finish('오류: '+str(exc))
            self.app.record(dict(event='processing_profile_error',message=str(exc)))

    def _tick(self):
        now=time.monotonic()
        if self.app.live.detecting or self.app.detection.running:
            raise ValueError('다른 검출 작업 시작으로 측정 중단')
        if self.state=='loading':
            if now>self.deadline:raise ValueError('모델 로딩 시간 초과')
            if not self.future.done():return
            self.model=self.future.result();self.future=None
            if self.settings[1] not in self.model.names:raise ValueError('PV class ID 확인 필요')
            self.state='camera';self.deadline=now+15
        if self.state=='camera':
            frame=self.app.current
            if now>self.deadline:raise ValueError('새 카메라 설정 영상 대기 시간 초과')
            if not frame or now-frame[2]>.5:return
            if frame[1].get('simulated'):raise ValueError('실제 카메라 영상이 필요합니다.')
            req=frame[1].get('requested',{})
            if any(req.get(k)!=v for k,v in self.config['camera'].items()):return
            if self.app.live.future is not None or self.app.detection.future is not None:return
            self.next_phase(now)
        frame=self.app.current
        if not frame or now-frame[2]>5:raise ValueError('영상 수신 5초 이상 중단')
        if self.future is not None and now-self.submitted>10:
            raise ValueError('YOLO 처리 10초 이상 지연')
        if self.future is not None and self.future.done():
            result,marks=self.future.result();self.future=None
            marks.update(phase=self.index,mode=self.mode,seq=self.job[1].get('seq'),
                         result_ready=time.monotonic(),inference_ms=result['inference_ms'],count=result['count'])
            marks.update(widget_updated=None,render_ms=None,receive_to_widget_ms=None,
                         result_to_render_ms=None,render_work_ms=None)
            if self.measure_start<=marks['receive'] and marks['result_ready']<=self.deadline:
                marks.update(receive_to_submit_ms=1000*(marks['submitted']-marks['receive']),
                    queue_ms=1000*(marks['worker_started']-marks['submitted']),
                    worker_ms=1000*(marks['worker_finished']-marks['worker_started']),
                    result_wait_ms=1000*(marks['result_ready']-marks['worker_finished']),
                    receive_to_result_ms=1000*(marks['result_ready']-marks['receive']))
                self.rows.append(marks)
            self.display=(result['image'],self.job,self.job_total,marks)
        if now>=self.deadline:
            # Drain the old phase's inference before advancing, excluding boundary samples.
            if self.future is not None:return
            self.next_phase(now)
            if not self.running:return
        self.state='measure' if now>=self.measure_start else 'warmup'
        status=f'처리 시간 {self.index//2+1}/3 · {"영상만" if self.mode=="video" else "YOLO 포함"} · {max(0,self.deadline-now):.0f}초 남음'
        if status!=getattr(self,'last_status',None):
            self.panel.measure_status.set(status);self.last_status=status
        if self.mode=='yolo' and self.future is None and now-frame[2]<=.5 and frame[2]!=self.last:
            self.last=frame[2];self.job=frame;self.job_total=self.app.received_count
            submitted=time.monotonic();self.submitted=submitted;self.future=self.pool.submit(timed_infer,self.model,frame,self.settings,submitted)

    def received(self,frame,total):
        if not self.running or self.state not in ('warmup','measure'):return
        if self.measure_start<=frame[2]<=self.deadline:
            self.receipts.append(dict(phase=self.index,receive=frame[2],total=total))

    def render(self):
        if not self.running or self.state not in ('warmup','measure'):return
        started=time.monotonic()
        if started>=self.deadline:return
        if self.mode=='video':
            frame=self.app.current;total=self.app.received_count;marks=None
            if not frame or frame[2]==self.rendered_frame:return
            image=Image.open(io.BytesIO(frame[0]))
        else:
            if self.display is None:return
            image,frame,total,marks=self.display
            if frame[2]==self.rendered_frame:return
            image=image.copy()
        image.thumbnail((1000,420));photo=ImageTk.PhotoImage(image)
        self.panel.canvas.configure(image=photo);self.panel.canvas.image=photo
        ended=time.monotonic();self.rendered_frame=frame[2]
        if marks is not None and ended<=self.deadline:
            marks.update(widget_updated=ended,render_ms=1000*(ended-marks['result_ready']),
                         receive_to_widget_ms=1000*(ended-frame[2]),
                         result_to_render_ms=1000*(started-marks['result_ready']),
                         render_work_ms=1000*(ended-started))
        self.frame(frame,started,ended,total)

    def frame(self,frame,display_start,display_end,total):
        if not self.running or self.state not in ('warmup','measure'):return
        if self.measure_start<=frame[2] and display_end<=self.deadline:
            self.frames.append(dict(phase=self.index,mode=self.mode,seq=frame[1].get('seq'),receive=frame[2],
                gui_start=display_start,widget_updated=display_end,total_received=total,
                receive_to_gui_ms=1000*(display_start-frame[2]),gui_work_ms=1000*(display_end-display_start),
                receive_to_widget_ms=1000*(display_end-frame[2])))

    def event(self,event):
        if self.running and event.get('event') in ('preview_timing','network','error','preview'):
            self.events.append(dict(event,profile_phase=self.index,laptop_event_time=time.monotonic()))

    def finish(self,reason='사용자 중단'):
        if not self.running:return
        self.running=False
        self.display=None
        for name,rows in [('processing.csv',self.rows),('frames.csv',self.frames)]:
            with (self.folder/name).open('w',newline='',encoding='utf-8') as f:
                if rows:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with (self.folder/'pi_events.jsonl').open('w',encoding='utf-8') as f:
            for event in self.events:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        summary=dict(reason=reason,phases=[],note=self.config['note'],schema_version=2,
                     pipeline=self.config.get('pipeline','inference_first_visible_render'))
        for phase in self.phases:
            entry=dict(phase)
            received=[r for r in self.receipts if r['phase']==phase['index']]
            if len(received)>1:
                elapsed=received[-1]['receive']-received[0]['receive']
                entry['receive_fps']=(received[-1]['total']-received[0]['total'])/elapsed if elapsed>0 else None
            for source,rows in [('video',self.frames),('inference',self.rows)]:
                group=[r for r in rows if r['phase']==phase['index']]
                entry[source]=dict(samples=len(group),metrics={})
                duration=phase.get('end',0)-phase.get('measure_start',0)
                if duration>0:entry[source]['sample_rate_hz']=len(group)/duration
                if source=='inference':entry[source]['rendered_samples']=sum(r.get('widget_updated') is not None for r in group)
                if group:
                    for key in group[0]:
                        if key.endswith('_ms'):entry[source]['metrics'][key]=describe([r[key] for r in group if r[key] is not None])
            summary['phases'].append(entry)
        (self.folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        self.panel.measure_status.set(reason+' · '+str(self.folder))
        self.app.record(dict(event='processing_profile_end',reason=reason,folder=str(self.folder)))

    def close(self):
        self.finish('GUI 종료');self.pool.shutdown(wait=False,cancel_futures=True)
