"""Episode controls layered over the existing live inference tab."""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from Tx.Controller.live_panel import LivePanel
from Tx.Controller.live_core import fresh


class LearningPanel(LivePanel):
    def __init__(self,parent,app):
        super().__init__(parent,app)
        self.learning=False;self.learner=None;self.episode=None;self.update_requested=None
        self.phase='idle';self.reset_id=None;self.episode_number=0
        self.batch=None;self.batch_settings=None
        import random
        self.seed=random.randrange(2**31)
        self.rng=None
        self.train_vars={}
        options=ttk.LabelFrame(self,text='SAC 실전 학습 · 지정 횟수 자동 반복 · 정지 버튼으로 전체 중단',padding=4)
        options.grid(row=9,column=0,columnspan=7,sticky='ew')
        fields=[('pan_min','시작 Pan min','-1'),('pan_max','max','1'),
            ('tilt_min','시작 Tilt min','-1'),('tilt_max','max','1'),
            ('seconds','최대 초','20'),('settle_seconds','초기 이동 대기 초','1'),
            ('misses','연속 미검출 횟수','3'),('loss_seconds','미검출 초','0.5'),
            ('loss_penalty','미검출 벌점 크기','10'),('updates','종료 후 업데이트','32')]
        for i,(key,label,value) in enumerate(fields):
            row=i//5;col=(i%5)*2
            ttk.Label(options,text=label).grid(row=row,column=col)
            var=tk.StringVar(value=value);self.train_vars[key]=var
            ttk.Entry(options,textvariable=var,width=7).grid(row=row,column=col+1)
        self.repeat_count=tk.StringVar(value='10')
        ttk.Label(options,text='반복 횟수').grid(row=2,column=0)
        ttk.Entry(options,textvariable=self.repeat_count,width=7).grid(row=2,column=1)
        ttk.Button(options,text='학습 시작 (지정 횟수)',command=lambda:self.guard(self.start_learning)).grid(row=2,column=2,columnspan=4,sticky='w')
        self.learning_status=tk.StringVar(value='추적 시작 = 평가만 / 학습 에피소드 시작 = SAC 추가 학습')
        ttk.Label(options,textvariable=self.learning_status,wraplength=950).grid(row=3,column=0,columnspan=10,sticky='w')
        new=ttk.LabelFrame(self,text='SAC 처음부터 학습 · YOLO만 필요 / SAC 파일 선택 불필요',padding=4)
        new.grid(row=10,column=0,columnspan=7,sticky='ew')
        self.new_vars={}
        for i,(key,label,value) in enumerate([
            ('laser_u','레이저 u (px)','696'),('laser_v','레이저 v (px)','384'),
            ('delta_limit_deg','1회 최대 Δ (°)','1'),('alignment_scale_px','보상 반감 거리 (px)','50')]):
            ttk.Label(new,text=label).grid(row=0,column=i*2)
            var=tk.StringVar(value=value);self.new_vars[key]=var
            entry=ttk.Entry(new,textvariable=var,width=7);entry.grid(row=0,column=i*2+1)
            self.widgets.append(entry)
        ttk.Button(new,text='YOLO + 새 SAC 생성',command=lambda:self.guard(self.create_new)).grid(row=1,column=0,columnspan=3,sticky='w')
        ttk.Label(new,text='기준점은 실측 보정값 입력. 새 모델: r=10/(1+d/거리), 명령 비용 0. 생성 후 미리보기 → 정지 → 학습 시작.').grid(row=2,column=0,columnspan=8,sticky='w')

    def busy(self):
        if getattr(self,'batch',None) is not None and self.batch.active:
            raise ValueError('연속 학습을 먼저 정지하세요.')
        if self.learning:
            raise ValueError('현재 학습 에피소드를 먼저 정지하세요.')
        if self.update_requested is not None or (self.future is not None and self.job[0]=='update'):
            raise ValueError('에피소드 저장/업데이트 완료를 기다리세요. 정지는 업데이트를 중단하고 저장합니다.')

    def load(self):
        self.busy()
        super().load()
        if self.learner is not None:self.learner.close()
        self.learner=None;self.episode_number=0
        self.learning_status.set('저장 모델 로딩 중 · 해당 config의 보상/행동 범위로 이어 학습')

    def create_new(self):
        self.busy()
        if self.future is not None:raise ValueError('진행 중 작업이 끝난 뒤 생성하세요.')
        from dataclasses import replace
        from simulation.config import Config
        from Tx.Controller.live_core import LiveModels, validate_config
        if not self.app.size:raise ValueError('카메라 영상을 먼저 수신하세요.')
        base=Config.load(self.paths['config'].get().strip() or None)
        values={k:float(v.get()) for k,v in self.new_vars.items()}
        cfg=replace(base,width=self.app.size[0],height=self.app.size[1],
                    scenario='stationary',action_mode='delta',command_step_deg=1.,
                    angle_limit_deg=None,slew_deg_s=None,command_delay_steps=0,actuator_tau_s=0.,
                    reward_mode='alignment',command_weight=0.,pointing_weight=10.,**values)
        validate_config(cfg)
        device=self.device.get().strip()
        if device not in ('cpu','cuda'):raise ValueError('장치는 cpu 또는 cuda')
        self.stop()
        if self.learner is not None:self.learner.close()
        self.learner=None;self.models=None;self.episode_number=0
        import random
        self.seed=random.randrange(2**31)
        self.load_paths=dict(yolo=self.paths['yolo'].get(),sac=None,
                             config=self.paths['config'].get(),mode='from_scratch',policy_seed=self.seed)
        for w in self.widgets:w.configure(state='disabled')
        self.future=self.executor.submit(LiveModels,self.load_paths['yolo'],None,None,device,cfg,self.seed)
        self.job=('load',self.generation)
        self.status.set('YOLO 로딩 / 무작위 SAC 초기화 중 · 이동 없음')
        self.learning_status.set(f'새 SAC · 레이저 ({cfg.laser_u}, {cfg.laser_v}) · Δ ±{cfg.delta_limit_deg}° · 보상 10/(1+d/{cfg.alignment_scale_px}) · 명령 비용 0')

    def preview(self):
        self.busy();super().preview()

    def start(self):
        self.busy();super().start()

    def start_learning(self):
        self.busy()
        if self.learning:raise ValueError('현재 에피소드를 먼저 정지하세요.')
        if self.future is not None:raise ValueError('검출을 정지하고 현재 추론이 끝난 뒤 시작하세요.')
        if self.models is None:raise ValueError('모델을 먼저 불러오세요.')
        from Tx.Controller.online_learning import EpisodeSettings
        ints={'pan_min','pan_max','tilt_min','tilt_max','misses','updates'}
        settings=EpisodeSettings(**{k:(int(v.get()) if k in ints else float(v.get())) for k,v in self.train_vars.items()})
        if not self.app.servo.limits:raise ValueError('Pan/Tilt 운용 범위를 먼저 적용하세요.')
        settings.validate(self.models.cfg,self.app.servo.limits)
        from Tx.Controller.online_learning import EpisodeBatch
        batch=EpisodeBatch(int(self.repeat_count.get()))
        self.batch_settings=settings
        self._start_episode(settings)
        if self.learning:
            self.batch=batch;self.batch.began()

    def _start_episode(self,settings):
        from Tx.Controller.online_learning import Episode, RealLearner
        import numpy as np
        settings.validate(self.models.cfg,self.app.servo.limits)
        # Base start calls stop; restore the batch only after successful startup.
        super().start()
        try:
            if self.learner is None:
                folder=Path('captures/M4/live_training')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                self.learner=RealLearner(self.models,folder,dict(self.load_paths,reset_seed=self.seed))
                self.rng=np.random.default_rng(self.seed)
            self.episode_number+=1
            self.episode=Episode(self.models.cfg,settings,self.episode_number)
            self.learning=True;self.phase='reset';self.models.learning=True
            self.previous_error=None
            command=settings.sample(self.rng)
            self.initial_command=command
            self.reset_id='live-reset-'+uuid.uuid4().hex
            from common.servo import move_from
            move=move_from(dict(self.app.servo.values(),pan=command[0],tilt=command[1]),self.app.servo.limits)
            if not self.app.send(dict(cmd='move',request_id=self.reset_id,**move),tracking=True):
                self.stop('초기 이동 전송 실패');return
            self.pending=(self.reset_id,time.monotonic())
            self.learning_status.set(f'에피소드 {self.episode_number}: 초기 명령 {command}° 응답 대기')
            self.app.record(dict(event='learning_episode_start',episode=self.episode_number,initial_command=command,
                                 folder=str(self.learner.folder),settings=vars(settings)))
        except Exception:
            self.stop('학습 시작 실패');raise

    def end_episode(self,reason,now,update=True):
        if not self.learning:return
        self.learning=False;self.models.learning=False;self.phase='saving'
        summary=self.episode.finish(reason,now)
        summary.update(initial_command=self.initial_command,reset_seed=self.seed)
        if getattr(self,'batch',None) is not None:
            summary.update(batch_episode=self.batch.started,batch_total=self.batch.total)
        self.learner.cancel.clear()
        if not update:self.learner.cancel.set()
        self.update_requested=(self.episode,summary,self.episode.settings.updates)
        super().stop(reason)
        self.learning_status.set(f'에피소드 {self.episode_number} 종료: {reason} · 저장 대기')

    def stop(self,reason='사용자 정지'):
        if getattr(self,'batch',None) is not None:self.batch.cancel()
        if getattr(self,'learning',False):
            self.end_episode(reason,time.monotonic(),update=False)
        else:
            if getattr(self,'learner',None) is not None and (self.update_requested is not None or (self.future is not None and self.job[0]=='update')):
                self.learner.cancel.set()
            super().stop(reason)

    def event(self,event):
        # Also cancel between episodes / while saving, when parent.running is false.
        if getattr(self,'batch',None) is not None and self.batch.active:
            foreign=(event.get('event')=='servo' and event.get('operation') in ('move','servo_config')
                     and (not self.pending or event.get('request_id')!=self.pending[0]))
            disconnected=(event.get('event') in ('hello','agent','ready','error') or
                          (event.get('event')=='network' and event.get('state')!='connected'))
            if foreign or disconnected:self.stop('외부 명령 또는 연결/장치 오류: 연속 학습 중단')
        reset_reply=self.learning and self.phase=='reset' and event.get('request_id')==self.reset_id and event.get('event')=='servo'
        super().event(event)
        if reset_reply and self.learning:
            self.phase='settle';self.not_before=time.monotonic()+self.episode.settings.settle_seconds
            self.learning_status.set('초기 이동 대기 중 · 실제 도달 피드백은 없음')

    def poll(self):
        now=time.monotonic()
        if self.learning:
            if self.phase=='settle' and now>=self.not_before:
                self.phase='acquire';self.episode.started=now
                self.learning_status.set('새 위치에서 PV 검출 대기 · 이동 없음')
            if self.episode.started is not None:
                if self.episode.loss_expired(now):self.end_episode('target_lost',now)
                elif now-self.episode.started>=self.episode.settings.seconds:
                    self.end_episode('time_limit',now)
        if self.future is not None and self.job[0]=='update':
            if not self.future.done():return
            f=self.future;self.future=None
            try:
                report=f.result();self.phase='idle'
                self.learning_status.set(f"저장 완료: {report['path']} | 경험 {report['buffer_samples']}개, 업데이트 {report['gradient_updates']}회 · 다음 시작 가능")
                self.app.record(dict(event='learning_episode_saved',**report))
                if getattr(self,'batch',None) is not None:
                    self.batch.saved(report['reason'],now)
                    if self.batch.active:
                        self.learning_status.set(f'반복 {self.batch.started}/{self.batch.total} 저장 완료 · 다음 에피소드 준비')
                    else:
                        self.learning_status.set(f"반복 종료 {self.batch.started}/{self.batch.total} · {report['reason']} · 저장: {report['path']}")
            except Exception as exc:
                self.phase='error';self.learning_status.set('저장/학습 실패: '+str(exc))
                self.app.record(dict(event='learning_error',message=str(exc)))
                # Keep staged episode and source model untouched for investigation.
                self.stop('저장/학습 오류');return
        # Parent processes even cancelled inference and pending servo responses.
        super().poll()
        if self.update_requested is not None and self.future is None:
            ep,summary,steps=self.update_requested;self.update_requested=None
            self.future=self.executor.submit(self.learner.update,ep,summary,steps)
            self.job=('update',self.generation)
            self.learning_status.set('SAC 업데이트/모델 저장 중 · 모터 명령 없음')
        self.advance_batch(now)

    def advance_batch(self,now):
        batch=getattr(self,'batch',None)
        if batch is None or not batch.ready(now):return
        if self.future is not None or self.update_requested is not None or self.pending or self.app.servo.pending:return
        # _start_episode calls parent.start -> stop, cancelling the old batch. A new
        # reference is restored only after every startup check/send has succeeded.
        remaining=(batch.total,batch.started)
        try:
            self._start_episode(self.batch_settings)
            if self.learning:
                from Tx.Controller.online_learning import EpisodeBatch
                self.batch=EpisodeBatch(remaining[0]);self.batch.started=remaining[1];self.batch.began()
        except Exception as exc:
            self.stop('다음 에피소드 시작 실패: '+str(exc))
            self.learning_status.set('연속 학습 중단: '+str(exc))
            self.app.record(dict(event='learning_error',message=str(exc)))

    def consume(self,result,frame,started,now):
        if not self.learning:
            return super().consume(result,frame,started,now)
        if not fresh(frame[2],now):self.stop('추론 결과 지연');return
        if self.phase not in ('acquire','track'):return
        if result['target'] is None:
            # Display/log miss but never execute an old action. Preserve last valid error only
            # for the next freshly detected observation, not for missing-frame decisions.
            previous=self.previous_error;self.running=False
            try:super().consume(result,frame,started,now)
            finally:self.running=True;self.previous_error=previous
            if result['count']>1:
                self.stop('다중 PV 검출: 표적 식별 불명');return
            end=self.episode.missed(now)
            self.learning_status.set(f'미검출 {self.episode.missing_count}회 · 이동 보류')
            if end:self.end_episode('target_lost',now)
            return
        # A fresh post-command observation closes the PREVIOUS action's transition.
        self.episode.observe(result,now)
        self.phase='track'
        super().consume(result,frame,started,now)
        if self.learning and self.running:
            self.episode.remember(result,now)
            left=max(0,self.episode.settings.seconds-(now-self.episode.started))
            self.learning_status.set(f'에피소드 {self.episode_number} · 남은 {left:.1f}초 · 경험 {len(self.episode.rows)}개 · SAC 탐색 중')

    def close(self):
        self.stop('창 닫힘')
        # Preserve the final episode even if the Tk poll loop is about to disappear.
        if self.update_requested is not None:
            ep,summary,steps=self.update_requested;self.update_requested=None
            self.learner.cancel.set()
            self.executor.submit(self.learner.update,ep,summary,steps)
        if self.learner is not None:self.executor.submit(self.learner.close)
        self.executor.shutdown(wait=False,cancel_futures=False)
