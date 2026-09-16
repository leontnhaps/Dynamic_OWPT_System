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
        import random
        self.seed=random.randrange(2**31)
        self.rng=None
        self.train_vars={}
        options=ttk.LabelFrame(self,text='SAC 추가 학습 · 종료 후 저장/업데이트 · 다음 에피소드는 버튼으로 시작',padding=4)
        options.grid(row=9,column=0,columnspan=7,sticky='ew')
        fields=[('pan_min','시작 Pan min','-3'),('pan_max','max','3'),
            ('tilt_min','시작 Tilt min','-3'),('tilt_max','max','3'),
            ('seconds','최대 초','20'),('settle_seconds','초기 이동 대기 초','1'),
            ('misses','연속 미검출 횟수','3'),('loss_seconds','미검출 초','0.5'),
            ('loss_penalty','미검출 벌점 크기','10'),('updates','종료 후 업데이트','32')]
        for i,(key,label,value) in enumerate(fields):
            row=i//5;col=(i%5)*2
            ttk.Label(options,text=label).grid(row=row,column=col)
            var=tk.StringVar(value=value);self.train_vars[key]=var
            ttk.Entry(options,textvariable=var,width=7).grid(row=row,column=col+1)
        ttk.Button(options,text='학습 에피소드 시작 / 다음',command=lambda:self.guard(self.start_learning)).grid(row=2,column=0,columnspan=4,sticky='w')
        self.learning_status=tk.StringVar(value='추적 시작 = 평가만 / 학습 에피소드 시작 = SAC 추가 학습')
        ttk.Label(options,textvariable=self.learning_status,wraplength=950).grid(row=3,column=0,columnspan=10,sticky='w')

    def busy(self):
        if self.learning:
            raise ValueError('현재 학습 에피소드를 먼저 정지하세요.')
        if self.update_requested is not None or (self.future is not None and self.job[0]=='update'):
            raise ValueError('에피소드 저장/업데이트 완료를 기다리세요. 정지는 업데이트를 중단하고 저장합니다.')

    def load(self):
        self.busy()
        super().load()
        if self.learner is not None:self.learner.close()
        self.learner=None;self.episode_number=0

    def preview(self):
        self.busy();super().preview()

    def start(self):
        self.busy();super().start()

    def start_learning(self):
        self.busy()
        if self.learning:raise ValueError('현재 에피소드를 먼저 정지하세요.')
        if self.future is not None:raise ValueError('검출을 정지하고 현재 추론이 끝난 뒤 시작하세요.')
        if self.models is None:raise ValueError('모델을 먼저 불러오세요.')
        from Tx.Controller.online_learning import EpisodeSettings, Episode, RealLearner
        import numpy as np
        ints={'pan_min','pan_max','tilt_min','tilt_max','misses','updates'}
        settings=EpisodeSettings(**{k:(int(v.get()) if k in ints else float(v.get())) for k,v in self.train_vars.items()})
        if not self.app.servo.limits:raise ValueError('Pan/Tilt 운용 범위를 먼저 적용하세요.')
        settings.validate(self.models.cfg,self.app.servo.limits)
        # Existing start validates known commanded pose, connection, pending requests and resolution.
        super().start()
        try:
            if self.learner is None:
                folder=Path('captures/M3/live_training')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
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
        self.learner.cancel.clear()
        if not update:self.learner.cancel.set()
        self.update_requested=(self.episode,summary,self.episode.settings.updates)
        super().stop(reason)
        self.learning_status.set(f'에피소드 {self.episode_number} 종료: {reason} · 저장 대기')

    def stop(self,reason='사용자 정지'):
        if getattr(self,'learning',False):
            self.end_episode(reason,time.monotonic(),update=False)
        else:
            if getattr(self,'learner',None) is not None and (self.update_requested is not None or (self.future is not None and self.job[0]=='update')):
                self.learner.cancel.set()
            super().stop(reason)

    def event(self,event):
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
