"""Manual M1-2 workflow using the existing GUI event loop and evidence log."""
import json
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from common.servo import number, limits_from, move_from


class ServoPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=8)
        self.app = app
        self.pending = None
        self.last = None
        self.limits = None
        self.online = False
        self.simulated = None
        self.reference = None
        self.vars = {}
        self.buttons = []
        self.state = tk.StringVar(value='Pi 상태 조회 필요 · 실제각/도달 여부 미측정')
        ttk.Label(self, textvariable=self.state).grid(row=0, column=0, columnspan=8, sticky='w')
        fields = [('pan_min','Pan min',''), ('pan_max','Pan max',''),
                  ('tilt_min','Tilt min',''), ('tilt_max','Tilt max',''),
                  ('pan','Pan °',''), ('tilt','Tilt °',''),
                  ('speed','SPD','100'), ('acc','ACC','1')]
        for col, (key, label, value) in enumerate(fields):
            ttk.Label(self, text=label).grid(row=1, column=col, sticky='w')
            var = tk.StringVar(value=value)
            self.vars[key] = var
            ttk.Entry(self, textvariable=var, width=10).grid(row=2, column=col, padx=2)
        actions = ttk.Frame(self)
        actions.grid(row=3, column=0, columnspan=8, sticky='w', pady=5)
        for label, fn in [('상태 조회', self.refresh), ('운용 범위 적용', self.configure_limits),
                          ('입력 각도로 이동', self.absolute), ('직전 명령을 기준 자세로 저장', self.set_reference),
                          ('기준 자세 복귀', self.go_reference)]:
            button = ttk.Button(actions, text=label, command=lambda f=fn:self.guard(f))
            button.pack(side='left', padx=2)
            self.buttons.append(button)
        jog = ttk.Frame(self)
        jog.grid(row=4, column=0, columnspan=8, sticky='w')
        ttk.Label(jog, text='Step °').pack(side='left')
        self.step = tk.StringVar(value='1')
        ttk.Entry(jog, textvariable=self.step, width=6).pack(side='left')
        for axis in ('pan', 'tilt'):
            for sign in (-1, 1):
                b = ttk.Button(jog, text=f'{axis.title()} {"+" if sign>0 else "−"}',
                               command=lambda a=axis,s=sign:self.guard(lambda:self.jog(a,s)))
                b.pack(side='left', padx=2)
                self.buttons.append(b)
        self.ref_text = tk.StringVar(value='기준 자세: 미등록 (저장은 이동/영점 변경을 하지 않음)')
        ttk.Label(jog, textvariable=self.ref_text).pack(side='left', padx=8)
        notes = ttk.Frame(self)
        notes.grid(row=5, column=0, columnspan=8, sticky='ew', pady=5)
        self.test = tk.StringVar(value='영점/기준 자세')
        ttk.Combobox(notes,textvariable=self.test,state='readonly',width=17,
                     values=['영점/기준 자세','Pan 부호','Tilt 부호','운용 범위','반복 복귀']).pack(side='left')
        self.note = tk.StringVar()
        ttk.Entry(notes,textvariable=self.note,width=42).pack(side='left',padx=3)
        ttk.Button(notes,text='관찰 + JPEG/JSON 저장',command=lambda:self.guard(self.observe)).pack(side='left')
        profiles = ttk.Frame(self)
        profiles.grid(row=6,column=0,columnspan=8,sticky='w')
        ttk.Button(profiles,text='설정 저장',command=lambda:self.guard(self.save_profile)).pack(side='left')
        ttk.Button(profiles,text='설정 불러오기',command=lambda:self.guard(self.load_profile)).pack(side='left')
        ttk.Label(profiles,text='불러온 범위는 별도 적용 필요 · ±는 명령 증가/감소 (좌우·상하 미확정)').pack(side='left',padx=8)
        ttk.Label(self,text='범위는 실물 간섭을 확인하며 입력. SPD/ACC는 기존 펌웨어 값이며 실측 각속도가 아님.\n'
                  '명령 전송 완료 ≠ 실제 도달. 레이저 OFF 상태에서 영상으로 관찰 후 다음 이동.').grid(
                      row=7,column=0,columnspan=8,sticky='w')

    def guard(self, fn):
        try:
            fn()
        except (ValueError, TypeError, OSError, KeyError) as exc:
            messagebox.showerror('M1-2',str(exc))

    def values(self):
        return {key:var.get().strip() for key,var in self.vars.items()}

    def request(self, cmd):
        if self.pending:
            raise ValueError('이전 요청의 응답을 기다리고 있습니다.')
        if not self.online:
            raise ValueError('Pi 제어 연결을 확인하세요.')
        token = uuid.uuid4().hex
        cmd = dict(cmd,request_id=token)
        if self.app.send(cmd):
            self.pending = (token,time.monotonic(),cmd['cmd'])
            self.state.set(f"{cmd['cmd']} 응답 대기 · 실제각 미측정")
            for b in self.buttons:
                b.state(['disabled'])

    def refresh(self):
        self.request(dict(cmd='servo_status'))

    def configure_limits(self):
        self.request(dict(cmd='servo_config',**limits_from(self.values())))

    def move(self, pan, tilt):
        values = dict(self.values(),pan=pan,tilt=tilt)
        # Edited limits must be applied explicitly before further movements.
        if limits_from(values) != self.limits:
            raise ValueError('편집한 운용 범위를 먼저 적용하세요.')
        move = move_from(values,self.limits)
        self.request(dict(cmd='move',**move))

    def absolute(self):
        self.move(self.vars['pan'].get(),self.vars['tilt'].get())

    def jog(self, axis, sign):
        if self.last is None:
            raise ValueError('먼저 명시적인 Pan/Tilt 각도 명령을 전송하세요. 현재 실제각은 알 수 없습니다.')
        step = number(self.step.get(),'step')
        if step <= 0:
            raise ValueError('Step은 양수여야 합니다.')
        target = {a:self.last[a] for a in ('pan','tilt')}
        target[axis] += sign * step
        self.move(**target)

    def set_reference(self):
        if self.last is None:
            raise ValueError('전송 완료된 명령이 없습니다.')
        self.reference = {a:self.last[a] for a in ('pan','tilt')}
        self.show_reference()
        self.app.record(dict(event='m1_2_reference',reference=self.reference,actual_angle=None))

    def show_reference(self):
        self.ref_text.set(f'기준 명령: {self.reference}' if self.reference else '기준 자세: 미등록')

    def go_reference(self):
        if self.reference is None:
            raise ValueError('기준 자세를 먼저 저장하세요.')
        self.move(**self.reference)

    def invalidate(self, text):
        self.pending = None
        self.last = None
        self.limits = None
        self.state.set(text + ' · 상태 조회/범위 재적용 필요 · 실제각 미측정')
        for b in self.buttons:
            b.state(['!disabled'])

    def event(self, event):
        kind = event.get('event')
        if kind=='network' and event.get('port')==7600 and event.get('state')!='connected':
            self.online=False
            self.invalidate('제어 연결 끊김')
        if kind in ('hello','agent','ready'):
            self.online = kind=='ready' or event.get('agent_state',event.get('state'))=='connected'
            self.invalidate('Pi 연결됨' if self.online else 'Pi 연결 끊김')
            if self.online:
                self.guard(self.refresh)
        if kind not in ('servo','error'):
            return
        if not self.pending or event.get('request_id')!=self.pending[0]:
            return
        self.pending=None
        for b in self.buttons:
            b.state(['!disabled'])
        if kind=='error':
            self.invalidate(event.get('message','요청 실패'))
            return
        self.limits=event.get('limits')
        self.last=event.get('commanded')
        self.simulated=event.get('simulated')
        prefix='SIMULATION | ' if self.simulated else ''
        self.state.set(prefix + f"직전 전송 명령: {self.last or '없음'} · 실제각/도달 미측정")
        if event.get('operation')=='move' and self.last:
            for a in ('pan','tilt'):
                self.vars[a].set(str(self.last[a]))

    def tick(self):
        if self.pending and time.monotonic()-self.pending[1]>5:
            self.invalidate('응답 시간 초과: 실행 여부 불명 (자동 재전송 없음)')

    def context(self):
        return dict(test=self.test.get(),note=self.note.get(),commanded=self.last,
                    applied_limits=self.limits,reference=self.reference,simulated=self.simulated,
                    pending=bool(self.pending),actual_angle=None,arrival_verified=False)

    def observe(self):
        if self.pending:
            raise ValueError('명령 응답 후 영상을 확인하고 기록하세요.')
        if not self.note.get().strip():
            raise ValueError('관찰 내용을 입력하세요. 예: Pan + → 장치 우회전, 표식은 영상 왼쪽으로 이동')
        self.app.save()

    def save_profile(self):
        data=dict(version=1,settings=self.values(),step=self.step.get(),reference=self.reference)
        limits_from(data['settings'])
        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='m1_2_settings.json')
        if path:
            Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            self.app.record(dict(event='m1_2_profile_saved',file=path,profile=data))

    def load_profile(self):
        if self.pending:
            raise ValueError('요청 완료 후 설정을 불러오세요.')
        path=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not path:
            return
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version')!=1:
            raise ValueError('지원하지 않는 설정 버전')
        values=data['settings']
        limits_from(values)
        for key in self.vars:
            if key not in values:
                raise ValueError(f'설정 누락: {key}')
        reference=data.get('reference')
        if reference is not None:
            reference={a:number(reference[a],a) for a in ('pan','tilt')}
        step=number(data.get('step'),'step')
        if step<=0:
            raise ValueError('Step은 양수여야 합니다.')
        for key,var in self.vars.items():
            var.set(str(values[key]))
        self.reference=reference
        self.show_reference()
        self.step.set(str(step))
        self.limits=None
        self.state.set('설정 불러옴 · 운용 범위 적용 필요 · 자동 이동 없음')
        self.app.record(dict(event='m1_2_profile_loaded',file=path,profile=data))
