from concurrent.futures import Future
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common.tx_moving import MovingConfig
from common.tx_tracking import Sample
from common.tx_setup import TX_CAMERA_SETTINGS
from common.moving_paths import MOVING_CHECKPOINT_PRESETS, DEFAULT_KALMAN_PATH
from Tx.Controller.moving_panel import MovingPanel
from Tx.Controller.moving_core import MovingRunSettings


class Variable:
    def __init__(self, value=''): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class MovingPanelTests(unittest.TestCase):
    def panel(self):
        p = MovingPanel.__new__(MovingPanel)
        p.tk = None
        p.cfg = MovingConfig()
        p.learner = SimpleNamespace(cancel=threading.Event(), mode='resume', episodes=13, finish_episode=Mock())
        p.run = p.log = p.detector = None
        p.preparing = p.setup_pending = p.work = p.inference = None
        p.previewing = p.hardware_ready = p.closed = False
        p._generation = 0; p._start_requested = False; p._save_submit_failed = False
        p.servo_available = True
        p.latest = p.display = p.rendered = p.last_frame = None
        p.path = Variable('captures/M5-3/yolo.pt')
        p.kalman_path = Variable('captures/M5-3/kf.json')
        p.checkpoint = Variable('captures/M5-3/checkpoint')
        p.model_preset = Variable(next(iter(MOVING_CHECKPOINT_PRESETS)))
        p.reward_text = Variable()
        p.transmission_fps = Variable('30')
        p.camera_config = dict(TX_CAMERA_SETTINGS)
        p.processing_text = Variable()
        p.mode = Variable('이어서 학습'); p.device = Variable('cpu'); p.sac_device = Variable('cpu')
        p.start_pan = Variable('0'); p.start_tilt = Variable('0'); p.start_note = Variable(); p.note = Variable()
        p.status = Variable(); p.progress = Variable(); p.result_text = Variable(); p.period_text = Variable()
        p.lock = Mock(); p.infer_pool = Mock(); p.learn_pool = Mock()
        p.learn_pool.submit.return_value = Future()
        p.app = SimpleNamespace(
            record=Mock(), send=Mock(return_value=True), current=None, size=(1296,972), received_count=1,
            servo=SimpleNamespace(online=True, simulated=False, limits=p.cfg.limits, pending=None,
                                  vars={k:Variable() for k in ('pan','tilt','pan_min','pan_max','tilt_min','tilt_max','speed','acc')}),
            values={k:Variable() for k in TX_CAMERA_SETTINGS},
            ook=SimpleNamespace(busy=False), sweep=SimpleNamespace(busy=False),
            stationary=SimpleNamespace(busy=False, run=None, inference=None, stop=Mock()),
            timing=SimpleNamespace(active=False, preparing=False, profile=SimpleNamespace(running=False),
                                   future=None, stop=Mock()),
            detection=SimpleNamespace(future=None, stop=Mock()))
        p.reset_processing_stats()
        p.identity = p.selection()
        return p

    def fake_run(self, phase='ready', **kwargs):
        r = SimpleNamespace(phase=phase, active=True, pending=None, save_request=None, episode=14,
            steps=2, new_transitions=1, inference_allowed=False, command=Mock(), tick=Mock(),
            set_start=Mock(), manual_end=Mock(), abort=Mock(), saved=Mock(), supply=Mock(),
            stop_requested=False, settings=MovingRunSettings(), **kwargs)
        r.command.tolist.return_value = [0,0]
        return r

    def test_explicit_preset_selects_paired_model_and_requires_preparation(self):
        for label, checkpoint in MOVING_CHECKPOINT_PRESETS.items():
            with self.subTest(model=label):
                p = self.panel(); p.hardware_ready = p.previewing = True
                p.model_preset.set(label); p.apply_model_preset()
                self.assertEqual(p.checkpoint.get(), str(checkpoint))
                self.assertEqual(p.kalman_path.get(), str(DEFAULT_KALMAN_PATH))
                self.assertFalse(p.hardware_ready); self.assertFalse(p.previewing)
                with self.assertRaisesRegex(ValueError, '자동 준비'): p.check_prepared()
                p.app.send.assert_not_called()

    def test_model_preset_cannot_replace_an_active_episode(self):
        p = self.panel(); p.run = self.fake_run()
        old = p.checkpoint.get()
        with self.assertRaisesRegex(ValueError, '먼저 종료'): p.apply_model_preset()
        self.assertEqual(p.checkpoint.get(), old)
        p.run.abort.assert_not_called(); p.app.send.assert_not_called()

    def test_transmission_fps_selection_requires_new_preparation_without_changing_model(self):
        p = self.panel(); p.hardware_ready = True
        previous_cfg = p.cfg
        for value in ('10', '5'):
            p.transmission_fps.set(value)
            self.assertEqual(p.selection()['camera_fps'], int(value))
            with self.assertRaisesRegex(ValueError, '자동 준비'): p.check_prepared()
        self.assertIs(p.cfg, previous_cfg)
        self.assertEqual(p.cfg.dt, .72)
        p.app.send.assert_not_called()
        for value in ('0', '30.0', '60', ''):
            p.transmission_fps.set(value)
            with self.assertRaisesRegex(ValueError, '전송 목표 FPS'): p.selection()

    def test_preparation_uses_selected_fps_snapshot_and_checks_matching_frames(self):
        for fps in (10, 5):
            with self.subTest(fps=fps):
                p = self.panel(); p.transmission_fps.set(str(fps)); p.identity = p.selection()
                # Loading uses its captured selection, never later widget values.
                p.transmission_fps.set('30')
                p.send = Mock(return_value=True)
                p.prepare_hardware()
                camera = p.send.call_args_list[0].args[0]
                expected = dict(TX_CAMERA_SETTINGS, fps=fps)
                for key, value in expected.items(): self.assertEqual(camera[key], value)
                self.assertEqual(p.app.values['fps'].get(), str(fps))
                frame = (b'jpeg', {'requested': expected}, 0., 0)
                self.assertTrue(p.camera_matches(frame))
                self.assertFalse(p.camera_matches((b'', {'requested': dict(TX_CAMERA_SETTINGS)}, 0., 0)))
                wrong = dict(expected, quality=60)
                self.assertFalse(p.camera_matches((b'', {'requested': wrong}, 0., 0)))
                self.assertEqual(TX_CAMERA_SETTINGS['fps'], 30)
                self.assertEqual(p.cfg.dt, .72)

    def test_run_records_the_prepared_transmission_profile(self):
        for fps in (10, 5):
            with self.subTest(fps=fps):
                p = self.panel(); p.hardware_ready = True; p.kalman = {'fixture': True}
                p.transmission_fps.set(str(fps)); p.identity = p.selection()
                p.camera_config = dict(TX_CAMERA_SETTINGS, fps=fps)
                frame = (b'', {'simulated': False, 'requested': dict(p.camera_config)}, 10., 0)
                p.check_prepared = Mock(); p.stop_other_panels = Mock(); p.check_camera = Mock(return_value=frame)
                p.app.stage_dir = Mock(return_value=Path('captures/M5-3'))
                with patch('Tx.Controller.moving_panel.MovingRunLog') as log, \
                        patch('Tx.Controller.moving_panel.MovingRun') as run:
                    p._start_episode()
                config = log.call_args.args[1]
                self.assertEqual(config['camera'], dict(TX_CAMERA_SETTINGS, fps=fps))
                self.assertEqual(config['model']['camera_fps'], fps)
                self.assertEqual(config['tracking']['dt'], .72)
                run.return_value.start.assert_called_once()

    def test_processing_diagnostics_separate_received_and_completed_fps_and_throttle_display(self):
        p = self.panel()
        with patch('Tx.Controller.moving_panel.time.monotonic', return_value=0.): p.reset_processing_stats()
        p.app.received_count += 10
        p._processing_completed = 2
        p._processing_last = dict(yolo_ms=25., receive_to_processed_ms=60., storage_ms=110.)
        p._processing_policy_ms = 5.
        with patch('Tx.Controller.moving_panel.time.monotonic', return_value=1.): p.update_processing_stats()
        text = p.processing_text.get()
        for part in ('수신 10.0 FPS', '처리 완료 2.0 FPS', 'YOLO 25 ms', '추론 완료 60 ms', '원본 저장 110 ms', 'SAC 행동 5 ms'):
            self.assertIn(part, text)
        p.app.received_count += 3
        with patch('Tx.Controller.moving_panel.time.monotonic', return_value=1.1): p.update_processing_stats()
        self.assertEqual(p.processing_text.get(), text)
        p.app.record.assert_not_called(); p.app.send.assert_not_called()
        p.run = self.fake_run('action_reply')
        with patch('Tx.Controller.moving_panel.time.monotonic', return_value=1.5): p.update_processing_stats()
        self.assertIn('추론 일시 정지', p.processing_text.get())

    def test_worker_keeps_raw_storage_before_return_and_reports_its_time(self):
        p = self.panel(); p.previewing = True; p.log = Mock()
        p.run = self.fake_run('result'); p.run.inference_allowed = True
        p.detector = SimpleNamespace(infer=Mock(return_value=dict(
            image=SimpleNamespace(width=1296, height=972), target=None, count=0, inference_ms=25.)))
        p.app.current = (b'original-jpeg', {'simulated': False, 'seq': 7}, 10., 123)
        with patch('Tx.Controller.moving_panel.time.monotonic', return_value=10.01): p._poll()
        worker = p.infer_pool.submit.call_args.args[0]
        with patch('Tx.Controller.moving_panel.time.monotonic', side_effect=[10.02, 10.03, 10.05]):
            output = worker()
        self.assertEqual(len(output), 4)
        result, sample, frame, generation = output
        self.assertEqual(frame, p.app.current); self.assertEqual(generation, p._generation)
        self.assertEqual(p.log.record_frame.call_args.args[4], b'original-jpeg')
        self.assertEqual(sample.processed, 10.02)
        self.assertAlmostEqual(result['_moving_timing']['storage_ms'], 20.)
        self.assertAlmostEqual(result['_moving_timing']['receive_to_processed_ms'], 20.)
        self.assertEqual(result['_moving_timing']['yolo_ms'], 25.)

    def test_startpoint_change_during_tracking_is_applied_without_sending_move(self):
        p = self.panel(); p.run = self.fake_run()
        p.start_pan.set('-17'); p.start_tilt.set('3'); p.apply_start()
        p.run.set_start.assert_called_once_with(-17,3)
        p.app.send.assert_not_called()
        self.assertIn('다음 복귀', p.start_note.get())

    def test_bad_startpoint_does_not_abort_running_episode(self):
        p = self.panel(); p.run = self.fake_run(); p.start_pan.set('100')
        with patch('Tx.Controller.moving_panel.messagebox.showerror') as error:
            p.guard(p.apply_start)
        error.assert_called_once(); p.run.abort.assert_not_called(); p.app.send.assert_not_called()

    def test_nonfinite_and_fractional_startpoint_rejected(self):
        p = self.panel()
        for value in ('nan','inf','1.5',''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.start_pan.set(value); p.start_values()

    def test_stop_loading_then_late_completion_does_not_prepare_hardware(self):
        p = self.panel(); p.preparing='model'; p.work_kind='prepare'; p.work=Future()
        p.prepare_hardware=Mock()
        p.stop()
        p.work.set_result((Mock(),Mock(),Mock(),{}))
        p._poll()
        p.prepare_hardware.assert_not_called(); p.app.send.assert_not_called()
        self.assertFalse(p.hardware_ready); self.assertIsNone(p.work)

    def test_start_drains_preview_before_starting_episode(self):
        p=self.panel(); p.hardware_ready=True; p.previewing=True
        p.check_camera=Mock(); p.stop_other_panels=Mock(); p._start_episode=Mock()
        p.inference=Future()
        p.start()
        self.assertTrue(p._start_requested); self.assertFalse(p.previewing)
        p._start_episode.assert_not_called()
        p._start_episode.side_effect=lambda:setattr(p,'_start_requested',False)
        p.inference.set_result((dict(image=object()),object(),(None,{}),p._generation))
        p._poll()
        p._start_episode.assert_called_once()

    def test_failed_immediate_start_does_not_leave_retry_pending(self):
        p=self.panel(); p.hardware_ready=True
        p.check_camera=Mock(); p.stop_other_panels=Mock()
        p._start_episode=Mock(side_effect=ValueError('connection changed'))
        with self.assertRaisesRegex(ValueError,'connection changed'): p.start()
        self.assertFalse(p._start_requested)

    def test_ook_busy_blocks_prepare_before_loading_or_movement(self):
        p=self.panel(); p.app.ook.busy=True
        with self.assertRaisesRegex(ValueError,'OOK'): p.prepare()
        p.learn_pool.submit.assert_not_called(); p.app.send.assert_not_called()

    def test_manual_end_requests_drain_without_emergency_cancel(self):
        p=self.panel(); p.run=self.fake_run(); p.end_episode()
        p.run.manual_end.assert_called_once(); p.run.abort.assert_not_called()
        self.assertFalse(p.learner.cancel.is_set())

    def test_episode_save_waits_for_inflight_frame_write(self):
        p=self.panel(); p.run=self.fake_run('saving'); p.run.save_request={'episode':14}
        p.inference=Future()
        p._poll(); p.learn_pool.submit.assert_not_called()
        p.inference.set_result((dict(image=object()),object(),(None,{}),p._generation-1))
        p._poll()
        p.learn_pool.submit.assert_called_once_with(p._finish_and_plot,{'episode':14})
        self.assertIsNone(p.run.save_request)

    def test_failed_save_submission_retains_summary_without_retry_loop(self):
        p=self.panel(); p.run=self.fake_run('saving'); summary={'episode':14}; p.run.save_request=summary
        p.learn_pool.submit.side_effect=RuntimeError('pool closed')
        with self.assertRaisesRegex(RuntimeError,'pool closed'): p._poll()
        self.assertEqual(p.run.save_request,summary)
        p._poll()
        self.assertEqual(p.learn_pool.submit.call_count,1)

    def test_graph_error_keeps_completed_training_report(self):
        p=self.panel(); p.run=self.fake_run('saving'); p.log=Mock()
        p.learner.finish_episode.return_value={'completed_updates':2}
        p.log.generate_plots.side_effect=OSError('graph disk failure')
        report,plots=p._finish_and_plot({'episode':14})
        self.assertEqual(report,{'completed_updates':2}); self.assertIn('graph disk failure',plots['errors'][0])

    def test_unexpected_partial_update_cannot_trigger_return(self):
        p=self.panel(); p.run=self.fake_run('saving'); p.work_kind='save'; p.work=Future()
        p.work.set_result(({'completed_updates':1,'planned_updates':2},{'files':[],'errors':[]}))
        with self.assertRaisesRegex(RuntimeError,'업데이트 횟수'): p._poll()
        p.run.saved.assert_called_once_with(False)
        self.assertFalse(p.hardware_ready); p.app.send.assert_not_called()

    def test_completed_save_selects_new_checkpoint_and_preserves_source_provenance(self):
        p=self.panel(); p.run=self.fake_run('saving'); p.log=SimpleNamespace(folder=Path('captures/M5-3/live'))
        p.identity['source_checkpoint']='original-checkpoint'
        p.work_kind='save'; p.work=Future()
        p.work.set_result((dict(episode=14,reason='manual_episode_end',completed_updates=2,planned_updates=2,
            checkpoint='captures/M5-3/live/checkpoints/episode_000014'),{'files':[],'errors':[]}))
        p._poll()
        self.assertTrue(p.checkpoint.get().endswith('live/checkpoints/episode_000014'))
        self.assertEqual(p.identity['checkpoint'],p.selection()['checkpoint'])
        self.assertEqual(p.identity['source_checkpoint'],'original-checkpoint')
        p.run.saved.assert_called_once_with()

    def test_evaluation_does_not_change_selected_checkpoint(self):
        p=self.panel(); p.learner.mode='evaluate'; original=p.checkpoint.get()
        p.select_saved_checkpoint(dict(checkpoint='ignored'))
        self.assertEqual(p.checkpoint.get(),original)

    def test_core_abort_save_disables_start_until_prepared_again(self):
        p=self.panel(); p.run=self.fake_run('saving'); p.run.stop_requested=True
        p.run.stop_detail='frame_command_mismatch'
        p.progress.set('Episode 14 · saving')
        p.hardware_ready=p.previewing=True
        p.log=SimpleNamespace(folder=Path('captures/M5-3/live'))
        p.work_kind='save'; p.work=Future()
        p.work.set_result((dict(episode=14,reason='observation_timeout',completed_updates=0,planned_updates=0),
                           {'files':[],'errors':[]}))
        p.run.saved.side_effect=lambda: (setattr(p.run,'phase','error'),setattr(p.run,'active',False))
        p._poll()
        self.assertFalse(p.hardware_ready); self.assertFalse(p.previewing)
        self.assertIn('자동 준비',p.status.get()); p.app.send.assert_not_called()
        self.assertIn('observation_timeout (frame_command_mismatch)',p.result_text.get())
        self.assertIn('중단 저장 완료',p.progress.get()); self.assertNotIn('saving',p.progress.get())

    def test_log_failure_still_disables_movement(self):
        p=self.panel(); p.run=self.fake_run(); p.run.abort.side_effect=OSError('disk full')
        p.hardware_ready=p.previewing=True; p.app.record.side_effect=OSError('disk full')
        with patch('Tx.Controller.moving_panel.messagebox.showerror'):
            p.fail(OSError('disk full'))
        self.assertFalse(p.hardware_ready); self.assertFalse(p.previewing)
        self.assertTrue(p.run.stop_requested); self.assertTrue(p.learner.cancel.is_set())

    def test_stop_invalidates_late_inference_delivery(self):
        p=self.panel(); p.run=self.fake_run(); p.run.inference_allowed=True; p.inference=Future()
        generation=p._generation
        p.stop()
        p.inference.set_result((dict(image=object()),object(),(None,{}),generation))
        p._poll()
        p.run.supply.assert_not_called(); self.assertTrue(p.learner.cancel.is_set())
        p.run.abort.assert_called_once_with('user_stop')

    def test_foreign_move_invalidates_prepared_idle_state(self):
        p=self.panel(); p.hardware_ready=True; p.previewing=True
        p.handle_event(dict(event='servo',operation='move',request_id='another-tab'))
        self.assertFalse(p.hardware_ready); self.assertFalse(p.previewing)
        p.app.send.assert_not_called()

    def test_leaving_never_prepared_tab_is_noop(self):
        p=self.panel(); p.stop=Mock(); p.on_tab_left(); p.stop.assert_not_called()

    def test_fresh_camera_and_matching_limits_required_for_preparation(self):
        p=self.panel(); p.preparing='limits'; p.setup_pending='m5-1-moving-setup-test'
        p.handle_event(dict(event='servo',operation='servo_config',request_id=p.setup_pending,
            available=True,simulated=False,limits=p.cfg.limits))
        self.assertEqual(p.preparing,'frames'); self.assertFalse(p.hardware_ready)
        p.camera_requested_at=10.; p.setup_deadline=30.
        p.app.current=(b'',dict(simulated=False,requested=dict(TX_CAMERA_SETTINGS)),11.,0)
        with patch('Tx.Controller.moving_panel.time.monotonic',return_value=11.1): p.prepare_poll()
        self.assertTrue(p.hardware_ready); self.assertTrue(p.previewing)


if __name__ == '__main__': unittest.main()
