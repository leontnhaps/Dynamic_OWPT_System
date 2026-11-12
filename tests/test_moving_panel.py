from concurrent.futures import Future
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common.tx_moving import MovingConfig
from common.tx_tracking import Sample
from common.tx_setup import TX_CAMERA_SETTINGS
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
        p.identity = p.selection()
        return p

    def fake_run(self, phase='ready', **kwargs):
        r = SimpleNamespace(phase=phase, active=True, pending=None, save_request=None, episode=14,
            steps=2, new_transitions=1, inference_allowed=False, command=Mock(), tick=Mock(),
            set_start=Mock(), manual_end=Mock(), abort=Mock(), saved=Mock(), supply=Mock(),
            stop_requested=False, settings=MovingRunSettings(), **kwargs)
        r.command.tolist.return_value = [0,0]
        return r

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
        p.hardware_ready=p.previewing=True
        p.log=SimpleNamespace(folder=Path('captures/M5-3/live'))
        p.work_kind='save'; p.work=Future()
        p.work.set_result((dict(episode=14,reason='ack_timeout',completed_updates=0,planned_updates=0),
                           {'files':[],'errors':[]}))
        p.run.saved.side_effect=lambda: (setattr(p.run,'phase','error'),setattr(p.run,'active',False))
        p._poll()
        self.assertFalse(p.hardware_ready); self.assertFalse(p.previewing)
        self.assertIn('자동 준비',p.status.get()); p.app.send.assert_not_called()

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
