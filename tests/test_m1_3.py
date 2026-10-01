import csv
import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from Tx.Controller.detection_panel import PVDetector, DetectionPanel


class DetectionTests(unittest.TestCase):
    def panel(self):
        p = DetectionPanel.__new__(DetectionPanel)
        p.app = SimpleNamespace(fps=30, record=Mock(), send=Mock())
        p.status = Mock(); p.canvas = Mock(); p.processed = 0; p.started = 9
        p.log = io.StringIO(); p.writer = csv.DictWriter(p.log, fieldnames=['receive_unix_ns','seq','simulated','width','height','status','count','confidence','u','v','x1','y1','x2','y2','inference_ms','result_age_s','receive_fps','detection_fps'])
        p.writer.writeheader()
        return p

    @patch('Tx.Controller.detection_panel.ImageTk.PhotoImage', return_value=object())
    def test_detect_missing_and_stale_clear_coordinates(self, _):
        p = self.panel()
        result = dict(image=Image.new('RGB',(1296,972)), target=dict(center=[100,200],box=[50,100,150,300],confidence=.9),count=1,inference_ms=20)
        frame = (b'jpeg', {'seq':1,'simulated':False}, 10, 123)
        p.consume(result, frame, 10.1)
        self.assertEqual(p.latest[2]['u'],100)
        result['target'] = None; result['count'] = 0
        p.consume(result, frame, 10.2)
        self.assertNotIn('u', p.latest[2])
        p.consume(result, frame, 13)
        self.assertIsNone(p.latest)
        rows=list(csv.DictReader(io.StringIO(p.log.getvalue())))
        self.assertEqual([r['status'] for r in rows], ['detected','missing','stale'])
        self.assertEqual(rows[1]['u'],'')
        p.app.send.assert_not_called()

    def test_detector_uses_original_rgb_and_highest_confidence(self):
        detector = PVDetector.__new__(PVDetector)
        detector.device = 'cpu'; detector.model = Mock()
        boxes=Mock()
        boxes.data.detach.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [[10,20,30,40,.6,0],[100,200,200,300,.95,0]]
        detector.model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        raw=io.BytesIO(); Image.new('RGB',(1296,972)).save(raw,format='JPEG')
        result=detector.infer(raw.getvalue(),.5,0)
        self.assertEqual(result['target']['center'],[150,250])
        self.assertEqual(result['count'],2)
        source=detector.model.predict.call_args.kwargs['source']
        self.assertEqual(source.size,(1296,972)); self.assertEqual(source.mode,'RGB')

    def test_save_matched_frame_and_overlay(self):
        p=self.panel(); p.loaded={'path':'pv.pt','device':'cpu'}; p.settings=(.5,0)
        raw=io.BytesIO(); Image.new('RGB',(100,80)).save(raw,format='JPEG')
        p.latest=(Image.new('RGB',(100,80)), (raw.getvalue(),{'seq':7},time.monotonic(),123),{'seq':7})
        with tempfile.TemporaryDirectory() as directory:
            p.app.stage_dir=Mock(return_value=Path(directory))
            p.save()
            p.app.stage_dir.assert_called_once_with('M1-3')
            self.assertEqual(len(list(Path(directory).glob('*.jpg'))),2)
            self.assertEqual(len(list(Path(directory).glob('*.json'))),1)
        p.app.send.assert_not_called()

    def test_missing_model_fails_before_import(self):
        with self.assertRaises(ValueError): PVDetector('/nonexistent/pv.pt','cpu')


if __name__ == '__main__': unittest.main()
