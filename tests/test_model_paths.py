import os
import tempfile
import unittest
from common.model_paths import DEFAULT_YOLO_PATH, REPOSITORY_ROOT, resolve_yolo_path


class ModelPathTests(unittest.TestCase):
    def test_default_independent_of_launch_directory(self):
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                for value in (None, '', '  '):
                    self.assertEqual(resolve_yolo_path(value), str(DEFAULT_YOLO_PATH))
                self.assertEqual(resolve_yolo_path('captures/M1-3/pv_detection/runs/y26n_v1/weights/best.pt'),str(DEFAULT_YOLO_PATH))
        finally:
            os.chdir(previous)

    def test_explicit_model_selection(self):
        self.assertEqual(resolve_yolo_path('other/model.pt'),str(REPOSITORY_ROOT/'other/model.pt'))
        self.assertEqual(resolve_yolo_path(DEFAULT_YOLO_PATH),str(DEFAULT_YOLO_PATH))
