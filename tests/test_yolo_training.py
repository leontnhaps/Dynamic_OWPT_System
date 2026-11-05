"""Exercise data integrity and training orchestration without a GPU or real model updates."""
import builtins
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from common.model_paths import DEFAULT_YOLO_PATH, REPOSITORY_ROOT
from tools import train_yolo


class FakeBackend:
    """Emulate the public calls, including inference fusion and saved best weights."""
    def __init__(self, original, fail_training=False, names=None, architecture_error=None,
                 mutate_architecture=False, version='8.4.201'):
        self.original = Path(original)
        self.fail_training = fail_training
        self.instances = []
        self.validations = []
        self.training_calls = []
        self.architecture_calls = []
        self.torch = ModuleType('torch')
        self.torch.__version__ = 'test-torch'
        self.torch.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        self.ultralytics = ModuleType('ultralytics')
        self.ultralytics.__version__ = version
        self.nn = ModuleType('ultralytics.nn')
        self.tasks = ModuleType('ultralytics.nn.tasks')
        backend = self

        class Model:
            def __init__(self, path):
                self.path = Path(path)
                self.task = 'detect'
                self.names = {0: 'pv_cell'} if names is None else names
                self.model = SimpleNamespace(yaml={
                    'nc': 1, 'backbone': [[-1, 1, 'SPPF', [256, 5, 3, True]]], 'head': []})
                self.fused = False
                backend.instances.append(self)

            def val(self, **kwargs):
                self.fused = True
                backend.validations.append((self, kwargs.copy()))
                score = .25 if self.path == backend.original else .75
                return SimpleNamespace(box=SimpleNamespace(mp=score, mr=score+.1,
                                                           map50=score+.05, map=score-.05))

            def train(self, **kwargs):
                if self.fused:
                    raise AssertionError('Training reused a model already fused for inference')
                backend.training_calls.append((self, kwargs.copy()))
                if backend.fail_training:
                    raise RuntimeError('simulated training failure')
                best = Path(kwargs['project']) / kwargs['name'] / 'weights' / 'best.pt'
                best.parent.mkdir(parents=True)
                best.write_bytes(b'new weights from fake backend')
                self.trainer = SimpleNamespace(best=best)

        self.ultralytics.YOLO = Model

        def detection_model(cfg, *, nc, ch, verbose):
            backend.architecture_calls.append(dict(cfg=cfg, original_cfg=deepcopy(cfg),
                                                   nc=nc, ch=ch, verbose=verbose))
            if mutate_architecture:
                cfg['backbone'][0][3].append('constructor mutation')
            if architecture_error is not None:
                raise architecture_error
            return SimpleNamespace()

        self.tasks.DetectionModel = detection_model
        self.nn.tasks = self.tasks
        self.ultralytics.nn = self.nn

    def installed(self):
        return patch.dict(sys.modules, {'torch': self.torch, 'ultralytics': self.ultralytics,
                                       'ultralytics.nn': self.nn, 'ultralytics.nn.tasks': self.tasks})


class YoloTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'dataset'
        self.data.mkdir()
        self.yaml_path = self.data / 'data.yaml'
        self.yaml_path.write_text('train: images/train\nval: images/val\nnames:\n  0: pv_cell\n', encoding='utf-8')
        for split_index, split in enumerate(('train', 'val')):
            (self.data / 'images' / split).mkdir(parents=True)
            (self.data / 'labels' / split).mkdir(parents=True)
            for index, name in enumerate(('positive', 'background')):
                Image.new('RGB', (32, 24), (40 + 70*split_index, 30+90*index, 80)).save(
                    self.data / 'images' / split / f'{name}.png')
                text = '0 0.5 0.5 0.2 0.3\n' if name == 'positive' else ''
                (self.data / 'labels' / split / f'{name}.txt').write_text(text, encoding='utf-8')
        self.weights = self.root / 'existing.pt'
        self.weights.write_bytes(b'original model bytes')
        self.output = self.root / 'runs'

    def args(self, *extra):
        return train_yolo.parser().parse_args([
            '--model', str(self.weights), '--data', str(self.yaml_path),
            '--output', str(self.output), '--device', 'cpu', *extra])

    def run_quietly(self, args):
        with redirect_stdout(io.StringIO()):
            return train_yolo.run(args)

    def test_shared_defaults_and_relative_paths_ignore_launch_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            args = train_yolo.parser().parse_args([])
            self.assertEqual(args.model, DEFAULT_YOLO_PATH)
            self.assertEqual(args.data, REPOSITORY_ROOT / 'captures/M5-3/yolo_outdoor_dataset/data.yaml')
            self.assertEqual(train_yolo.repository_path('captures/example'), REPOSITORY_ROOT / 'captures/example')
            _, _, _, dataset, _, _ = train_yolo.prepare(self.args())
            self.assertEqual(Path(dataset['train']), self.data / 'images/train')
        finally:
            os.chdir(previous)
        result = subprocess.run([sys.executable, str(REPOSITORY_ROOT / 'tools/train_yolo.py'), '--help'],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--check-only', result.stdout)

    def test_valid_dataset_keeps_reviewed_backgrounds_and_provenance(self):
        _, _, _, resolved, rows, summary = train_yolo.prepare(self.args())
        self.assertEqual(resolved['names'], {0: 'pv_cell'})
        self.assertEqual(len(rows), 4)
        for split in ('train', 'val'):
            self.assertEqual(summary[split], dict(images=2, positive=1, negative=1, boxes=1))
        self.assertEqual(len({row['image_sha256'] for row in rows}), 4)
        self.assertTrue(all(len(row['label_sha256']) == 64 for row in rows))

    def test_missing_label_is_not_silently_treated_as_background(self):
        (self.data / 'labels/train/background.txt').unlink()
        with self.assertRaisesRegex(ValueError, '라벨 파일'):
            train_yolo.prepare(self.args())

    def test_rejects_malformed_detection_labels(self):
        label = self.data / 'labels/train/positive.txt'
        for text in ('0 .5 .5 .2', '1 .5 .5 .2 .2', '0 nan .5 .2 .2',
                     '0 .5 .5 .2 0', '0 .99 .5 .2 .2', '0 .5 .5 -.1 .2'):
            with self.subTest(text=text):
                label.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    train_yolo.prepare(self.args())

    def test_rejects_wrong_dataset_class(self):
        self.yaml_path.write_text('train: images/train\nval: images/val\nnames: [drone]\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'pv_cell'):
            train_yolo.prepare(self.args())

    def test_rejects_duplicate_contents_across_splits_even_with_different_names(self):
        original = self.data / 'images/train/positive.png'
        (self.data / 'images/val/positive.png').write_bytes(original.read_bytes())
        with self.assertRaisesRegex(ValueError, '중복'):
            train_yolo.prepare(self.args())

    def test_rejects_missing_weights_before_creating_output(self):
        self.weights.unlink()
        with self.assertRaisesRegex(ValueError, '모델'):
            self.run_quietly(self.args())
        self.assertFalse(self.output.exists())

    def test_check_only_does_not_import_gpu_libraries_or_create_outputs(self):
        imported = builtins.__import__

        def reject_training_modules(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'ultralytics'}:
                raise AssertionError(f'Unexpected training import: {name}')
            return imported(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=reject_training_modules):
            self.assertIsNone(self.run_quietly(self.args('--check-only')))
        self.assertFalse(self.output.exists())
        self.assertEqual(self.weights.read_bytes(), b'original model bytes')

    def test_finetune_reloads_weights_and_compares_on_identical_validation_settings(self):
        backend = FakeBackend(self.weights)
        default_before = train_yolo.DEFAULT_YOLO_PATH
        with backend.installed():
            run_dir = self.run_quietly(self.args('--lr0', '0.0002', '--epochs', '2'))
        self.assertEqual(self.weights.read_bytes(), b'original model bytes')
        self.assertEqual(train_yolo.DEFAULT_YOLO_PATH, default_before)
        self.assertEqual(len(backend.instances), 3)
        self.assertIsNot(backend.validations[0][0], backend.training_calls[0][0])
        self.assertEqual(backend.training_calls[0][0].path, self.weights)
        train = backend.training_calls[0][1]
        self.assertFalse(train['resume'])
        self.assertEqual(train['optimizer'], 'AdamW')
        self.assertEqual(train['lr0'], .0002)
        self.assertEqual(train['epochs'], 2)
        before_args, after_args = [call[1].copy() for call in backend.validations]
        self.assertEqual(before_args.pop('name'), 'before')
        self.assertEqual(after_args.pop('name'), 'after')
        self.assertEqual(before_args, after_args)
        self.assertEqual(before_args['split'], 'val')
        self.assertFalse(before_args['augment'])
        saved_data = yaml.safe_load((run_dir / 'data.yaml').read_text())
        self.assertTrue(all(Path(saved_data[s]).is_absolute() for s in ('train', 'val')))
        saved = json.loads((run_dir / 'config.json').read_text())
        self.assertEqual(saved['status'], 'completed')
        self.assertTrue(saved['original_model_unchanged'])
        comparison = json.loads((run_dir / 'comparison.json').read_text())
        self.assertAlmostEqual(comparison['delta']['mAP50'], .5)
        self.assertNotEqual(Path(comparison['best_model']), self.weights)
        with (run_dir / 'comparison.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual([r['stage'] for r in csv.DictReader(stream)], ['before', 'after', 'delta'])
        with (run_dir / 'dataset_manifest.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_repeated_runs_preserve_prior_results(self):
        backend = FakeBackend(self.weights)
        with backend.installed():
            first = self.run_quietly(self.args())
            first_contents = (first / 'comparison.json').read_bytes()
            second = self.run_quietly(self.args())
        self.assertNotEqual(first, second)
        self.assertEqual((first / 'comparison.json').read_bytes(), first_contents)
        self.assertTrue((second / 'comparison.json').is_file())

    def test_training_failure_preserves_baseline_and_records_failed_status(self):
        backend = FakeBackend(self.weights, fail_training=True)
        with backend.installed(), self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.run_quietly(self.args())
        run_dir, = self.output.iterdir()
        config = json.loads((run_dir / 'config.json').read_text())
        self.assertEqual(config['status'], 'failed')
        self.assertTrue(config['original_model_unchanged'])
        self.assertIn('simulated training failure', config['error'])
        self.assertTrue((run_dir / 'before.json').is_file())
        self.assertFalse((run_dir / 'comparison.json').exists())
        self.assertEqual(self.weights.read_bytes(), b'original model bytes')

    def test_wrong_checkpoint_class_is_rejected_before_creating_results(self):
        backend = FakeBackend(self.weights, names={0: 'drone'})
        with backend.installed(), self.assertRaisesRegex(ValueError, 'pv_cell'):
            self.run_quietly(self.args())
        self.assertFalse(self.output.exists())
        self.assertFalse(backend.training_calls)

    def test_sppf_constructor_mismatch_stops_before_evaluation_and_outputs(self):
        underlying = 'SPPF.__init__() takes from 3 to 4 positional arguments but 6 were given'
        backend = FakeBackend(self.weights, architecture_error=TypeError(underlying), version='8.3.203')
        with backend.installed(), self.assertRaises(ValueError) as caught:
            self.run_quietly(self.args())
        message = str(caught.exception)
        self.assertIn('8.3.203', message)
        self.assertIn('SPPF', message)
        self.assertIn(underlying, message)
        self.assertIn('-m pip install', message)
        self.assertIn('ultralytics', message)
        self.assertFalse(self.output.exists())
        self.assertFalse(backend.validations)
        self.assertFalse(backend.training_calls)
        self.assertEqual(len(backend.architecture_calls), 1)
        self.assertEqual(self.weights.read_bytes(), b'original model bytes')

    def test_architecture_preflight_uses_a_deep_copy_without_changing_checkpoint_config(self):
        backend = FakeBackend(self.weights, mutate_architecture=True)
        with backend.installed():
            run_dir = self.run_quietly(self.args())
        self.assertEqual(len(backend.architecture_calls), 1)
        call = backend.architecture_calls[0]
        self.assertEqual(call['nc'], 1)
        self.assertEqual(call['ch'], 3)
        self.assertFalse(call['verbose'])
        self.assertEqual(backend.instances[0].model.yaml, call['original_cfg'])
        self.assertIsNot(call['cfg'], backend.instances[0].model.yaml)
        self.assertNotEqual(call['cfg'], backend.instances[0].model.yaml)
        self.assertEqual(self.weights.read_bytes(), b'original model bytes')
        self.assertTrue((run_dir / 'comparison.json').is_file())
        config = json.loads((run_dir / 'config.json').read_text())
        self.assertEqual(config['compatibility'], {'architecture_reconstructed': True, 'device': 'cpu'})


if __name__ == '__main__':
    unittest.main()
