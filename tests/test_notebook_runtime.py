import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from notebook_runtime import required_data_files, restore_preprocessed_data, run_logged


class NotebookRuntimeTests(unittest.TestCase):
    def test_output_is_displayed_before_process_finishes_and_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / 'done'
            code = ('import time; from pathlib import Path; '
                    'print("training batch 1", flush=True); time.sleep(0.3); '
                    f'Path({str(done)!r}).touch(); print("validation score", flush=True)')
            observed = []

            def display(*args, **kwargs):
                if args and str(args[0]).startswith('training batch 1'):
                    self.assertFalse(done.exists(), 'First batch was buffered until completion')
                    observed.append(True)

            with patch('builtins.print', side_effect=display):
                run_logged([sys.executable, '-u', '-c', code], root / 'train.log')
            self.assertEqual(observed, [True])
            self.assertIn('training batch 1\nvalidation score', (root / 'train.log').read_text())

    def test_stderr_and_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / 'eval.log'
            with self.assertRaises(subprocess.CalledProcessError):
                run_logged([sys.executable, '-u', '-c',
                            'import sys; print("evaluation failed", file=sys.stderr); sys.exit(2)'], path)
            self.assertIn('evaluation failed', path.read_text())

    def test_flat_and_nested_prepared_archives(self):
        for prefix in ('', 'data/', 'custom_variant/training_files/'):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / 'prepared.zip'
                with zipfile.ZipFile(archive, 'w') as zipped:
                    for path in required_data_files(root, 'custom'):
                        zipped.writestr(prefix + path.name, b'prepared contents')
                destination = root / 'output'
                destination.mkdir()
                (destination / 'note.txt').touch()
                restore_preprocessed_data(archive, destination, 'custom')
                self.assertTrue(all(p.is_file() for p in required_data_files(destination, 'custom')))

    def test_incomplete_archive_does_not_install_partial_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'incomplete.zip'
            with zipfile.ZipFile(archive, 'w') as zipped:
                zipped.writestr('WORDMAP_custom.json', '{}')
            destination = root / 'output'
            with self.assertRaisesRegex(RuntimeError, 'found 0'):
                restore_preprocessed_data(archive, destination, 'custom')
            self.assertEqual(list(destination.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
