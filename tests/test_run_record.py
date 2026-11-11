import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from notebook_runtime import extract_raw_dataset, fetch_archive, sha256, training_summary, update_run


class RunRecordTests(unittest.TestCase):
    def test_update_merges_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'run.json').write_text(json.dumps({'status': 'running', 'config': {'SEED': 1}}))
            record = update_run(tmp, status='completed', test={'metrics': {'Bleu_4': 0.5}})
            self.assertEqual(record, json.loads((Path(tmp) / 'run.json').read_text()))
            self.assertEqual(record['status'], 'completed')
            self.assertEqual(record['config'], {'SEED': 1})

    def test_training_summary_selects_best_validation_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training_metrics.jsonl'
            path.write_text(''.join(json.dumps({'epoch': epoch, 'metrics': {'Bleu_4': score}}) + '\n'
                                    for epoch, score in ((1, 0.2), (2, 0.6), (3, 0.5))))
            self.assertEqual(training_summary(path),
                             {'epochs_completed': 3, 'best_epoch': 2, 'best_validation': {'Bleu_4': 0.6}})


class DatasetArchiveTests(unittest.TestCase):
    def test_local_archive_is_copied_hashed_and_located(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / 'source.zip', 'w') as archive:
                archive.writestr('LEVIR/LevirCCcaptions.json', '{}')
                archive.writestr('LEVIR/images/train/A/x.png', 'x')
            copied, digest = fetch_archive(root / 'source.zip', root / 'copy.zip', root / 'data.log')
            self.assertEqual(digest, sha256(root / 'source.zip'))
            self.assertEqual(extract_raw_dataset(copied, root / 'raw'), root / 'raw' / 'LEVIR')

    def test_archive_without_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / 'source.zip', 'w') as archive:
                archive.writestr('readme.txt', 'x')
            with self.assertRaises(FileNotFoundError):
                extract_raw_dataset(root / 'source.zip', root / 'raw')


if __name__ == '__main__':
    unittest.main()
