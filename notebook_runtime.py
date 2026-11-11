"""Colab output streaming and explicit preprocessed-data restoration."""
from pathlib import Path
import codecs
import shutil
import subprocess
import zipfile


def run_logged(command, log_path, cwd=None):
    """Display merged stdout/stderr and persist it; stop the child on interruption."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8', newline='') as log:
        log.write(f"\nStarting: {command!r}\n")
        log.flush()
        # Binary reads preserve tqdm's carriage returns; text mode translates
        # them into newlines and makes every progress update a separate line.
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            try:
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                while chunk := process.stdout.read1(8192):
                    output = decoder.decode(chunk)
                    print(output, end='', flush=True)
                    log.write(output)
                    log.flush()
                output = decoder.decode(b'', final=True)
                if output:
                    print(output, end='', flush=True)
                    log.write(output)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def required_data_files(directory, data_name):
    directory = Path(directory)
    files = [directory / f'WORDMAP_{data_name}.json']
    for split in ('TRAIN', 'VAL', 'TEST'):
        files.extend(directory / f'{split}_{kind}_{data_name}.{extension}'
                     for kind, extension in (('IMAGES', 'hdf5'),
                                             ('CAPTIONS', 'json'), ('CAPLENS', 'json')))
    return files


def restore_preprocessed_data(archive, destination, data_name):
    """Select one complete data folder, supporting flat or nested ZIP layouts."""
    import tempfile
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in required_data_files(destination, data_name)):
        raise FileExistsError('Destination already contains generated data files; use a fresh run.')
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        staging = Path(staging)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(staging)
        candidates = [p.parent for p in staging.rglob(f'WORDMAP_{data_name}.json')
                      if all(f.is_file() for f in required_data_files(p.parent, data_name))]
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one complete preprocessed dataset for {data_name}; found {len(candidates)}.')
        for path in required_data_files(candidates[0], data_name):
            shutil.move(str(path), destination / path.name)
        metadata = candidates[0] / 'preprocessing_config.json'
        if metadata.is_file():
            shutil.move(str(metadata), destination / metadata.name)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(source, destination, log_path):
    """Copy a local/Drive ZIP or download a share URL; returns (path, sha256)."""
    import sys
    source, destination = str(source), Path(destination)
    if Path(source).is_file():
        shutil.copy2(source, destination)
    elif source.startswith(('https://', 'http://')):
        gdown = [sys.executable, '-u', '-m', 'gdown']
        # Older gdown needs --fuzzy for share links; newer versions parse them and reject the flag.
        if '--fuzzy' in subprocess.check_output(gdown + ['--help'], text=True):
            gdown.append('--fuzzy')
        run_logged(gdown + [source, '-O', str(destination)], log_path)
    else:
        raise FileNotFoundError(f'Expected an existing ZIP path or download URL: {source}')
    return destination, sha256(destination)


def extract_raw_dataset(archive, destination):
    """Extract a LEVIR-CC-layout ZIP; returns the folder holding the captions file and images/."""
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(destination)
    candidates = [p.parent for p in Path(destination).rglob('LevirCCcaptions.json')
                  if (p.parent / 'images').is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one LevirCCcaptions.json next to images/; found {len(candidates)}.')
    return candidates[0]


def start_run(run_dir, config, repo_dir='.'):
    """Create the run folder and its single record, run.json."""
    import json
    import platform
    import sys
    from datetime import datetime, timezone
    import torch
    import torchvision

    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=repo_dir, text=True).strip()

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {
        'run_id': run_dir.name, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'code': {'repo_url': git('remote', 'get-url', 'origin'), 'commit': git('rev-parse', 'HEAD'),
                 'uncommitted_changes': bool(git('status', '--porcelain'))},
        'config': config,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'torchvision': torchvision.__version__, 'cuda_build': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    (run_dir / 'run.json').write_text(json.dumps(record, indent=2, default=str))
    (run_dir / 'packages.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    return record


def update_run(run_dir, **fields):
    """Merge fields into run.json and return the updated record."""
    import json
    path = Path(run_dir) / 'run.json'
    record = json.loads(path.read_text())
    record.update(fields)
    path.write_text(json.dumps(record, indent=2, default=str))
    return record


def training_summary(metrics_path):
    """Best-validation epoch and epoch count from training_metrics.jsonl."""
    import json
    records = [json.loads(line) for line in Path(metrics_path).read_text().splitlines() if line.strip()]
    best = max(records, key=lambda r: r['metrics']['Bleu_4'])
    return {'epochs_completed': records[-1]['epoch'], 'best_epoch': best['epoch'],
            'best_validation': best['metrics']}
