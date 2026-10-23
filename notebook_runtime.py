"""Colab output streaming and explicit preprocessed-data restoration."""
from pathlib import Path
import shutil
import subprocess
import zipfile


def run_logged(command, log_path, cwd=None):
    """Display merged stdout/stderr and persist it; stop the child on interruption."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting: {' '.join(map(str, command))}", flush=True)
    print(f"Log: {log_path}", flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f"\nStarting: {command!r}\n")
        log.flush()
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
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
    print('Command completed.', flush=True)


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
