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
