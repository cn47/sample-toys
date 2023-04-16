import shutil
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Pool


def rm_files(target_file: str) -> None:
    try:
        shutil.rmtree(target_file)
    except OSError as e:
        pass


@contextmanager
def timer(name: str, logger=None):
    started_at = datetime.now()
    yield
    elapsed = str(datetime.now() - started_at).split(".")[0]
    if logger:
        logger.info(f"[{name}] done in [{elapsed}]")
    else:
        print(f"[{name}] done in [{elapsed}]")
