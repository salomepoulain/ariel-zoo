import time
from datetime import datetime
from functools import wraps

from ea.config import config


def time_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time

        # Write to file instead of console
        log_file = config.OUTPUT_FOLDER / "time_logs.txt"
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {func.__name__} took {duration:.4f}s\n")

        return result
    return wrapper
