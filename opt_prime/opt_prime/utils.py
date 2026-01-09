from datetime import datetime
from collections import defaultdict

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(line: str) -> None:
    print(f"[{ts()}] {line}", flush=True)