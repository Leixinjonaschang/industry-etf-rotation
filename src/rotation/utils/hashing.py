"""缓存键：配置哈希与源文件指纹。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rotation.utils.io import _to_builtin


def stable_hash(obj: Any, length: int = 12) -> str:
    payload = json.dumps(_to_builtin(obj), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def directory_fingerprint(path: Path, pattern: str = "*.csv") -> dict[str, Any]:
    files = sorted(path.glob(pattern))
    digest = hashlib.sha1()
    for file in files:
        stat = file.stat()
        digest.update(f"{file.name}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return {"dir": path.name, "count": len(files), "digest": digest.hexdigest()[:16]}
