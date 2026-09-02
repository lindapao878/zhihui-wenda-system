"""批量导入脚本：断点续传 + MD5 内容去重。

合并自 batch_import.py / batch_import_v2.py / batch_import_resume.py。
进度文件格式：每行 `路径\tMD5_hash`，兼容旧版仅路径格式。
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import argparse
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_ROOTS = [
    _PROJECT_ROOT / "data" / "import",
]
_SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".docx"}
_DEFAULT_EXCLUDE = ()
_progress_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 知识库批量导入（断点续传 + MD5 去重）")
    parser.add_argument(
        "--roots", nargs="*", default=None,
        help="扫描根目录，默认 data/import",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="导入 API 地址",
    )
    parser.add_argument(
        "--progress", default=str(_PROJECT_ROOT / "import_progress.txt"),
        help="断点进度文件",
    )
    parser.add_argument(
        "--log", default=str(_PROJECT_ROOT / "batch_import.log"),
        help="日志文件",
    )
    parser.add_argument(
        "--max-wait", type=int, default=1200,
        help="单个任务最大等待秒数，默认 1200",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅扫描统计，不执行导入",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="并发导入线程数，默认 2",
    )
    parser.add_argument(
        "--retry", type=int, default=2,
        help="失败重试次数，默认 2",
    )
    parser.add_argument(
        "--retry-delay", type=int, default=10,
        help="重试间隔秒数，默认 10",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=None,
        help="额外排除子串",
    )
    return parser.parse_args()


def log_msg(message: str, log_file: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with _progress_lock:
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def collect_files(roots: List[Path], exclude_substrs: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in _SUPPORTED_SUFFIXES
            and not any(token in str(p) for token in exclude_substrs)
        )
    return sorted(set(files), key=lambda p: str(p))


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_progress(progress_file: Path) -> Tuple[Set[str], Dict[str, str]]:
    """读取进度文件，返回 (已导入路径集合, {路径: hash})。
    兼容旧格式（仅路径，无 hash）和新格式（路径\t hash）。
    """
    done_paths: Set[str] = set()
    path_hashes: Dict[str, str] = {}
    if not progress_file.exists():
        return done_paths, path_hashes
    for line in progress_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
            p = parts[0].strip()
            h = parts[1].strip()
            done_paths.add(p)
            if h:
                path_hashes[p] = h
        else:
            done_paths.add(line)
    return done_paths, path_hashes


def append_progress(progress_file: Path, file_path: Path, file_hash: str) -> None:
    with _progress_lock:
        with progress_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{file_path}\t{file_hash}\n")


def upgrade_progress(progress_file: Path, path_hashes: Dict[str, str],
                     done_paths: Set[str]) -> None:
    """将旧格式进度文件（纯路径）升级为新格式（路径 + hash）。
    仅当存在无 hash 条目时触发。
    """
    needs_update = any(p not in path_hashes for p in done_paths)
    if not needs_update:
        return
    new_lines: List[str] = []
    for p_str in sorted(done_paths):
        h = path_hashes.get(p_str, "")
        if h:
            new_lines.append(f"{p_str}\t{h}")
        else:
            p = Path(p_str)
            if p.exists():
                h = file_md5(p)
                path_hashes[p_str] = h
                new_lines.append(f"{p_str}\t{h}")
            else:
                new_lines.append(p_str)
    with _progress_lock:
        progress_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def import_one(file_path: Path, url: str, max_wait: int) -> bool:
    with file_path.open("rb") as fh:
        resp = requests.post(
            f"{url}/upload",
            files={"file": (file_path.name, fh)},
            timeout=30,
        )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    for _ in range(max_wait):
        time.sleep(1)
        status_resp = requests.get(f"{url}/status/{task_id}", timeout=10)
        status_resp.raise_for_status()
        status = status_resp.json().get("status", "pending")
        if status in {"completed", "failed"}:
            return status == "completed"
    raise TimeoutError(f"task {task_id} timeout after {max_wait}s")


def main() -> None:
    args = parse_args()
    roots = [Path(r) for r in args.roots] if args.roots else _DEFAULT_ROOTS
    exclude = _DEFAULT_EXCLUDE + tuple(args.exclude or [])
    log_file = Path(args.log)
    progress_file = Path(args.progress)

    files = collect_files(roots, exclude)
    done_paths, path_hashes = read_progress(progress_file)
    upgrade_progress(progress_file, path_hashes, done_paths)

    pending: List[Tuple[Path, str]] = []
    skipped: int = 0
    for fpath in files:
        fpath_str = str(fpath)
        if fpath_str in done_paths:
            old_hash = path_hashes.get(fpath_str)
            if old_hash:
                current_hash = file_md5(fpath)
                if old_hash == current_hash:
                    skipped += 1
                    continue
                # 内容变了，重新导入
                pending.append((fpath, current_hash))
            else:
                # 旧格式无 hash，视为已导入但补算 hash
                current_hash = file_md5(fpath)
                path_hashes[fpath_str] = current_hash
                append_progress(progress_file, fpath, current_hash)
                skipped += 1
        else:
            current_hash = file_md5(fpath)
            pending.append((fpath, current_hash))

    log_msg(
        f"total={len(files)} skipped={skipped} pending={len(pending)}",
        log_file,
    )

    if args.dry_run:
        for fpath, _h in pending[:30]:
            log_msg(f"  would import: {fpath}", log_file)
        if len(pending) > 30:
            log_msg(f"  ... 及另外 {len(pending) - 30} 个文件", log_file)
        return

    def run_one(item, retry, retry_delay):
        fpath, fhash, idx = item
        for attempt in range(1, retry + 2):
            try:
                if import_one(fpath, args.url, args.max_wait):
                    tag = f"[retry {attempt - 1}]" if attempt > 1 else ""
                    append_progress(progress_file, fpath, fhash)
                    log_msg(f"[{idx}/{len(pending)}] OK{tag} {fpath.name}", log_file)
                    return True
                if attempt <= retry:
                    log_msg(f"[{idx}/{len(pending)}] 失败，{retry_delay}s 后重试({attempt}/{retry}) {fpath.name}", log_file)
                    time.sleep(retry_delay)
                    continue
                log_msg(f"[{idx}/{len(pending)}] FAILED {fpath.name} (重试{retry}次后仍失败)", log_file)
                return False
            except Exception as exc:
                if attempt <= retry:
                    log_msg(f"[{idx}/{len(pending)}] ERROR {fpath.name}: {exc}，{retry_delay}s 后重试({attempt}/{retry})", log_file)
                    time.sleep(retry_delay)
                    continue
                log_msg(f"[{idx}/{len(pending)}] ERROR {fpath.name}: {exc} (重试{retry}次后仍失败)", log_file)
                return False

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_one, (fpath, fhash, idx), args.retry, args.retry_delay)
            for idx, (fpath, fhash) in enumerate(pending, 1)
        ]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                log_msg(f"worker error: {exc}", log_file)
                results.append(False)

    ok = sum(results)
    failed = len(results) - ok
    log_msg(f"done ok={ok} failed={failed}", log_file)


if __name__ == "__main__":
    main()
