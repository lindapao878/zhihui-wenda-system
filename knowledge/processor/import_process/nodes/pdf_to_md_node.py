"""PDF to Markdown conversion node using MinerU."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import (
    FileProcessingError,
    PdfConversionError,
    ValidationError,
)
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.logger_util import logger


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)
        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)

        state["md_path"] = self._get_md_paths(import_file_path, file_dir_path)
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)

        if not file_dir:
            file_dir = import_file_path_obj.parent

        file_dir_path_obj = Path(file_dir)
        logger.info("上传文件的路径:{}", import_file_path)
        logger.info("输出的目录:{}", file_dir)
        return import_file_path_obj, file_dir_path_obj

    def _execute_mineru(self, import_file_path: Path, file_dir_path: Path) -> int:
        cmd = [
            str(Path(sys.executable).parent / "mineru.exe"),
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir_path),
            "--source",
            "local",
        ]
        start_time = time.time()
        timeout_seconds = self.config.mineru_timeout_seconds

        mineru_env = {
            **os.environ,
            "MINERU_MODEL_SOURCE": os.getenv("MINERU_MODEL_SOURCE", "modelscope"),
            "MODELSCOPE_CACHE": os.getenv("MODELSCOPE_CACHE", ""),
            "HF_HOME": os.getenv("HF_HOME", ""),
            "MODELSCOPE_OFFLINE": os.getenv("MODELSCOPE_OFFLINE", "0"),
        }
        logger.info(
            "MinerU 环境变量: MINERU_MODEL_SOURCE={} MODELSCOPE_CACHE={} HF_HOME={}",
            mineru_env["MINERU_MODEL_SOURCE"],
            mineru_env["MODELSCOPE_CACHE"],
            mineru_env["HF_HOME"],
        )

        proc = subprocess.Popen(
            args=cmd,
            env=mineru_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        if proc.stdout:
            self._drain_stdout(proc)

        return self._wait_with_timeout(proc, timeout_seconds, start_time, import_file_path)

    def _drain_stdout(self, proc) -> None:
        """Stream MinerU stdout to logs from a daemon thread."""
        def _reader():
            for line in proc.stdout:
                logger.info("执行MinerU产生的日志：{}", line.rstrip())

        threading.Thread(target=_reader, daemon=True).start()

    def _wait_with_timeout(
        self,
        proc,
        timeout_seconds: int,
        start_time: float,
        import_file_path: Path,
    ) -> int:
        """Poll process until exit; on timeout kill the tree and raise PdfConversionError."""
        deadline = start_time + timeout_seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(1)
        else:
            self._kill_process_tree(proc)
            elapsed = time.time() - start_time
            raise PdfConversionError(
                f"MinerU解析PDF超时（>{timeout_seconds}s，实际等待 {elapsed:.1f}s），已终止进程树",
                self.name,
            )

        elapsed = time.time() - start_time
        if proc.returncode == 0:
            logger.info("MinerU成功解析PDF文件：{} 耗时:{:.2f}s", import_file_path.name, elapsed)
        else:
            logger.error("MinerU解析PDF文件：{}失败, code={}", import_file_path.name, proc.returncode)
        return proc.returncode

    @staticmethod
    def _kill_process_tree(proc) -> None:
        """Force kill on Windows, fall back to process.kill()."""
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, text=True, timeout=10,
            )
        else:
            proc.kill()

    def _get_md_paths(self, import_file_path: Path, file_dir_path: Path) -> str:
        file_name = import_file_path.stem
        return str(file_dir_path / file_name / "hybrid_auto" / f"{file_name}.md")
