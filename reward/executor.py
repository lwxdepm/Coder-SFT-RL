"""代码执行器 - 使用 Sandbox（修复版）"""

from __future__ import annotations

import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

# 添加项目根目录到 path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from reward.sandbox import create_sandbox


class CodeExecutor:
    """使用 Sandbox 的代码执行器。"""

    def __init__(
        self,
        sandbox_type: str = "firejail",
        timeout: float = 5.0,
        max_memory_mb: int = 512,
    ):
        """
        Args:
            sandbox_type: 'firejail' 或 'subprocess'
            timeout: 执行超时时间
            max_memory_mb: 内存限制（MB）
        """
        self.timeout = timeout
        self.sandbox = create_sandbox(
            sandbox_type=sandbox_type,
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            protected_paths=[str(project_root)],
        )

    @staticmethod
    def _build_script(code: str, test: str, sentinel: str) -> str:
        """构造最终执行脚本。

        关键修复：追加一个随机 sentinel。
        只有当脚本以 0 退出且确实打印了 sentinel，才算测试真正跑完。
        这样可以避免生成代码在测试前 `sys.exit(0)` / `os._exit(0)` 导致误判成功。
        """
        return textwrap.dedent(
            f"""
            # -*- coding: utf-8 -*-
            {code}

            {test}

            print({sentinel!r})
            """
        ).strip() + "\n"

    @staticmethod
    def _strip_sentinel(output: str, sentinel: str) -> str:
        return "\n".join(line for line in output.splitlines() if line.strip() != sentinel).strip()

    def execute(self, code: str, test: str, timeout: float | None = None) -> dict[str, Any]:
        """执行代码和测试。"""
        exec_timeout = self.timeout if timeout is None else timeout
        sentinel = f"__TESTS_COMPLETED__:{uuid.uuid4().hex}"
        script = self._build_script(code, test, sentinel)

        try:
            result = self.sandbox.run(script, timeout=exec_timeout)
            cleaned_output = self._strip_sentinel(result.output, sentinel)

            if result.blocked:
                return {
                    "status": "error",
                    "message": cleaned_output or "Security violation",
                    "error_type": result.error_type or "SECURITY_VIOLATION",
                }

            if result.timed_out:
                return {
                    "status": "error",
                    "message": "Timeout",
                    "error_type": "TIMEOUT",
                }

            # returncode=0 但没有 sentinel，说明代码提前退出，测试并未真正跑完
            if result.returncode == 0 and sentinel not in result.output:
                return {
                    "status": "error",
                    "message": "Program exited before tests completed",
                    "error_type": "INCOMPLETE_TEST_RUN",
                }

            if result.returncode == 0 and sentinel in result.output:
                return {
                    "status": "success",
                    "message": "All tests passed",
                    "error_type": None,
                }

            if "AssertionError" in result.output:
                return {
                    "status": "error",
                    "message": cleaned_output or "Test failed",
                    "error_type": "ASSERTION_ERROR",
                }
            if "SyntaxError" in result.output:
                return {
                    "status": "error",
                    "message": cleaned_output or "Syntax error",
                    "error_type": "SYNTAX_ERROR",
                }
            if "FIREJAIL_NOT_AVAILABLE" in result.output:
                return {
                    "status": "error",
                    "message": cleaned_output,
                    "error_type": "FIREJAIL_NOT_AVAILABLE",
                }
            return {
                "status": "error",
                "message": cleaned_output or "Execution failed",
                "error_type": result.error_type or "RUNTIME_ERROR",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"{type(e).__name__}: {str(e)}",
                "error_type": type(e).__name__,
            }

    def execute_batch(self, code_test_pairs: list[tuple[str, str]], timeout: float | None = None) -> list[dict[str, Any]]:
        """批量执行代码和测试。"""
        return [self.execute(code, test, timeout=timeout) for code, test in code_test_pairs]
