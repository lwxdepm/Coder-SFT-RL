"""
Sandbox implementations for executing untrusted code.

设计目标：
1. 默认优先使用 Firejail；
2. subprocess 仅作为受限降级方案；
3. 对明显危险的文件/进程/网络相关代码做预检查；
4. 统一返回结构化结果，避免仅靠 returncode 判断成功。
"""

from __future__ import annotations

import ast
import os
import resource
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional


# 对于算法题 / 代码生成评测，这些模块通常不应该出现。
# 这里不是“绝对安全边界”，但能挡住大量误删、开子进程、网络访问等高风险行为。
BANNED_IMPORT_ROOTS = {
    "os",
    "shutil",
    "subprocess",
    "socket",
    "pathlib",
    "tempfile",
    "glob",
    "resource",
    "signal",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
}

BANNED_CALL_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
}

BANNED_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "spawnl"),
    ("os", "spawnlp"),
    ("os", "spawnv"),
    ("os", "spawnvp"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "removedirs"),
    ("os", "rename"),
    ("os", "replace"),
    ("os", "chdir"),
    ("os", "walk"),
    ("shutil", "rmtree"),
    ("shutil", "move"),
    ("shutil", "copytree"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("socket", "socket"),
}


@dataclass
class SandboxResult:
    ok: bool
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False
    blocked: bool = False
    error_type: Optional[str] = None

    @property
    def output(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


class SecurityVisitor(ast.NodeVisitor):
    """轻量 AST 预检查。不是完整沙箱，但能过滤明显危险代码。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            local_name = alias.asname or root
            self.aliases[local_name] = root
            if root in BANNED_IMPORT_ROOTS:
                self.errors.append(f"import '{root}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".", 1)[0]
        if module:
            for alias in node.names:
                local_name = alias.asname or alias.name
                self.aliases[local_name] = module
        if module in BANNED_IMPORT_ROOTS:
            self.errors.append(f"from '{module}' import ... is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in BANNED_CALL_NAMES:
                self.errors.append(f"call '{func.id}(...)' is not allowed")
            aliased_root = self.aliases.get(func.id)
            if aliased_root in BANNED_IMPORT_ROOTS:
                self.errors.append(f"use of aliased dangerous module '{aliased_root}' is not allowed")
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base_name = func.value.id
            root = self.aliases.get(base_name, base_name)
            pair = (root, func.attr)
            if pair in BANNED_ATTR_CALLS:
                self.errors.append(f"call '{root}.{func.attr}(...)' is not allowed")
        self.generic_visit(node)


def _validate_script(script: str) -> Optional[str]:
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # 语法错误交给真实执行阶段返回更完整信息
        return None

    visitor = SecurityVisitor()
    visitor.visit(tree)
    if visitor.errors:
        uniq = []
        seen = set()
        for err in visitor.errors:
            if err not in seen:
                seen.add(err)
                uniq.append(err)
        return "; ".join(uniq)
    return None


def _minimal_env(tmpdir: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": tmpdir,
        "TMPDIR": tmpdir,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "",
    }
    return env


class BaseSandbox:
    def __init__(self, timeout: float = 5.0, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

    def _write_script(self, script: str) -> tuple[str, str]:
        tmpdir = tempfile.mkdtemp(prefix="sandbox_")
        fname = os.path.join(tmpdir, "script.py")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(script)
        return tmpdir, fname

    def _cleanup(self, tmpdir: Optional[str]) -> None:
        if tmpdir and os.path.exists(tmpdir):
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    def _blocked_result(self, reason: str) -> SandboxResult:
        return SandboxResult(
            ok=False,
            returncode=None,
            stdout="",
            stderr=f"SECURITY_VIOLATION: {reason}",
            blocked=True,
            error_type="SECURITY_VIOLATION",
        )


class FirejailSandbox(BaseSandbox):
    """Firejail-based sandbox for untrusted code."""

    def __init__(
        self,
        timeout: float = 5.0,
        max_memory_mb: int = 512,
        cpu_limit_seconds: Optional[int] = None,
        protected_paths: Optional[Iterable[str]] = None,
    ):
        super().__init__(timeout=timeout, max_memory_mb=max_memory_mb)
        self.cpu_limit_seconds = cpu_limit_seconds or max(1, int(timeout) + 1)
        self.protected_paths = [os.path.abspath(p) for p in (protected_paths or []) if p]

    def run(self, script: str, timeout: Optional[float] = None) -> SandboxResult:
        reason = _validate_script(script)
        if reason:
            return self._blocked_result(reason)

        tmpdir = None
        try:
            tmpdir, fname = self._write_script(script)
            exec_timeout = self.timeout if timeout is None else timeout

            if shutil.which("firejail") is None:
                return SandboxResult(
                    ok=False,
                    returncode=None,
                    stdout="",
                    stderr="FIREJAIL_NOT_AVAILABLE: firejail command not found",
                    error_type="FIREJAIL_NOT_AVAILABLE",
                )

            cmd = [
                "firejail",
                "--quiet",
                "--noprofile",
                "--net=none",
                "--noroot",
                "--nonewprivs",
                "--nosound",
                "--seccomp",
                "--private-tmp",
                f"--private={tmpdir}",
                f"--rlimit-cpu={self.cpu_limit_seconds}",
                f"--rlimit-as={self.max_memory_bytes}",
                "--rlimit-nproc=32",
                "--rlimit-nofile=64",
            ]

            for path in self.protected_paths:
                cmd.append(f"--blacklist={path}")

            cmd += ["python3", fname]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=exec_timeout,
                cwd=tmpdir,
                env=_minimal_env(tmpdir),
            )
            return SandboxResult(
                ok=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                ok=False,
                returncode=None,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "TIMEOUT",
                timed_out=True,
                error_type="TIMEOUT",
            )
        except Exception as e:  # pragma: no cover
            return SandboxResult(
                ok=False,
                returncode=None,
                stdout="",
                stderr=f"ERROR: {type(e).__name__}: {e}",
                error_type=type(e).__name__,
            )
        finally:
            self._cleanup(tmpdir)


class SubprocessSandbox(BaseSandbox):
    """受限降级方案。

    注意：这不是严格的文件系统沙箱，只适合本地开发或 Firejail 不可用时兜底。
    为了降低风险，这里默认会先做 AST 预检查，拒绝明显危险的代码。
    """

    def _set_limits(self) -> None:
        resource.setrlimit(resource.RLIMIT_AS, (self.max_memory_bytes, self.max_memory_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CPU, (int(self.timeout) + 2, int(self.timeout) + 2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # 防止落地超大文件
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))

    def run(self, script: str, timeout: Optional[float] = None) -> SandboxResult:
        reason = _validate_script(script)
        if reason:
            return self._blocked_result(reason)

        tmpdir = None
        try:
            tmpdir, fname = self._write_script(script)
            exec_timeout = self.timeout if timeout is None else timeout

            result = subprocess.run(
                ["python3", fname],
                capture_output=True,
                text=True,
                timeout=exec_timeout,
                preexec_fn=self._set_limits,
                cwd=tmpdir,
                env=_minimal_env(tmpdir),
            )
            return SandboxResult(
                ok=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                ok=False,
                returncode=None,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "TIMEOUT",
                timed_out=True,
                error_type="TIMEOUT",
            )
        except subprocess.SubprocessError as e:
            return SandboxResult(
                ok=False,
                returncode=None,
                stdout="",
                stderr=f"SUBPROCESS_ERROR: {e}",
                error_type="SUBPROCESS_ERROR",
            )
        except MemoryError:
            return SandboxResult(
                ok=False,
                returncode=None,
                stdout="",
                stderr="MEMORY_LIMIT_EXCEEDED",
                error_type="MEMORY_LIMIT_EXCEEDED",
            )
        except Exception as e:  # pragma: no cover
            return SandboxResult(
                ok=False,
                returncode=None,
                stdout="",
                stderr=f"ERROR: {type(e).__name__}: {e}",
                error_type=type(e).__name__,
            )
        finally:
            self._cleanup(tmpdir)


def create_sandbox(sandbox_type: str = "firejail", **kwargs):
    """Factory function to create sandbox."""
    if sandbox_type == "firejail":
        return FirejailSandbox(**kwargs)
    if sandbox_type == "subprocess":
        return SubprocessSandbox(**kwargs)
    raise ValueError(f"Unknown sandbox type: {sandbox_type}")
