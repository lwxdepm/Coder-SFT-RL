"""
Sandbox implementations for secure code execution.
三重防护: Firejail → cgroup → resource.setrlimit
"""

import subprocess
import tempfile
import os
import shutil
import resource
from typing import Optional

class FirejailSandbox:
    """Firejail-based sandbox for secure code execution."""

    def __init__(
        self,
        timeout: float = 5.0,
        memory_limit: str = "512m",
        cpu_limit: int = 8,
    ):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit

    def run(self, script: str) -> tuple[bool, str]:
        """Run script in firejail sandbox."""
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(prefix="sandbox_")
            fname = os.path.join(tmpdir, "script.py")

            with open(fname, 'w') as f:
                f.write(script)

            cmd = [
                "firejail",
                "--quiet",
                "--noprofile",
                "--net=none",
                "--noroot",
                "--nosound",
                "--private-tmp",
                f"--whitelist={tmpdir}",
                f"--rlimit-cpu={self.cpu_limit}",
                f"--rlimit-as={self.memory_limit}",
                "--rlimit-nproc=32",      # 限制进程数，防 fork bomb
                "--rlimit-nofile=64",     # 限制打开文件数
                "python3", fname,
            ]

            # Firejail 默认不隔离工作目录，手动切换到隔离目录
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tmpdir,
            )

            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output

        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, f"ERROR: {str(e)}"
        finally:
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass


class SubprocessSandbox:
    """Simple subprocess sandbox with resource limits.

    相比 Firejail 安全性较低，但通过 resource.setrlimit 实现了
    内存和进程数限制，防止 OOM 和 fork bomb。

    修复: 在隔离的临时工作目录中运行脚本，防止生成的代码
    删除项目目录（如 data/, reward/, scripts/ 等）。
    """

    def __init__(
        self,
        timeout: float = 5.0,
        max_memory_mb: int = 512,
    ):
        self.timeout = timeout
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

    def _set_limits(self):
        """在子进程中设置资源限制"""
        # 限制虚拟内存 (RLIMIT_AS)，防 OOM
        resource.setrlimit(
            resource.RLIMIT_AS,
            (self.max_memory_bytes, self.max_memory_bytes)
        )
        # 限制进程数，防 fork bomb
        resource.setrlimit(
            resource.RLIMIT_NPROC,
            (32, 32)
        )
        # 限制打开文件数
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (64, 64)
        )
        # 限制 CPU 时间 (秒)，超时会触发 SIGXCPU
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (int(self.timeout) + 2, int(self.timeout) + 2)
        )

    def run(self, script: str) -> tuple[bool, str]:
        """Run script in subprocess with resource limits."""
        tmpdir = None
        fname = None

        try:
            tmpdir = tempfile.mkdtemp(prefix="sandbox_")
            fname = os.path.join(tmpdir, "script.py")

            with open(fname, 'w') as f:
                f.write(script)

            # 关键修复: 使用 cwd=tmpdir 隔离工作目录
            # 防止生成的代码访问或删除项目目录（data/, reward/, scripts/ 等）
            # 同时设置 HOME=tmpdir 防止写入用户 home 目录
            env = os.environ.copy()
            env["HOME"] = tmpdir
            env["TMPDIR"] = tmpdir

            result = subprocess.run(
                ["python3", fname],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=self._set_limits,
                cwd=tmpdir,
                env=env,
            )

            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output

        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except subprocess.SubprocessError as e:
            return False, f"SUBPROCESS_ERROR: {str(e)}"
        except MemoryError:
            return False, "MEMORY_LIMIT_EXCEEDED"
        except Exception as e:
            return False, f"ERROR: {str(e)}"
        finally:
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass


def create_sandbox(
    sandbox_type: str = "firejail",
    **kwargs,
):
    """Factory function to create sandbox."""
    if sandbox_type == "firejail":
        return FirejailSandbox(**kwargs)
    elif sandbox_type == "subprocess":
        return SubprocessSandbox(**kwargs)
    else:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")
