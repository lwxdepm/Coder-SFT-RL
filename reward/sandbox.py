"""
Sandbox implementations for secure code execution.
"""

import subprocess
import tempfile
import os
import shutil
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
            # 创建临时目录
            tmpdir = tempfile.mkdtemp(prefix="sandbox_")
            fname = os.path.join(tmpdir, "script.py")
            
            # 写入脚本
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
                "python3", fname,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output

        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, f"ERROR: {str(e)}"
        finally:
            # 清理临时目录
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass


class SubprocessSandbox:
    """Simple subprocess sandbox (less secure)."""

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def run(self, script: str) -> tuple[bool, str]:
        """Run script in subprocess."""
        tmpdir = None
        fname = None
        
        try:
            # 创建临时目录
            tmpdir = tempfile.mkdtemp(prefix="sandbox_")
            fname = os.path.join(tmpdir, "script.py")
            
            # 写入脚本
            with open(fname, 'w') as f:
                f.write(script)

            cmd = ["python3", fname]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            passed = result.returncode == 0
            output = result.stdout + result.stderr
            return passed, output

        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, f"ERROR: {str(e)}"
        finally:
            # 清理临时目录
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