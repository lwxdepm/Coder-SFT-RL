"""代码执行器 - 使用 Sandbox"""
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reward.sandbox import create_sandbox


class CodeExecutor:
    """使用 Sandbox 的代码执行器"""
    
    def __init__(self, sandbox_type: str = "firejail", timeout: float = 5.0):
        """
        Args:
            sandbox_type: 'firejail' 或 'subprocess'
            timeout: 执行超时时间
        """
        self.sandbox = create_sandbox(sandbox_type=sandbox_type, timeout=timeout)
    
    def execute(self, code: str, test: str, timeout: int = 5):
        """
        执行代码和测试
        
        Args:
            code: 要执行的代码
            test: 测试代码
            timeout: 超时时间（此参数保留兼容性，实际使用初始化时的timeout）
        
        Returns:
            {'status': 'success'|'error', 'message': str}
        """
        # 合并代码和测试
        script = f"{code}\n\n{test}"
        
        try:
            passed, output = self.sandbox.run(script)
            
            if passed:
                return {'status': 'success', 'message': 'All tests passed'}
            else:
                if "TIMEOUT" in output:
                    return {'status': 'error', 'message': 'Timeout'}
                elif "AssertionError" in output:
                    return {'status': 'error', 'message': f'Test failed: {output}'}
                else:
                    return {'status': 'error', 'message': output}
                    
        except Exception as e:
            return {'status': 'error', 'message': f'{type(e).__name__}: {str(e)}'}
    
    def execute_batch(self, code_test_pairs: list) -> list:
        """
        批量执行代码和测试
        
        Args:
            code_test_pairs: [(code1, test1), (code2, test2), ...]
        
        Returns:
            [result1, result2, ...] 每个result是 {'status': ..., 'message': ...}
        """
        results = []
        for code, test in code_test_pairs:
            result = self.execute(code, test)
            results.append(result)
        return results