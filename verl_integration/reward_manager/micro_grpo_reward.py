"""
Micro-GRPO Reward Manager for VERL
使用 Sandbox 批量并行执行代码
"""
import sys
import os
from pathlib import Path
import re
import random
import json
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from reward.executor import CodeExecutor
from reward.metrics import calculate_pass_rate


def _execute_single_pair(args):
    """
    在独立进程中执行单个代码-测试对
    这个函数必须在模块级别以支持多进程
    """
    code, test, sandbox_type, timeout = args
    try:
        executor = CodeExecutor(sandbox_type=sandbox_type, timeout=timeout)
        result = executor.execute(code, test)
        return result
    except Exception as e:
        return {'status': 'error', 'message': f'Executor error: {str(e)}'}


class CodeRewardManager:
    """代码执行 Reward Manager - 支持并行批量执行"""

    def __init__(
        self, 
        sandbox_type: str = "firejail", 
        timeout: float = 5.0,
        max_workers: int = None
    ):
        """
        Args:
            sandbox_type: 'firejail' 或 'subprocess'
            timeout: 代码执行超时时间
            max_workers: 并行执行的最大工作进程数，None表示使用CPU核心数
        """
        self.sandbox_type = sandbox_type
        self.timeout = timeout
        self.executor = CodeExecutor(sandbox_type=sandbox_type, timeout=timeout)
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)

    def extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    def compute_reward(self, code: str, tests: list) -> dict:
        """计算单个代码的奖励（串行版本）"""
        if not code or not tests:
            return {'reward': 0.0, 'passed': 0, 'total': len(tests) if tests else 0}
        
        results = []
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                result = self.executor.execute(code, test)
                results.append(result)
                if result['status'] == 'success':
                    passed += 1
            except Exception:
                results.append({'status': 'error'})
        
        pass_rate = calculate_pass_rate(results)
        return {'reward': pass_rate, 'passed': passed, 'total': total}
    
    def compute_reward_batch_parallel(
        self, 
        codes: List[str], 
        tests_list: List[list]
    ) -> List[dict]:
        """
        并行批量计算奖励
        
        Args:
            codes: 代码列表
            tests_list: 测试列表的列表
        
        Returns:
            结果列表
        """
        if len(codes) != len(tests_list):
            raise ValueError("codes and tests_list must have the same length")
        
        # 准备所有的 (code, test) 对
        all_pairs = []
        pair_to_sample = []  # 记录每个pair属于哪个样本
        
        for idx, (code, tests) in enumerate(zip(codes, tests_list)):
            if not code or not tests:
                continue
            for test_idx, test in enumerate(tests):
                all_pairs.append((code, test, self.sandbox_type, self.timeout))
                pair_to_sample.append((idx, test_idx))
        
        # 并行执行所有pair
        if not all_pairs:
            batch_results = []
        else:
            # 如果pair数量少，使用串行执行
            if len(all_pairs) < 4:
                batch_results = [_execute_single_pair(args) for args in all_pairs]
            else:
                # 使用进程池并行执行
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    batch_results = list(executor.map(_execute_single_pair, all_pairs))
        
        # 组装结果
        results = []
        for idx, tests in enumerate(tests_list):
            code = codes[idx]
            
            if not code or not tests:
                results.append({
                    'reward': 0.0, 
                    'passed': 0, 
                    'total': len(tests) if tests else 0
                })
                continue
            
            # 找到属于这个样本的所有结果
            sample_results = []
            passed = 0
            
            for pair_idx, (sample_idx, test_idx) in enumerate(pair_to_sample):
                if sample_idx == idx:
                    result = batch_results[pair_idx]
                    sample_results.append(result)
                    if result['status'] == 'success':
                        passed += 1
            
            total = len(tests)
            pass_rate = calculate_pass_rate(sample_results)
            
            results.append({
                'reward': pass_rate,
                'passed': passed,
                'total': total
            })
        
        return results
    
    def compute_reward_batch(
        self, 
        codes: List[str], 
        tests_list: List[list],
        use_parallel: bool = True
    ) -> List[dict]:
        """
        批量计算奖励（智能选择串行或并行）
        
        Args:
            codes: 代码列表
            tests_list: 测试列表的列表
            use_parallel: 是否使用并行执行
        
        Returns:
            结果列表
        """
        # 计算总的测试数量
        total_tests = sum(len(tests) for tests in tests_list if tests)
        
        # 如果测试数量较少，使用串行执行
        if not use_parallel or total_tests < 10:
            if len(codes) != len(tests_list):
                raise ValueError("codes and tests_list must have the same length")
            
            results = []
            for code, tests in zip(codes, tests_list):
                result = self.compute_reward(code, tests)
                results.append(result)
            return results
        else:
            # 使用并行执行
            return self.compute_reward_batch_parallel(codes, tests_list)


# 全局单例
_reward_manager = None


def get_reward_manager():
    """获取全局 reward manager 实例"""
    global _reward_manager
    if _reward_manager is None:
        # 优先使用 firejail，如果不可用则使用 subprocess
        try:
            _reward_manager = CodeRewardManager(sandbox_type="firejail")
            print("[REWARD] Using firejail sandbox")
        except Exception as e:
            print(f"[REWARD] Firejail not available ({e}), using subprocess sandbox")
            _reward_manager = CodeRewardManager(sandbox_type="subprocess")
    return _reward_manager


def _parse_tests(ground_truth):
    """解析测试用例"""
    if ground_truth is None:
        return []
    
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
            if isinstance(parsed, dict):
                for k in ('test', 'test_cases', 'tests', 'assertion', 'check'):
                    if k in parsed:
                        v = parsed[k]
                        return [str(x) for x in v] if isinstance(v, list) else [str(v)]
                return [str(parsed)]
            return [ground_truth]
        except (json.JSONDecodeError, TypeError):
            return [ground_truth]
    
    if isinstance(ground_truth, list):
        return [str(t) for t in ground_truth]
    
    if isinstance(ground_truth, dict):
        for k in ('test', 'test_cases', 'tests', 'assertion', 'check'):
            if k in ground_truth:
                v = ground_truth[k]
                return [str(x) for x in v] if isinstance(v, list) else [str(v)]
        return [str(ground_truth)]
    
    return [str(ground_truth)]


def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, **kwargs):
    """
    VERL 单样本接口
    
    必须返回包含 'reward' 键的字典！
    """
    # 处理空输入
    if not solution_str or not solution_str.strip():
        return {'reward': 0.0}

    manager = get_reward_manager()
    code = manager.extract_code(solution_str)

    if not code.strip():
        return {'reward': 0.0}

    # 检查语法
    try:
        compile(code, '<string>', 'exec')
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False

    tests = _parse_tests(ground_truth)

    if not tests:
        # 没有测试用例时的奖励
        reward = 0.1 if syntax_valid else -0.5
        return {
            'reward': reward,
            'syntax_valid': syntax_valid
        }

    # 使用 sandbox 执行
    result = manager.compute_reward(code, tests)

    passed = result['passed']
    total = result['total']
    pass_rate = passed / total if total > 0 else 0.0
    
    # 计算最终奖励：范围 [-0.5, 1.0]
    reward = pass_rate * 1.5 - 0.5
    acc = 1.0 if passed == total and total > 0 else 0.0

    # 偶尔打印日志
    if random.randint(1, 20) == 1:
        print(f"[REWARD] reward={reward:.2f} pass={passed}/{total} syntax={syntax_valid}")

    # 返回格式：必须包含 'reward' 键！
    return {
        'score': reward,           # ← 这是最重要的！
        'pass_rate': pass_rate,
        'passed': passed,
        'total': total,
        'acc': acc,
        'syntax_valid': syntax_valid
    }


def compute_score_batch(batch_data, use_parallel: bool = True):
    """
    VERL 批量接口
    
    Args:
        batch_data: 可以是多种格式:
            - List[Dict]: [{'solution_str': ..., 'ground_truth': ...}, ...]
            - Dict: 其他格式
    
    Returns:
        List[Dict]: 每个元素必须包含 'reward' 键
    """
    # 调试：查看输入格式
    if random.randint(1, 50) == 1:
        print(f"[DEBUG] compute_score_batch input type: {type(batch_data)}")
        if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
            print(f"[DEBUG] First item type: {type(batch_data[0])}")
            if isinstance(batch_data[0], dict):
                print(f"[DEBUG] First item keys: {batch_data[0].keys()}")
    
    # 处理不同的输入格式
    if isinstance(batch_data, dict):
        # 如果是字典，尝试提取列表
        if 'items' in batch_data:
            items = batch_data['items']
        elif 'data' in batch_data:
            items = batch_data['data']
        else:
            # 可能整个字典就是一个样本
            items = [batch_data]
    elif isinstance(batch_data, (list, tuple)):
        items = batch_data
    else:
        print(f"[ERROR] Unknown batch_data type: {type(batch_data)}")
        return []
    
    if not items:
        return []
    
    manager = get_reward_manager()
    
    # 提取所有代码和测试
    codes = []
    all_tests = []
    metadata = []
    
    for item in items:
        # 提取字段（兼容多种命名）
        if isinstance(item, dict):
            solution_str = (
                item.get('solution_str', '') or 
                item.get('response', '') or 
                item.get('output', '') or 
                item.get('completion', '')
            )
            ground_truth = (
                item.get('ground_truth', []) or 
                item.get('test', []) or 
                item.get('tests', []) or 
                item.get('test_cases', [])
            )
        else:
            solution_str = str(item)
            ground_truth = []
        
        if not solution_str or not solution_str.strip():
            codes.append('')
            all_tests.append([])
            metadata.append({
                'has_code': False,
                'syntax_valid': False,
                'response_length': 0,
                'code_length': 0
            })
            continue
        
        code = manager.extract_code(solution_str)
        
        if not code.strip():
            codes.append('')
            all_tests.append([])
            metadata.append({
                'has_code': False,
                'syntax_valid': False,
                'response_length': len(solution_str),
                'code_length': 0
            })
            continue
        
        # 检查语法
        try:
            compile(code, '<string>', 'exec')
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        
        tests = _parse_tests(ground_truth)
        
        codes.append(code)
        all_tests.append(tests)
        metadata.append({
            'has_code': True,
            'syntax_valid': syntax_valid,
            'response_length': len(solution_str),
            'code_length': len(code)
        })
    
    # 批量执行
    exec_results = manager.compute_reward_batch(codes, all_tests, use_parallel=use_parallel)
    
    # 组装最终结果
    results = []
    for idx, (meta, exec_result) in enumerate(zip(metadata, exec_results)):
        if not meta['has_code']:
            results.append({'reward': 0.0})
            continue
        
        tests = all_tests[idx]
        
        if not tests:
            reward = 0.1 if meta['syntax_valid'] else -0.5
            results.append({
                'reward': reward,
                'syntax_valid': meta['syntax_valid']
            })
            continue
        
        passed = exec_result['passed']
        total = exec_result['total']
        pass_rate = passed / total if total > 0 else 0.0
        reward = pass_rate * 1.5 - 0.5
        acc = 1.0 if passed == total and total > 0 else 0.0
        
        results.append({
            'score': reward,           # ← 最重要的字段
            'pass_rate': pass_rate,
            'passed': passed,
            'total': total,
            'acc': acc,
            'syntax_valid': meta['syntax_valid']
        })
    
    # 打印统计
    if results and random.randint(1, 10) == 1:
        avg_reward = sum(r['reward'] for r in results) / len(results)
        print(f"[REWARD BATCH] Processed {len(results)} samples, avg reward: {avg_reward:.3f}")
    
    return results


# 别名（以防 VERL 使用不同的函数名）
compute_reward = compute_score
compute_rewards = compute_score_batch
batch_compute_score = compute_score_batch