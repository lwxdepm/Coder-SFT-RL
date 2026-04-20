"""
Production-Grade GRPO Reward Manager for VERL
包含完整的防崩溃机制、严格单调递增的奖励塑形，以及防御静默满分(Silent Pass)的解析器。
"""
import sys
import os
from pathlib import Path
import re
import random
import json
import ast
import time
import traceback
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np

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


def _create_default_result():
    """
    创建默认的结果字典，确保所有键都存在
    VERL 会收集所有出现过的键，所以必须保证每个样本都有相同的结构
    """
    return {
        # VERL 框架必需
        'reward': 0.0,

        # 正确性指标
        'score': 0.0,
        'pass_rate': 0.0,
        'passed': 0.0,
        'total': 0.0,
        'acc': 0.0,

        # 代码格式与验证阶段
        'has_code': 0.0,
        'syntax_valid': 0.0,
        'compile_valid': 0.0,
        'format_error': 0.0,
        'syntax_error': 0.0,
        'compile_error': 0.0,

        # 运行时问题
        'timeout_count': 0.0,
        'runtime_error_count': 0.0,

        # 代码特征（供 W&B 监控退化和分布）
        'code_length': 0.0,
        'is_markdown': 0.0,

        # 性能指标
        'execution_time': 0.0,
    }


class RewardShaper:
    """
    Production-Grade Reward Shaping
    严格单调递增的奖励设计，彻底杜绝语法错误坍缩
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """阶梯式奖励配置 - 保证每进一步都有收益，绝不倒挂"""
        return {
            # === 阶段 1：底线惩罚（必须遵守格式且有代码）===
            'format_penalty': -1.5,        # 没有代码块或为空

            # === 阶段 2：语法与编译（严格递增）===
            'syntax_error_penalty': -1.0,  # 语法错误，比格式错误好
            'compile_error_penalty': -0.5, # 编译错误，比语法错误好

            # === 阶段 3：执行阶段基准分（只要能跑，就比编译错误好）===
            'execution_base_reward': 0.0,  # 运行且无通过时的底薪
            
            # === 运行时扣分（扣完也必须大于 compile_error_penalty）===
            'runtime_error_penalty': -0.2, # 运行时崩溃扣分 (最终 0.0 - 0.2 = -0.2)
            'timeout_penalty': -0.3,       # 超时扣分 (最终 0.0 - 0.3 = -0.3)
            
            # === 阶段 4：正确性奖励（大头）===
            'pass_rate_weight': 2.0,       # 按照通过率给分 (最高 +2.0)
            'full_pass_bonus': 1.0,        # 全对额外奖励 (+1.0)
        }
    
    def shape_reward(self, code: str, syntax_valid: bool, compile_valid: bool,
                 execution_results: List[Dict[str, Any]], tests_count: int,
                 code_length: int = 0, is_markdown: bool = False) -> Dict[str, Any]:
        cfg = self.config
        details = {}

        base_result = {
            'reward': 0.0,
            'has_code': 0.0,
            'syntax_valid': 0.0,
            'compile_valid': 0.0,
            'pass_rate': 0.0,
            'passed': 0.0,
            'total': float(tests_count),
            'acc': 0.0,
            'timeout_count': 0.0,
            'runtime_error_count': 0.0,
            'code_length': float(code_length),
            'is_markdown': 1.0 if is_markdown else 0.0,
        }

        # 1. Format / Empty Check
        if not code or not code.strip():
            details['reason'] = 'format_or_empty_error'
            base_result['reward'] = cfg['format_penalty']
            base_result['has_code'] = 0.0
            base_result['format_error'] = 1.0
            base_result['syntax_error'] = 0.0
            base_result['compile_error'] = 0.0
            return base_result

        base_result['has_code'] = 1.0
        base_result['format_error'] = 0.0

        # 2. Syntax Error
        if not syntax_valid:
            details['reason'] = 'syntax_error'
            base_result['reward'] = cfg['syntax_error_penalty']
            base_result['syntax_valid'] = 0.0
            base_result['syntax_error'] = 1.0
            base_result['compile_error'] = 0.0
            return base_result

        base_result['syntax_valid'] = 1.0
        base_result['syntax_error'] = 0.0

        # 3. Compile Error
        if not compile_valid:
            details['reason'] = 'compile_error'
            base_result['reward'] = cfg['compile_error_penalty']
            base_result['compile_valid'] = 0.0
            base_result['compile_error'] = 1.0
            return base_result

        base_result['compile_valid'] = 1.0
        base_result['compile_error'] = 0.0

        # 4. Execution Check
        if not execution_results or tests_count == 0:
            details['reason'] = 'no_tests'
            base_result['reward'] = cfg['execution_base_reward'] + 0.2
            base_result['total'] = 0.0
            return base_result

        # 统计执行结果
        passed = 0
        runtime_errors = 0
        timeouts = 0

        for result in execution_results:
            status = result.get('status', 'error')
            if status == 'success': passed += 1
            elif status == 'timeout': timeouts += 1
            elif status in ('error', 'runtime_error'): runtime_errors += 1

        pass_rate = passed / tests_count if tests_count > 0 else 0.0

        # --- 计算最终奖励 ---
        reward = cfg['execution_base_reward']

        # 惩罚项（保证扣除后依然严格大于 compile_error_penalty）
        if timeouts > 0:
            reward += cfg['timeout_penalty']
            details['reason'] = 'timeout'
        elif runtime_errors > 0 and passed == 0:
            reward += cfg['runtime_error_penalty']
            details['reason'] = 'runtime_error'
        elif passed == 0:
            details['reason'] = 'zero_pass'

        # 奖励项
        if passed > 0:
            reward += pass_rate * cfg['pass_rate_weight']
            details['reason'] = 'partial_pass'

        if passed == tests_count and tests_count > 0:
            reward += cfg['full_pass_bonus']
            details['reason'] = 'full_pass'

        base_result.update({
            'reward': reward,
            'pass_rate': pass_rate,
            'passed': float(passed),
            'total': float(tests_count),
            'acc': 1.0 if passed == tests_count and tests_count > 0 else 0.0,
            'timeout_count': float(timeouts),
            'runtime_error_count': float(runtime_errors),
        })

        return base_result


class CodeValidator:
    """代码验证器 - 检查语法和编译"""

    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, str]:
        if not code or not code.strip():
            return False, "Empty code"
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Parse error: {e}"

    @staticmethod
    def check_compile(code: str) -> Tuple[bool, str]:
        if not code or not code.strip():
            return False, "Empty code"
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Compile error (syntax): {e}"
        except Exception as e:
            return False, f"Compile error: {e}"

    @staticmethod
    def validate_code(code: str) -> Dict[str, Any]:
        result = {
            'has_code': bool(code and code.strip()),
            'syntax_valid': False,
            'compile_valid': False,
            'syntax_error': '',
            'compile_error': ''
        }
        if not result['has_code']:
            return result

        syntax_valid, syntax_error = CodeValidator.check_syntax(code)
        result['syntax_valid'] = syntax_valid
        result['syntax_error'] = syntax_error

        if not syntax_valid:
            return result

        compile_valid, compile_error = CodeValidator.check_compile(code)
        result['compile_valid'] = compile_valid
        result['compile_error'] = compile_error

        return result

    @staticmethod
    def extract_reason(validation: Dict[str, Any]) -> str:
        """从验证结果提取失败原因"""
        if not validation.get('has_code'):
            return 'empty_or_format'
        if not validation.get('syntax_valid'):
            return 'syntax'
        if not validation.get('compile_valid'):
            return 'compile'
        return 'none'


class CodeRewardManager:
    """Production-Grade 代码执行 Reward Manager"""

    def __init__(
        self, 
        sandbox_type: str = "firejail", 
        timeout: float = 5.0,
        max_workers: int = None,
        reward_config: Dict[str, Any] = None,
        verbose: bool = False
    ):
        self.sandbox_type = sandbox_type
        self.timeout = timeout
        self.executor = CodeExecutor(sandbox_type=sandbox_type, timeout=timeout)
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.validator = CodeValidator()
        self.shaper = RewardShaper(config=reward_config)
        self.verbose = verbose
        
        self.stats = {
            'total_samples': 0,
            'empty_samples': 0,
            'syntax_errors': 0,
            'compile_errors': 0,
            'runtime_errors': 0,
            'full_pass': 0,
        }

    def extract_code(self, response: str) -> str:
        """从响应中提取代码（带有纯代码宽容机制）"""
        if not response:
            return ""
        
        # 1. 首选：匹配 markdown 代码块
        pattern = r'```(?:python)?\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            # 提取最后一块，因为模型通常先思考 (CoT)，最后输出代码
            return matches[-1].strip()
        
        # 2. 智能宽容机制 (Fallback)：
        # 如果模型没有使用 markdown，但它输出的【整段文本】就是一段完全合法的 Python 代码
        # 那么我们也放过它，将其视为代码提取出来。
        try:
            ast.parse(response)
            return response.strip() # 整段都是合法代码，原谅它！
        except SyntaxError:
            pass # 包含了解析不出的闲聊文本，且没有代码块，坚决提取失败
            
        # 3. 既没有代码块，整段文本也不是纯代码 -> 触发 -1.5 格式惩罚
        return ""

    def compute_reward(self, code: str, tests: list) -> dict:
        self.stats['total_samples'] += 1
        result = _create_default_result()

        start_time = time.time()
        validation = self.validator.validate_code(code)

        code_length = len(code) if code else 0
        result['code_length'] = float(code_length)

        if not validation['has_code']:
            self.stats['empty_samples'] += 1
            result['reward'] = self.shaper.config['format_penalty']
            result['score'] = self.shaper.config['format_penalty']
            result['has_code'] = 0.0
            result['format_error'] = 1.0
            result['execution_time'] = time.time() - start_time
            return result

        if not validation['syntax_valid']:
            self.stats['syntax_errors'] += 1
            shaped = self.shaper.shape_reward(
                code=code, syntax_valid=False, compile_valid=False,
                execution_results=[], tests_count=len(tests) if tests else 0,
                code_length=code_length
            )
            result.update(shaped)
            result['score'] = shaped['reward']
            result['syntax_error'] = 1.0
            result['execution_time'] = time.time() - start_time
            return result

        if not validation['compile_valid']:
            self.stats['compile_errors'] += 1
            shaped = self.shaper.shape_reward(
                code=code, syntax_valid=True, compile_valid=False,
                execution_results=[], tests_count=len(tests) if tests else 0,
                code_length=code_length
            )
            result.update(shaped)
            result['score'] = shaped['reward']
            result['compile_error'] = 1.0
            result['execution_time'] = time.time() - start_time
            return result

        if not tests:
            shaped = self.shaper.shape_reward(
                code=code, syntax_valid=True, compile_valid=True,
                execution_results=[], tests_count=0,
                code_length=code_length
            )
            result.update(shaped)
            result['score'] = shaped['reward']
            result['execution_time'] = time.time() - start_time
            return result

        execution_results = []
        for test in tests:
            try:
                exec_result = self.executor.execute(code, test)
                execution_results.append(exec_result)
            except Exception as e:
                execution_results.append({'status': 'error', 'message': str(e)})

        shaped = self.shaper.shape_reward(
            code=code, syntax_valid=True, compile_valid=True,
            execution_results=execution_results, tests_count=len(tests),
            code_length=code_length
        )

        if shaped.get('acc', 0.0) == 1.0:
            self.stats['full_pass'] += 1

        result.update(shaped)
        result['score'] = shaped['reward']
        result['execution_time'] = time.time() - start_time

        if self.verbose and random.random() < 0.1:
            print(f"[REWARD] score={result['score']:.2f} | "
                f"pass={int(shaped.get('passed', 0))}/{int(shaped.get('total', 0))} | "
                f"code_len={code_length}")

        if self.verbose and (self.stats['total_samples'] <= 20 or random.random() < 0.02):
            print(f"\n{'='*80}")
            print(f"[REWARD DETAIL] Sample #{self.stats['total_samples']}")
            print(f"Score: {result['score']:.2f} | Pass: {int(shaped.get('passed', 0))}/{int(shaped.get('total', 0))}")
            print(f"\n[Generated Code ({len(code)} chars)]")
            print(code[:400])
            if len(code) > 400: print("... (truncated)")
            if tests:
                print(f"\n[First Test]")
                print(tests[0][:250])
            if execution_results:
                failed = [(i, r) for i, r in enumerate(execution_results) if r.get('status') != 'success']
                if failed:
                    print(f"\n[Failed Tests: {len(failed)}/{len(execution_results)}]")
                    for i, r in failed[:2]:
                        print(f"  Test {i+1}: {r.get('status')} - {r.get('message', 'N/A')[:150]}")
            print(f"{'='*80}\n")

        return result
    
    def compute_reward_batch_parallel(self, codes: List[str], tests_list: List[list]) -> List[dict]:
        if len(codes) != len(tests_list):
            raise ValueError("codes and tests_list must have the same length")

        all_pairs = []
        pair_to_sample = []
        sample_validations = []

        for idx, (code, tests) in enumerate(zip(codes, tests_list)):
            validation = self.validator.validate_code(code)
            sample_validations.append(validation)
            if validation['has_code'] and validation['syntax_valid'] and validation['compile_valid']:
                if tests:
                    for test_idx, test in enumerate(tests):
                        all_pairs.append((code, test, self.sandbox_type, self.timeout))
                        pair_to_sample.append((idx, test_idx))

        if not all_pairs:
            batch_results = []
        else:
            if len(all_pairs) < 4:
                batch_results = [_execute_single_pair(args) for args in all_pairs]
            else:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    batch_results = list(executor.map(_execute_single_pair, all_pairs))

        results = []
        pair_idx = 0

        for sample_idx, (code, tests) in enumerate(zip(codes, tests_list)):
            validation = sample_validations[sample_idx]
            result = _create_default_result()
            result['code_length'] = float(len(code)) if code else 0.0

            if not validation['has_code']:
                result['reward'] = self.shaper.config['format_penalty']
                result['score'] = self.shaper.config['format_penalty']
                result['has_code'] = 0.0
                result['format_error'] = 1.0
                results.append(result)
                continue

            if not validation['syntax_valid']:
                shaped = self.shaper.shape_reward(
                    code=code, syntax_valid=False, compile_valid=False,
                    execution_results=[], tests_count=len(tests) if tests else 0,
                    code_length=result['code_length']
                )
                result.update(shaped)
                result['score'] = shaped['reward']
                result['syntax_error'] = 1.0
                results.append(result)
                continue

            if not validation['compile_valid']:
                shaped = self.shaper.shape_reward(
                    code=code, syntax_valid=True, compile_valid=False,
                    execution_results=[], tests_count=len(tests) if tests else 0,
                    code_length=result['code_length']
                )
                result.update(shaped)
                result['score'] = shaped['reward']
                result['compile_error'] = 1.0
                results.append(result)
                continue

            execution_results = []
            if tests:
                for _ in range(len(tests)):
                    if pair_idx < len(batch_results):
                        execution_results.append(batch_results[pair_idx])
                        pair_idx += 1

            shaped = self.shaper.shape_reward(
                code=code, syntax_valid=True, compile_valid=True,
                execution_results=execution_results, tests_count=len(tests) if tests else 0,
                code_length=result['code_length']
            )
            result.update(shaped)
            result['score'] = shaped['reward']
            results.append(result)

        return results
    
    def compute_reward_batch(self, codes: List[str], tests_list: List[list], use_parallel: bool = True) -> List[dict]:
        total_tests = sum(len(tests) for tests in tests_list if tests)
        if not use_parallel or total_tests < 10:
            if len(codes) != len(tests_list):
                raise ValueError("codes and tests_list must have the same length")
            results = []
            for code, tests in zip(codes, tests_list):
                results.append(self.compute_reward(code, tests))
            return results
        else:
            return self.compute_reward_batch_parallel(codes, tests_list)
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.stats['total_samples']
        if total == 0:
            return self.stats
        return {
            **self.stats,
            'empty_rate': self.stats['empty_samples'] / total,
            'syntax_error_rate': self.stats['syntax_errors'] / total,
            'compile_error_rate': self.stats['compile_errors'] / total,
            'full_pass_rate': self.stats['full_pass'] / total,
        }


# 全局单例
_reward_manager = None


def get_reward_manager():
    """获取全局 reward manager 实例"""
    global _reward_manager
    if _reward_manager is None:
        try:
            _reward_manager = CodeRewardManager(sandbox_type="firejail", verbose=True)
            print("[REWARD] Using firejail sandbox with strict monotonic reward shaping")
        except Exception as e:
            print(f"[REWARD] Firejail not available ({e}), using subprocess sandbox")
            _reward_manager = CodeRewardManager(sandbox_type="subprocess", verbose=True)
    return _reward_manager


def _parse_tests(ground_truth):
    """
    鲁棒的测试用例解析器
    彻底防范由于 Numpy 强制转换为 String (缺少逗号) 导致的静默全通(Silent Pass) Bug

    修复: 处理 pandas Series/DataFrame 等 parquet 读取后的类型
    """
    if ground_truth is None:
        return []

    # pandas Series / DataFrame: 转为 list
    type_name = type(ground_truth).__name__
    if type_name in ('Series',):
        ground_truth = ground_truth.tolist()
    elif type_name in ('DataFrame',):
        ground_truth = ground_truth.to_dict('records')

    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()

    # 处理 JSON string 或受损的 stringified lists
    if isinstance(ground_truth, str):
        ground_truth = ground_truth.strip()
        if not ground_truth:
            return []

        parsed = False

        # 1. 尝试标准 JSON 解析
        try:
            ground_truth = json.loads(ground_truth)
            parsed = True
        except Exception:
            pass

        # 2. 尝试 Python literal_eval
        if not parsed:
            try:
                ground_truth = ast.literal_eval(ground_truth)
                parsed = True
            except Exception:
                pass

        # 3. 核心防御：处理由于缺失逗号等原因导致解析失败的 Numpy Stringified Array
        if not parsed and ground_truth.startswith('[') and ground_truth.endswith(']'):
            try:
                tree = ast.parse(ground_truth)
                extracted = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        extracted.append(node.value)
                    elif type(node).__name__ == 'Str':
                        extracted.append(node.s)

                if extracted:
                    ground_truth = extracted
                    parsed = True
                else:
                    return []
            except Exception:
                return []

        # 4. 如果所有解析都失败且不是列表表现形式，那就默认它是纯原生 Python 代码片段
        if not parsed and isinstance(ground_truth, str):
            ground_truth = [ground_truth]

    # dict
    if isinstance(ground_truth, dict):
        for k in ("test", "tests", "test_cases", "assertion", "check"):
            if k in ground_truth:
                ground_truth = ground_truth[k]
                break

    # flatten
    tests = []
    def _flatten(x):
        if isinstance(x, list):
            for i in x:
                _flatten(i)
        elif isinstance(x, dict):
            tests.append(str(x))
        else:
            if x is not None:
                tests.append(str(x))

    _flatten(ground_truth)

    return [t.strip() for t in tests if t and t.strip()]


def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, **kwargs):
    """
    VERL 单样本接口（Production 版本）

    完整的防崩溃机制：
    1. Anti-format-error reward (强制 markdown)
    2. Strict Monotonic Syntax shaping
    3. Compile shaping
    4. Execution shaping
    5. Anti-Silent-Pass (AST 级解析器)
    6. 顶层 try/except 防护 Ray worker 崩溃
    """
    try:
        manager = get_reward_manager()

        code = manager.extract_code(solution_str) if solution_str else ""

        # 从 extra_info 中提取 tests
        tests = []
        if extra_info is not None:
            try:
                tests = _parse_tests(extra_info.get("tests", ""))
            except Exception:
                tests = []

        result = manager.compute_reward(code, tests)

        if random.randint(1, 100) == 1:
            stats = manager.get_stats()
            print(f"\n[REWARD STATS] "
                  f"total={stats['total_samples']} "
                  f"format_err={stats['empty_rate']:.2%} "
                  f"syntax_err={stats['syntax_error_rate']:.2%} "
                  f"compile_err={stats['compile_error_rate']:.2%} "
                  f"full_pass={stats['full_pass_rate']:.2%}")

        return result

    except Exception as e:
        # 关键修复: 任何异常都返回默认结果，防止 Ray worker 崩溃导致训练终止
        print(f"[REWARD ERROR] compute_score crashed: {e}")
        traceback.print_exc()
        default = _create_default_result()
        default['reward'] = -2.0  # 给予严重惩罚
        default['score'] = -2.0
        return default