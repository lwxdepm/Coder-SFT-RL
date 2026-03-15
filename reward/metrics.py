"""Metrics for code evaluation"""

def calculate_pass_rate(results):
    """Calculate pass rate from test results"""
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get('status') == 'success')
    return passed / len(results)
