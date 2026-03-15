Code-RL: Code Generation via Reinforcement Learning
====================================================

A framework for training code generation models using execution-based rewards.
Integrates with VERL (Volcengine Efficient RL) for efficient training.

QUICK START
-----------

1. Install dependencies:
   pip install -r requirements.txt
   pip install git+https://github.com/volcengine/verl.git

2. Prepare dataset:
   python data/prepare_dataset.py --sources rlvr --max_per_source 2000

3. Try example usage:
   python example_usage.py

4. Launch interactive demo:
   cd compare_generate
   python app.py

5. Run training:
   chmod +x run_training.sh
   ./run_training.sh

PROJECT STRUCTURE
-----------------

reward/              - Code execution and reward computation
verl_integration/    - VERL integration (data loaders, reward managers)
compare_generate/    - Gradio UI for model comparison
data/                - Dataset preparation and processing
eval/                - Evaluation and benchmarking tools
scripts/             - Utility scripts

KEY FILES
---------

setup.py                    - Package configuration
requirements.txt            - Python dependencies
example_usage.py            - Examples of core components
run_training.sh             - Training script with VERL
reward/executor.py          - Sandboxed code execution
reward/sandbox.py           - Sandbox implementations
verl_integration/reward_manager/micro_grpo_reward.py - Main reward manager
compare_generate/app.py     - Interactive model comparison UI
compare_generate/generate.py - Code generator
data/prepare_dataset.py     - Dataset preparation
eval/evaluate.py            - Model evaluation

KEY FUNCTIONS
-------------

* get_reward_manager() - Get singleton reward manager instance
* compute_score() - VERL single-sample reward interface
* compute_score_batch() - VERL batch reward interface (parallel)
* CodeExecutor.execute() - Execute code with tests
* CodeRewardManager.compute_reward() - Compute reward for code

CONFIGURATION
-------------

Sandbox: firejail (default) or subprocess
Timeout: 5.0 seconds
Max workers: CPU cores - 1

Reward range: [-0.5, 1.0]
  -0.5 = all tests failed or syntax error
   0.0 = no tests, valid syntax
   1.0 = all tests passed

TRAINING
--------

Uses VERL framework with GRPO algorithm. Key settings in run_training.sh:

- Model: Qwen2.5-Coder-1.5B-Instruct
- LoRA rank: 16
- Batch size: 16
- Response samples per prompt: 8
- Temperature: 0.7
- Reward function: micro_grpo_reward.compute_score

EVALUATION
----------

Evaluate on benchmarks:
  python eval/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --benchmark mbpp

NOTES
-----

- Firejail recommended for secure sandboxing
- Dataset verification recommended before training
- Weights & Biases logging enabled by default

SEE ALSO
--------

README.md - Detailed documentation with examples
compare_example.md - Example model comparisons