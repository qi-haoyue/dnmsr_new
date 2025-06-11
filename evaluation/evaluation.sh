#!/bin/bash
# 直接运行retrieval_evaluator.py而不是使用模块路径

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 设置环境变量，添加项目根目录到PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/../../..:$PYTHONPATH"

# 配置参数
EMBEDDINGS_DIR="/home/qhy/MML/dnmsr/dnmsr_new/evaluation/embeddings"
RESULTS_DIR="/home/qhy/MML/dnmsr/dnmsr_new/evaluation/results"

# 创建输出目录
mkdir -p "$RESULTS_DIR"

# 运行脚本
echo "开始评估检索性能..."
python "$SCRIPT_DIR/retrieval_evaluator.py" \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --output_dir "$RESULTS_DIR"

echo "检索评估完成。结果保存在: $RESULTS_DIR" 