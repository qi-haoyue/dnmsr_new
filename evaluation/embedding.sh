#!/bin/bash
# 直接运行embedding_builder.py而不是使用模块路径

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 设置环境变量，添加项目根目录到PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/../../..:$PYTHONPATH"

# 配置参数
DNMSR_MODEL_PATH="/home/qhy/MML/dnmsr/Visualized_m3.pth"
SAMPLES_FILE="/home/qhy/MML/dnmsr/dnmsr_new/data_preprocessing/samples/sampled_products.json"
CANDIDATES_FILE="/home/qhy/MML/dnmsr/dnmsr_new/data_preprocessing/samples/candidate_documents.json"
OUTPUT_DIR="/home/qhy/MML/dnmsr/dnmsr_new/evaluation/embeddings"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行脚本
echo "开始构建嵌入..."
python "$SCRIPT_DIR/embedding_builder.py" \
    --dnmsr_model_path "$DNMSR_MODEL_PATH" \
    --samples_file "$SAMPLES_FILE" \
    --candidates_file "$CANDIDATES_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "嵌入构建完成。结果保存在: $OUTPUT_DIR"