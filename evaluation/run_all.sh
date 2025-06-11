#!/bin/bash
# 用于运行DNMSR评估的完整流程脚本

# 获取当前脚本所在目录（绝对路径）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DNMSR_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

# 默认配置参数 - 使用相对路径便于在不同环境运行
MODEL_WEIGHT="$DNMSR_ROOT/Visualized_m3.pth"
DATA_ROOT="/home/qhy/MML/DNM_dataset/changan"
JSON_FILE="/home/qhy/MML/DNM_dataset/changan/ca_fixed.json"
IMAGE_DIR="/home/qhy/MML/DNM_dataset/exchange_market/pic/full"
SAMPLES_DIR="$DNMSR_ROOT/dnmsr_new/data_preprocessing/samples"
SAMPLES_FILE="$SAMPLES_DIR/sampled_products.json"
CANDIDATES_FILE="$SAMPLES_DIR/candidate_documents.json"
EMBEDDING_DIR="$SCRIPT_DIR/embeddings"
RESULTS_DIR="$SCRIPT_DIR/results"

# 设置环境变量 - 确保Python能找到模块
export PYTHONPATH="$DNMSR_ROOT:$PYTHONPATH"

# 确保必要的目录存在
mkdir -p "$SAMPLES_DIR"
mkdir -p "$EMBEDDING_DIR"
mkdir -p "$RESULTS_DIR"

# 颜色和样式
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RED="\033[0;31m"
NC="\033[0m" # 无颜色

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查文件存在函数
check_file_exists() {
    if [[ ! -f "$1" ]]; then
        print_error "文件不存在: $1"
        return 1
    fi
    return 0
}

# 检查目录存在函数
check_dir_exists() {
    if [[ ! -d "$1" ]]; then
        print_error "目录不存在: $1"
        return 1
    fi
    return 0
}

# 显示当前配置
show_config() {
    echo ""
    echo -e "${BOLD}当前配置:${NC}"
    echo "-------------------------"
    echo "模型权重文件: $MODEL_WEIGHT"
    echo "数据根目录: $DATA_ROOT"
    echo "JSON文件: $JSON_FILE"
    echo "图像目录: $IMAGE_DIR"
    echo "采样文件: $SAMPLES_FILE"
    echo "候选文档文件: $CANDIDATES_FILE"
    echo "嵌入向量目录: $EMBEDDING_DIR"
    echo "结果输出目录: $RESULTS_DIR"
    echo "-------------------------"
    echo ""
}

# 更新配置
update_config() {
    echo ""
    echo -e "${BOLD}更新配置参数${NC}"
    echo "-------------------------"
    read -p "模型权重文件 [$MODEL_WEIGHT]: " new_model
    MODEL_WEIGHT=${new_model:-$MODEL_WEIGHT}
    
    read -p "数据根目录 [$DATA_ROOT]: " new_data_root
    DATA_ROOT=${new_data_root:-$DATA_ROOT}
    
    read -p "JSON文件 [$JSON_FILE]: " new_json
    JSON_FILE=${new_json:-$JSON_FILE}
    
    read -p "图像目录 [$IMAGE_DIR]: " new_image_dir
    IMAGE_DIR=${new_image_dir:-$IMAGE_DIR}
    
    read -p "采样文件目录 [$SAMPLES_DIR]: " new_samples_dir
    SAMPLES_DIR=${new_samples_dir:-$SAMPLES_DIR}
    SAMPLES_FILE="$SAMPLES_DIR/sampled_products.json"
    CANDIDATES_FILE="$SAMPLES_DIR/candidate_documents.json"
    
    read -p "嵌入向量目录 [$EMBEDDING_DIR]: " new_embedding_dir
    EMBEDDING_DIR=${new_embedding_dir:-$EMBEDDING_DIR}
    
    read -p "结果输出目录 [$RESULTS_DIR]: " new_results_dir
    RESULTS_DIR=${new_results_dir:-$RESULTS_DIR}
    
    mkdir -p "$SAMPLES_DIR"
    mkdir -p "$EMBEDDING_DIR"
    mkdir -p "$RESULTS_DIR"
    
    print_success "配置已更新"
    show_config
}

# 生成采样和候选文档
generate_samples() {
    print_info "开始生成采样和候选文档..."
    
    # 检查必要文件是否存在
    if ! check_file_exists "$JSON_FILE"; then
        print_error "无法继续，JSON文件不存在"
        return 1
    fi
    
    if ! check_dir_exists "$IMAGE_DIR"; then
        print_warning "图像目录不存在，可能会导致图像无法加载"
    fi
    
    # 运行数据预处理脚本
    print_info "运行数据预处理脚本..."
    python "$DNMSR_ROOT/dnmsr_new/data_preprocessing/DarkWebProductLoader.py" \
           --json_file "$JSON_FILE" \
           --image_dir "$IMAGE_DIR" \
           --samples_output "$SAMPLES_FILE" \
           --candidates_output "$CANDIDATES_FILE" \
           --sample_size 100
    
    if [[ $? -ne 0 ]]; then
        print_error "采样生成失败"
        return 1
    fi
    
    if check_file_exists "$SAMPLES_FILE" && check_file_exists "$CANDIDATES_FILE"; then
        print_success "采样和候选文档生成成功"
        echo "采样文件: $SAMPLES_FILE"
        echo "候选文档文件: $CANDIDATES_FILE"
        return 0
    else
        print_error "采样文件或候选文档文件不存在"
        return 1
    fi
}

# 构建嵌入
build_embeddings() {
    print_info "开始构建嵌入..."
    
    # 检查必要文件是否存在
    if ! check_file_exists "$MODEL_WEIGHT"; then
        print_error "无法继续，模型权重文件不存在"
        return 1
    fi
    
    if ! check_file_exists "$SAMPLES_FILE"; then
        print_error "无法继续，采样文件不存在"
        return 1
    fi
    
    if ! check_file_exists "$CANDIDATES_FILE"; then
        print_error "无法继续，候选文档文件不存在"
        return 1
    fi
    
    # 运行嵌入构建脚本
    print_info "运行嵌入构建脚本..."
    python "$SCRIPT_DIR/retrieval_evaluator.py" \
       --embeddings_dir "$EMBEDDING_DIR" \
       --output_dir "$RESULTS_DIR"
    
    if [[ $? -ne 0 ]]; then
        print_error "嵌入构建失败"
        return 1
    fi
    
    # 检查嵌入文件是否生成
    if [[ -f "$EMBEDDING_DIR/query_embeddings.npz" ]] && \
       [[ -f "$EMBEDDING_DIR/dnmsr_gallery_embeddings.npz" ]]; then
        print_success "嵌入构建成功"
        echo "嵌入文件保存在: $EMBEDDING_DIR"
        return 0
    else
        print_error "嵌入文件未生成"
        return 1
    fi
}

# 运行评估
run_evaluation() {
    print_info "开始运行评估..."
    
    # 检查嵌入文件是否存在
    if ! check_file_exists "$EMBEDDING_DIR/query_embeddings.npz" || \
       ! check_file_exists "$EMBEDDING_DIR/dnmsr_gallery_embeddings.npz"; then
        print_error "无法继续，嵌入文件不存在"
        return 1
    fi
    
    # 运行评估脚本
    print_info "运行评估脚本..."
    python "$SCRIPT_DIR/retrieval_evaluator.py" \
           --embeddings_dir "$EMBEDDING_DIR" \
           --output_dir "$RESULTS_DIR"
    
    if [[ $? -ne 0 ]]; then
        print_error "评估运行失败"
        return 1
    fi
    
    print_success "评估完成"
    echo "评估结果保存在: $RESULTS_DIR"
    return 0
}

# 运行整个流程
run_all() {
    print_info "开始运行完整评估流程..."
    
    # 1. 生成采样和候选文档
    generate_samples
    if [[ $? -ne 0 ]]; then
        print_error "无法继续，采样生成失败"
        return 1
    fi
    
    # 2. 构建嵌入
    build_embeddings
    if [[ $? -ne 0 ]]; then
        print_error "无法继续，嵌入构建失败"
        return 1
    fi
    
    # 3. 运行评估
    run_evaluation
    if [[ $? -ne 0 ]]; then
        print_error "评估运行失败"
        return 1
    fi
    
    print_success "完整评估流程运行成功"
    return 0
}

# 检查模块导入问题
check_imports() {
    print_info "检查模块导入情况..."
    
    # 尝试导入关键模块
    python -c "
import sys
import os
import torch
print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

# 设置Python路径
sys.path.insert(0, '$DNMSR_ROOT')

try:
    import visual_bge
    print('visual_bge 导入成功')
    from visual_bge.modeling import Visualized_BGE
    print('Visualized_BGE 导入成功')
except ImportError as e:
    print('导入visual_bge失败:', e)
    print('当前Python路径:')
    for p in sys.path:
        print(' -', p)
    sys.exit(1)

try:
    from visual_bge.eva_clip import create_eva_vision_and_transforms
    print('eva_clip 导入成功')
except ImportError as e:
    print('导入eva_clip失败:', e)
    sys.exit(1)

print('所有模块导入检查通过')
"
    
    if [[ $? -ne 0 ]]; then
        print_error "模块导入检查失败"
        return 1
    fi
    
    print_success "模块导入检查通过"
    return 0
}

# 显示交互式菜单
show_menu() {
    echo ""
    echo -e "${BOLD}DNMSR 评估系统${NC}"
    echo "=================================="
    echo "1) 显示当前配置"
    echo "2) 更新配置参数"
    echo "3) 检查模块导入情况"
    echo "4) 生成采样和候选文档"
    echo "5) 构建嵌入向量"
    echo "6) 运行检索评估"
    echo "7) 运行完整评估流程"
    echo "0) 退出"
    echo "=================================="
    read -p "请选择操作 [0-7]: " choice
    
    case $choice in
        1) show_config ;;
        2) update_config ;;
        3) check_imports ;;
        4) generate_samples ;;
        5) build_embeddings ;;
        6) run_evaluation ;;
        7) run_all ;;
        0) print_info "退出程序"; exit 0 ;;
        *) print_error "无效选择" ;;
    esac
    
    # 继续显示菜单，除非用户选择退出
    show_menu
}

# 主函数
main() {
    # 打印欢迎信息
    echo -e "${BOLD}欢迎使用DNMSR暗网多模态检索评估系统${NC}"
    echo "=================================="
    
    # 显示交互式菜单
    show_menu
}

# 运行主函数
main 