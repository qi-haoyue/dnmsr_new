#!/bin/bash
# 用于运行DNMSR评估的完整流程脚本

# 获取当前脚本所在目录（绝对路径）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DNMSR_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")



# 默认配置参数 - 使用相对路径便于在不同环境运行
MODEL_WEIGHT="$DNMSR_ROOT/Visualized_m3.pth"
DATA_ROOT="/home/qhy/MML/DNM_dataset/changan"
JSON_FILE="/home/qhy/MML/DNM_dataset/changan/ca_fixed.json"
IMAGE_DIR="/home/qhy/MML/DNM_dataset/changan/pic/full"
SAMPLES_DIR="$DNMSR_ROOT/dnmsr_new/data_preprocessing/samples"
SAMPLES_FILE="$SAMPLES_DIR/sampled_products.json"
CANDIDATES_FILE="$SAMPLES_DIR/candidate_documents.json"
EMBEDDING_DIR="$SCRIPT_DIR/embeddings"
RESULTS_DIR="$SCRIPT_DIR/results"

# 为三种检索类型创建结果目录
MULTIMODAL_RESULTS_DIR="$RESULTS_DIR/multimodal"
TEXT_TO_IMAGE_RESULTS_DIR="$RESULTS_DIR/text2image"
IMAGE_TO_TEXT_RESULTS_DIR="$RESULTS_DIR/image2text"

# 设置环境变量 - 确保Python能找到模块
export PYTHONPATH="$DNMSR_ROOT:$PYTHONPATH"

# 确保必要的目录存在
mkdir -p "$SAMPLES_DIR"
mkdir -p "$EMBEDDING_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$MULTIMODAL_RESULTS_DIR/evaluation"
mkdir -p "$MULTIMODAL_RESULTS_DIR/examples"
mkdir -p "$TEXT_TO_IMAGE_RESULTS_DIR/evaluation"
mkdir -p "$TEXT_TO_IMAGE_RESULTS_DIR/examples"
mkdir -p "$IMAGE_TO_TEXT_RESULTS_DIR/evaluation"
mkdir -p "$IMAGE_TO_TEXT_RESULTS_DIR/examples"

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
    echo "结果输出目录结构:"
    echo "  ├── 多模态混合检索: $MULTIMODAL_RESULTS_DIR"
    echo "  ├── 文搜图: $TEXT_TO_IMAGE_RESULTS_DIR"
    echo "  └── 图搜文: $IMAGE_TO_TEXT_RESULTS_DIR"
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
    print_info "采样文件将保存到: $SAMPLES_FILE"
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
    python "$SCRIPT_DIR/embedding_builder.py" \
       --dnmsr_model_path "$MODEL_WEIGHT" \
       --samples_file "$SAMPLES_FILE" \
       --candidates_file "$CANDIDATES_FILE" \
       --output_dir "$EMBEDDING_DIR"
    
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

# 运行多模态混合检索评估
run_multimodal_evaluation() {
    print_info "开始运行多模态混合检索评估..."
    
    # 检查嵌入文件是否存在
    if ! check_file_exists "$EMBEDDING_DIR/query_embeddings.npz" || \
       ! check_file_exists "$EMBEDDING_DIR/dnmsr_gallery_embeddings.npz"; then
        print_error "无法继续，嵌入文件不存在"
        return 1
    fi
    
    # 检查候选文档文件是否存在
    if ! check_file_exists "$CANDIDATES_FILE"; then
        print_warning "候选文档文件不存在，将无法进行完整评估"
    fi
    
    # 运行评估脚本
    print_info "运行多模态混合检索评估脚本..."
    python "$SCRIPT_DIR/retrieval_evaluator.py" \
           --embeddings_dir "$EMBEDDING_DIR" \
           --output_dir "$MULTIMODAL_RESULTS_DIR/evaluation"
    
    if [[ $? -ne 0 ]]; then
        print_error "多模态混合检索评估运行失败"
        return 1
    fi
    
    print_success "多模态混合检索评估完成"
    echo "评估结果保存在: $MULTIMODAL_RESULTS_DIR/evaluation"
    return 0
}

# 生成多模态混合检索示例分析
generate_multimodal_examples() {
    print_info "开始生成多模态混合检索示例分析..."
    
    # 检查评估结果文件是否存在
    if ! check_file_exists "$MULTIMODAL_RESULTS_DIR/evaluation/retrieval_results.json"; then
        print_error "无法继续，多模态混合检索评估结果文件不存在"
        return 1
    fi
    
    # 检查嵌入文件是否存在
    if ! check_dir_exists "$EMBEDDING_DIR"; then
        print_error "无法继续，嵌入目录不存在"
        return 1
    fi
    
    # 检查候选文档文件是否存在
    if ! check_file_exists "$CANDIDATES_FILE"; then
        print_warning "候选文档文件不存在，将无法显示文档内容"
        CANDIDATES_PARAM=""
    else
        CANDIDATES_PARAM="--candidates_file $CANDIDATES_FILE"
    fi
    
    # 运行检索示例分析脚本
    print_info "运行多模态混合检索示例分析脚本..."
    python "$SCRIPT_DIR/generate_examples.py" \
           --results_file "$MULTIMODAL_RESULTS_DIR/evaluation/retrieval_results.json" \
           --embedding_dir "$EMBEDDING_DIR" \
           --output_dir "$MULTIMODAL_RESULTS_DIR/examples" \
           --num_examples 10 \
           --top_k 10 \
           $CANDIDATES_PARAM
    
    if [[ $? -ne 0 ]]; then
        print_error "多模态混合检索示例分析生成失败"
        return 1
    fi
    
    if check_file_exists "$MULTIMODAL_RESULTS_DIR/examples/retrieval_examples.txt"; then
        print_success "多模态混合检索示例分析生成成功"
        echo "检索示例文件: $MULTIMODAL_RESULTS_DIR/examples/retrieval_examples.txt"
        return 0
    else
        print_error "多模态混合检索示例文件未生成"
        return 1
    fi
}

# 运行图搜文评估
run_image_to_text_evaluation() {
    print_info "开始运行图搜文检索评估..."
    
    # 检查嵌入文件是否存在
    if ! check_file_exists "$EMBEDDING_DIR/query_embeddings.npz" || \
       ! check_file_exists "$EMBEDDING_DIR/dnmsr_gallery_embeddings.npz"; then
        print_error "无法继续，嵌入文件不存在"
        return 1
    fi
    
    # 检查候选文档文件是否存在
    if ! check_file_exists "$CANDIDATES_FILE"; then
        print_warning "候选文档文件不存在，将无法进行完整评估"
    fi
    
    # 运行评估脚本，指定图搜文模式
    print_info "运行图搜文检索评估脚本..."
    python "$SCRIPT_DIR/retrieval_evaluator.py" \
           --embeddings_dir "$EMBEDDING_DIR" \
           --output_dir "$IMAGE_TO_TEXT_RESULTS_DIR/evaluation" \
           --mode image_to_text
    
    if [[ $? -ne 0 ]]; then
        print_error "图搜文检索评估运行失败"
        return 1
    fi
    
    print_success "图搜文检索评估完成"
    echo "评估结果保存在: $IMAGE_TO_TEXT_RESULTS_DIR/evaluation"
    return 0
}

# 生成图搜文检索示例分析
generate_image_to_text_examples() {
    print_info "开始生成图搜文检索示例分析..."
    
    # 检查评估结果文件是否存在
    if ! check_file_exists "$IMAGE_TO_TEXT_RESULTS_DIR/evaluation/retrieval_results.json"; then
        print_error "无法继续，图搜文检索评估结果文件不存在"
        return 1
    fi
    
    # 检查嵌入文件是否存在
    if ! check_dir_exists "$EMBEDDING_DIR"; then
        print_error "无法继续，嵌入目录不存在"
        return 1
    fi
    
    # 检查候选文档文件是否存在
    if ! check_file_exists "$CANDIDATES_FILE"; then
        print_warning "候选文档文件不存在，将无法显示文档内容"
        CANDIDATES_PARAM=""
    else
        CANDIDATES_PARAM="--candidates_file $CANDIDATES_FILE"
    fi
    
    # 运行图搜文检索示例分析脚本
    print_info "运行图搜文检索示例分析脚本..."
    python "$SCRIPT_DIR/generate_examples.py" \
           --results_file "$IMAGE_TO_TEXT_RESULTS_DIR/evaluation/retrieval_results.json" \
           --embedding_dir "$EMBEDDING_DIR" \
           --output_dir "$IMAGE_TO_TEXT_RESULTS_DIR/examples" \
           --num_examples 10 \
           --top_k 10 \
           --mode image_to_text \
           $CANDIDATES_PARAM
    
    if [[ $? -ne 0 ]]; then
        print_error "图搜文检索示例分析生成失败"
        return 1
    fi
    
    if check_file_exists "$IMAGE_TO_TEXT_RESULTS_DIR/examples/image_to_text_examples.txt"; then
        print_success "图搜文检索示例分析生成成功"
        echo "图搜文检索示例文件: $IMAGE_TO_TEXT_RESULTS_DIR/examples/image_to_text_examples.txt"
        return 0
    else
        print_error "图搜文检索示例文件未生成"
        return 1
    fi
}

# 运行文搜图评估
run_text_to_image_evaluation() {
    print_info "开始运行文搜图检索评估..."
    
    # 检查嵌入文件是否存在
    if ! check_file_exists "$EMBEDDING_DIR/query_embeddings.npz" || \
       ! check_file_exists "$EMBEDDING_DIR/dnmsr_gallery_embeddings.npz"; then
        print_error "无法继续，嵌入文件不存在"
        return 1
    fi
    
    # 检查候选文档文件是否存在
    if ! check_file_exists "$CANDIDATES_FILE"; then
        print_warning "候选文档文件不存在，将无法进行完整评估"
    fi
    
    # 运行评估脚本，指定文搜图模式
    print_info "运行文搜图检索评估脚本..."
    python "$SCRIPT_DIR/retrieval_evaluator.py" \
           --embeddings_dir "$EMBEDDING_DIR" \
           --output_dir "$TEXT_TO_IMAGE_RESULTS_DIR/evaluation" \
           --mode text_to_image
    
    if [[ $? -ne 0 ]]; then
        print_error "文搜图检索评估运行失败"
        return 1
    fi
    
    print_success "文搜图检索评估完成"
    echo "评估结果保存在: $TEXT_TO_IMAGE_RESULTS_DIR/evaluation"
    return 0
}

# 生成文搜图检索示例分析
generate_text_to_image_examples() {
    print_info "开始生成文搜图检索示例分析..."
    
    # 检查评估结果文件是否存在
    if ! check_file_exists "$TEXT_TO_IMAGE_RESULTS_DIR/evaluation/retrieval_results.json"; then
        print_error "无法继续，文搜图检索评估结果文件不存在"
        return 1
    fi
    
    # 检查嵌入文件是否存在
    if ! check_dir_exists "$EMBEDDING_DIR"; then
        print_error "无法继续，嵌入目录不存在"
        return 1
    fi
    
    # 检查候选文档文件是否存在
    if ! check_file_exists "$CANDIDATES_FILE"; then
        print_warning "候选文档文件不存在，将无法显示文档内容"
        CANDIDATES_PARAM=""
    else
        CANDIDATES_PARAM="--candidates_file $CANDIDATES_FILE"
    fi
    
    # 运行文搜图检索示例分析脚本
    print_info "运行文搜图检索示例分析脚本..."
    python "$SCRIPT_DIR/generate_examples.py" \
           --results_file "$TEXT_TO_IMAGE_RESULTS_DIR/evaluation/retrieval_results.json" \
           --embedding_dir "$EMBEDDING_DIR" \
           --output_dir "$TEXT_TO_IMAGE_RESULTS_DIR/examples" \
           --num_examples 10 \
           --top_k 10 \
           --mode text_to_image \
           $CANDIDATES_PARAM
    
    if [[ $? -ne 0 ]]; then
        print_error "文搜图检索示例分析生成失败"
        return 1
    fi
    
    if check_file_exists "$TEXT_TO_IMAGE_RESULTS_DIR/examples/text_to_image_examples.txt"; then
        print_success "文搜图检索示例分析生成成功"
        echo "文搜图检索示例文件: $TEXT_TO_IMAGE_RESULTS_DIR/examples/text_to_image_examples.txt"
        return 0
    else
        print_error "文搜图检索示例文件未生成"
        return 1
    fi
}

# 多模态混合检索菜单
show_multimodal_menu() {
    echo ""
    echo -e "${BOLD}多模态混合检索子菜单${NC}"
    echo "=================================="
    echo "1) 运行多模态检索评估"
    echo "2) 生成多模态检索示例分析"
    echo "0) 返回主菜单"
    echo "=================================="
    read -p "请选择操作 [0-2]: " choice
    
    case $choice in
        1) run_multimodal_evaluation ;;
        2) generate_multimodal_examples ;;
        0) return ;;
        *) print_error "无效选择" ;;
    esac
    
    # 继续显示子菜单，除非用户选择返回
    show_multimodal_menu
}

# 图搜文菜单
show_image_to_text_menu() {
    echo ""
    echo -e "${BOLD}图搜文子菜单${NC}"
    echo "=================================="
    echo "1) 运行图搜文检索评估"
    echo "2) 生成图搜文检索示例分析"
    echo "0) 返回主菜单"
    echo "=================================="
    read -p "请选择操作 [0-2]: " choice
    
    case $choice in
        1) run_image_to_text_evaluation ;;
        2) generate_image_to_text_examples ;;
        0) return ;;
        *) print_error "无效选择" ;;
    esac
    
    # 继续显示子菜单，除非用户选择返回
    show_image_to_text_menu
}

# 文搜图菜单
show_text_to_image_menu() {
    echo ""
    echo -e "${BOLD}文搜图子菜单${NC}"
    echo "=================================="
    echo "1) 运行文搜图检索评估"
    echo "2) 生成文搜图检索示例分析"
    echo "0) 返回主菜单"
    echo "=================================="
    read -p "请选择操作 [0-2]: " choice
    
    case $choice in
        1) run_text_to_image_evaluation ;;
        2) generate_text_to_image_examples ;;
        0) return ;;
        *) print_error "无效选择" ;;
    esac
    
    # 继续显示子菜单，除非用户选择返回
    show_text_to_image_menu
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
    
    # 3. 运行多模态混合检索评估和示例分析
    run_multimodal_evaluation
    generate_multimodal_examples
    
    # 4. 运行图搜文检索评估和示例分析
    run_image_to_text_evaluation
    generate_image_to_text_examples
    
    # 5. 运行文搜图检索评估和示例分析
    run_text_to_image_evaluation
    generate_text_to_image_examples
    
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
    echo "6) 多模态混合检索"
    echo "7) 图搜文"
    echo "8) 文搜图"
    echo "9) 运行完整评估流程"
    echo "0) 退出"
    echo "=================================="
    read -p "请选择操作 [0-9]: " choice
    
    case $choice in
        1) show_config ;;
        2) update_config ;;
        3) check_imports ;;
        4) generate_samples ;;
        5) build_embeddings ;;
        6) show_multimodal_menu ;;
        7) show_image_to_text_menu ;;
        8) show_text_to_image_menu ;;
        9) run_all ;;
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
    echo "当前工作目录: $(pwd)"
    echo "DNMSR根目录: $DNMSR_ROOT"
    echo "当前脚本目录: $SCRIPT_DIR"
    
    # 显示交互式菜单
    show_menu
}

# 运行主函数
main 