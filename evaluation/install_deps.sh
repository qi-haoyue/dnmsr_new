#!/bin/bash
# 安装DNMSR评估所需的依赖包

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DNMSR_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

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

# 检查Python版本
check_python_version() {
    print_info "检查Python版本..."
    
    python_version=$(python --version 2>&1 | awk '{print $2}')
    python_major=$(echo $python_version | cut -d. -f1)
    python_minor=$(echo $python_version | cut -d. -f2)
    
    echo "当前Python版本: $python_version"
    
    if [[ $python_major -lt 3 || ($python_major -eq 3 && $python_minor -lt 6) ]]; then
        print_error "Python版本过低，需要Python 3.6或更高版本"
        return 1
    else
        print_success "Python版本符合要求"
        return 0
    fi
}

# 安装基础依赖
install_basic_deps() {
    print_info "安装基础依赖..."
    
    # 安装基础包
    pip install --upgrade pip
    pip install numpy tqdm pillow requests 
    
    if [ $? -ne 0 ]; then
        print_error "安装基础依赖失败"
        return 1
    else
        print_success "基础依赖安装成功"
        return 0
    fi
}

# 安装PyTorch
install_pytorch() {
    print_info "安装PyTorch..."
    
    # 检查CUDA是否可用
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        cuda_major=$(echo $cuda_version | cut -d. -f1)
        cuda_minor=$(echo $cuda_version | cut -d. -f2)
        
        print_info "检测到CUDA版本: $cuda_version"
        
        # 根据CUDA版本安装适配的PyTorch
        if [[ $cuda_major -ge 11 ]]; then
            print_info "使用CUDA 11.x安装PyTorch..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        elif [[ $cuda_major -eq 10 ]]; then
            print_info "使用CUDA 10.x安装PyTorch..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu102
        else
            print_warning "CUDA版本可能不受支持，尝试安装最新PyTorch CPU版本"
            pip install torch torchvision torchaudio
        fi
    else
        print_info "未检测到CUDA，安装PyTorch CPU版本..."
        pip install torch torchvision torchaudio
    fi
    
    if [ $? -ne 0 ]; then
        print_error "安装PyTorch失败"
        return 1
    else
        print_success "PyTorch安装成功"
        
        # 验证PyTorch安装
        python -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
    print('GPU型号:', torch.cuda.get_device_name(0))
"
        return 0
    fi
}

# 安装HuggingFace Transformers
install_transformers() {
    print_info "安装HuggingFace Transformers..."
    
    pip install transformers sentencepiece
    
    if [ $? -ne 0 ]; then
        print_error "安装Transformers失败"
        return 1
    else
        print_success "Transformers安装成功"
        return 0
    fi
}

# 安装BGE模型依赖
install_bge_deps() {
    print_info "安装BGE模型依赖..."
    
    pip install faiss-cpu
    
    if [ $? -ne 0 ]; then
        print_error "安装BGE依赖失败"
        return 1
    else
        print_success "BGE依赖安装成功"
        return 0
    fi
}

# 安装评估依赖
install_eval_deps() {
    print_info "安装评估依赖..."
    
    pip install matplotlib scikit-learn
    
    if [ $? -ne 0 ]; then
        print_error "安装评估依赖失败"
        return 1
    else
        print_success "评估依赖安装成功"
        return 0
    fi
}

# 创建requirements.txt文件
create_requirements() {
    print_info "创建requirements.txt文件..."
    
    cat > "$SCRIPT_DIR/requirements.txt" << EOF
# DNMSR评估所需依赖
numpy>=1.19.0
torch>=1.7.0
torchvision>=0.8.0
Pillow>=8.0.0
tqdm>=4.50.0
transformers>=4.20.0
sentencepiece>=0.1.96
faiss-cpu>=1.7.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
requests>=2.25.0
EOF
    
    print_success "requirements.txt文件已创建: $SCRIPT_DIR/requirements.txt"
    echo "可以使用以下命令安装所有依赖:"
    echo "pip install -r $SCRIPT_DIR/requirements.txt"
}

# 主函数
main() {
    echo -e "${BOLD}安装DNMSR评估所需的依赖包${NC}"
    echo "=================================================="
    
    # 检查Python版本
    check_python_version
    if [ $? -ne 0 ]; then
        print_error "Python版本检查失败，请安装Python 3.6或更高版本"
        return 1
    fi
    
    # 提供选项菜单
    echo ""
    echo -e "${BOLD}请选择安装选项:${NC}"
    echo "1) 安装所有依赖 (推荐)"
    echo "2) 仅安装基础依赖"
    echo "3) 仅安装PyTorch"
    echo "4) 仅安装Transformers和BGE依赖"
    echo "5) 创建requirements.txt文件"
    echo "0) 退出"
    
    read -p "请选择 [1-5, 0]: " option
    
    case $option in
        1)
            # 安装所有依赖
            install_basic_deps
            install_pytorch
            install_transformers
            install_bge_deps
            install_eval_deps
            create_requirements
            ;;
        2)
            # 仅安装基础依赖
            install_basic_deps
            ;;
        3)
            # 仅安装PyTorch
            install_pytorch
            ;;
        4)
            # 仅安装Transformers和BGE依赖
            install_transformers
            install_bge_deps
            ;;
        5)
            # 创建requirements.txt文件
            create_requirements
            ;;
        0)
            print_info "退出安装"
            return 0
            ;;
        *)
            print_error "无效选项"
            return 1
            ;;
    esac
    
    # 打印完成信息
    echo ""
    echo "=================================================="
    echo -e "${BOLD}依赖安装完成${NC}"
    echo ""
    echo "下一步:"
    echo "1) 运行环境设置脚本: python $SCRIPT_DIR/setup_module.py"
    echo "2) 检查必要文件: bash $SCRIPT_DIR/check_files.sh"
    echo "3) 运行评估: bash $SCRIPT_DIR/start.sh"
    echo ""
    
    return 0
}

# 执行主函数
main 