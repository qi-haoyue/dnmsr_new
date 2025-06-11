#!/bin/bash
# DNMSR评估系统 - 综合启动脚本

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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

# 打印标题
print_title() {
    echo ""
    echo -e "${BOLD}$1${NC}"
    echo "=================================================="
}

# 显示欢迎信息
print_welcome() {
    clear
    echo -e "${BOLD}=================================================${NC}"
    echo -e "${BOLD}  DNMSR暗网多模态检索评估系统 - 综合启动脚本  ${NC}"
    echo -e "${BOLD}=================================================${NC}"
    echo ""
    echo "此脚本提供了一站式解决方案，用于解决模块导入问题并运行评估"
    echo ""
}

# 设置环境
setup_environment() {
    print_title "1. 环境设置"
    
    # 确保setup_module.py存在
    if [ ! -f "$SCRIPT_DIR/setup_module.py" ]; then
        print_error "找不到setup_module.py脚本，无法设置环境"
        return 1
    fi
    
    # 运行模块设置脚本
    print_info "运行模块设置脚本..."
    python "$SCRIPT_DIR/setup_module.py"
    
    if [ $? -ne 0 ]; then
        print_error "环境设置失败"
        echo ""
        print_info "尝试手动导入模块路径..."
        
        # 尝试手动设置PYTHONPATH
        DNMSR_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
        export PYTHONPATH="$DNMSR_ROOT:$PYTHONPATH"
        print_info "已设置PYTHONPATH=$PYTHONPATH"
        
        return 1
    fi
    
    # 应用环境变量
    if [ -f "$SCRIPT_DIR/set_env.sh" ]; then
        print_info "应用环境变量设置..."
        source "$SCRIPT_DIR/set_env.sh"
    fi
    
    print_success "环境设置完成"
    return 0
}

# 检查依赖
check_dependencies() {
    print_title "2. 检查依赖"
    
    # 检查必要的Python包
    print_info "检查Python依赖..."
    python -c "
import sys
try:
    import torch
    import numpy as np
    import tqdm
    import PIL
    print('PyTorch版本:', torch.__version__)
    print('CUDA可用:', torch.cuda.is_available())
    print('NumPy版本:', np.__version__)
    print('PIL版本:', PIL.__version__)
    sys.exit(0)
except ImportError as e:
    print('缺少依赖:', e)
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        print_error "依赖检查失败，某些必要的Python包缺失"
        
        # 提示安装依赖
        echo ""
        echo -e "${BOLD}建议安装以下依赖:${NC}"
        echo "pip install torch numpy tqdm pillow"
        echo ""
        
        read -p "是否现在安装这些依赖? [y/N] " install_deps
        if [[ $install_deps == "y" || $install_deps == "Y" ]]; then
            pip install torch numpy tqdm pillow
            if [ $? -ne 0 ]; then
                print_error "依赖安装失败"
                return 1
            else
                print_success "依赖安装成功"
            fi
        else
            return 1
        fi
    else
        print_success "所有必要的Python依赖已安装"
    fi
    
    # 检查模型文件
    print_info "检查模型文件..."
    DNMSR_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
    MODEL_PATH="$DNMSR_ROOT/Visualized_m3.pth"
    
    if [ ! -f "$MODEL_PATH" ]; then
        print_warning "找不到模型文件: $MODEL_PATH"
        echo "请确保模型文件位于正确位置，或在运行评估时指定正确的路径"
    else
        print_success "找到模型文件: $MODEL_PATH"
    fi
    
    return 0
}

# 运行评估
run_evaluation() {
    print_title "3. 运行评估"
    
    # 检查评估脚本是否存在
    if [ ! -f "$SCRIPT_DIR/run_all.sh" ]; then
        print_error "找不到评估脚本: $SCRIPT_DIR/run_all.sh"
        return 1
    fi
    
    # 提示用户
    print_info "准备运行评估..."
    echo "评估脚本将提供交互式菜单，您可以选择执行完整评估或单独步骤"
    echo ""
    
    read -p "是否继续? [Y/n] " continue_eval
    if [[ $continue_eval == "n" || $continue_eval == "N" ]]; then
        print_info "已取消评估"
        return 0
    fi
    
    # 运行评估脚本
    bash "$SCRIPT_DIR/run_all.sh"
    
    return $?
}

# 显示帮助信息
show_help() {
    print_title "DNMSR评估系统 - 帮助信息"
    echo "用法: ./start.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -e, --env      仅设置环境"
    echo "  -d, --deps     仅检查依赖"
    echo "  -r, --run      仅运行评估"
    echo "  --debug        启用调试模式"
    echo ""
    echo "不带参数运行将执行完整流程: 设置环境 -> 检查依赖 -> 运行评估"
    echo ""
}

# 诊断模式
run_diagnostics() {
    print_title "DNMSR系统诊断"
    
    print_info "系统信息:"
    echo "操作系统: $(uname -s)"
    echo "Python版本: $(python --version 2>&1)"
    echo "工作目录: $(pwd)"
    echo "脚本目录: $SCRIPT_DIR"
    
    print_info "目录结构:"
    DNMSR_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
    echo "DNMSR根目录: $DNMSR_ROOT"
    echo "目录内容:"
    ls -la "$DNMSR_ROOT"
    
    print_info "Python路径:"
    python -c "import sys; print('\n'.join(sys.path))"
    
    print_info "尝试导入visual_bge模块:"
    python -c "
import sys
try:
    import visual_bge
    print('导入成功，模块位置:', visual_bge.__file__)
    print('模块内容:', dir(visual_bge))
except ImportError as e:
    print('导入失败:', e)
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        print_error "模块导入失败，请先运行环境设置"
    fi
}

# 主函数
main() {
    # 显示欢迎信息
    print_welcome
    
    # 处理命令行参数
    case "$1" in
        -h|--help)
            show_help
            return 0
            ;;
        -e|--env)
            setup_environment
            return $?
            ;;
        -d|--deps)
            check_dependencies
            return $?
            ;;
        -r|--run)
            run_evaluation
            return $?
            ;;
        --debug)
            run_diagnostics
            return $?
            ;;
        *)
            # 默认运行完整流程
            ;;
    esac
    
    # 运行完整流程
    setup_environment
    if [ $? -ne 0 ]; then
        print_warning "环境设置存在问题，但将继续执行"
    fi
    
    check_dependencies
    if [ $? -ne 0 ]; then
        print_error "依赖检查失败，无法继续"
        return 1
    fi
    
    run_evaluation
    eval_result=$?
    
    # 显示总结
    echo ""
    if [ $eval_result -eq 0 ]; then
        print_success "评估流程已完成"
    else
        print_error "评估流程失败，返回代码: $eval_result"
    fi
    
    return $eval_result
}

# 执行主函数
main "$@" 