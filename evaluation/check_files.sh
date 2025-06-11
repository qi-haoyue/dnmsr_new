#!/bin/bash
# 检查DNMSR评估所需的所有必要文件

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

# 检查文件是否存在
check_file() {
    local file_path="$1"
    local file_desc="$2"
    
    if [ -f "$file_path" ]; then
        print_success "$file_desc 存在: $file_path"
        return 0
    else
        print_error "$file_desc 不存在: $file_path"
        return 1
    fi
}

# 检查目录是否存在
check_dir() {
    local dir_path="$1"
    local dir_desc="$2"
    
    if [ -d "$dir_path" ]; then
        print_success "$dir_desc 存在: $dir_path"
        return 0
    else
        print_error "$dir_desc 不存在: $dir_path"
        return 1
    fi
}

# 检查Python模块是否可导入
check_module() {
    local module_name="$1"
    
    python -c "import $module_name" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Python模块 '$module_name' 可以导入"
        return 0
    else
        print_error "Python模块 '$module_name' 无法导入"
        return 1
    fi
}

# 主函数
main() {
    echo -e "${BOLD}检查DNMSR评估所需的所有必要文件${NC}"
    echo "=================================================="
    
    # 设置计数器
    local success_count=0
    local error_count=0
    
    print_info "检查项目目录结构..."
    
    # 检查项目根目录
    check_dir "$DNMSR_ROOT" "DNMSR根目录"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    # 检查评估脚本
    check_file "$SCRIPT_DIR/run_all.sh" "评估运行脚本"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    check_file "$SCRIPT_DIR/embedding_builder.py" "嵌入构建脚本"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    check_file "$SCRIPT_DIR/setup_module.py" "模块设置脚本"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    check_file "$SCRIPT_DIR/start.sh" "启动脚本"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    # 检查模型文件
    print_info "检查模型文件..."
    
    check_file "$DNMSR_ROOT/Visualized_m3.pth" "DNMSR模型权重文件"
    model_status=$?
    [ $model_status -eq 0 ] && ((success_count++)) || ((error_count++))
    
    # 如果模型文件不存在，搜索可能的位置
    if [ $model_status -ne 0 ]; then
        print_info "搜索模型文件的可能位置..."
        
        # 可能的模型文件位置
        possible_locations=(
            "$HOME/MML/dnmsr/Visualized_m3.pth"
            "$DNMSR_ROOT/../Visualized_m3.pth"
            "$DNMSR_ROOT/dnmsr_new/Visualized_m3.pth"
            "$DNMSR_ROOT/models/Visualized_m3.pth"
        )
        
        for location in "${possible_locations[@]}"; do
            if [ -f "$location" ]; then
                print_success "在 $location 找到模型文件"
                print_info "建议创建符号链接: ln -s $location $DNMSR_ROOT/Visualized_m3.pth"
                break
            fi
        done
    fi
    
    # 检查visual_bge模块文件
    print_info "检查visual_bge模块文件..."
    
    visual_bge_path="$DNMSR_ROOT/visual_bge"
    check_dir "$visual_bge_path" "visual_bge模块目录"
    vbge_status=$?
    [ $vbge_status -eq 0 ] && ((success_count++)) || ((error_count++))
    
    if [ $vbge_status -eq 0 ]; then
        # 检查重要文件
        check_file "$visual_bge_path/modeling.py" "visual_bge/modeling.py文件"
        [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
        
        check_file "$visual_bge_path/__init__.py" "visual_bge/__init__.py文件"
        [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
        
        # 检查eva_clip子模块
        check_dir "$visual_bge_path/eva_clip" "eva_clip子模块目录"
        eva_status=$?
        [ $eva_status -eq 0 ] && ((success_count++)) || ((error_count++))
        
        if [ $eva_status -eq 0 ]; then
            check_file "$visual_bge_path/eva_clip/__init__.py" "eva_clip/__init__.py文件"
            [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
            
            check_file "$visual_bge_path/eva_clip/factory.py" "eva_clip/factory.py文件"
            [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
        fi
    else
        # 检查dnmsr_new/visual_bge目录
        alt_visual_bge_path="$DNMSR_ROOT/dnmsr_new/visual_bge"
        check_dir "$alt_visual_bge_path" "替代visual_bge模块目录"
        alt_vbge_status=$?
        
        if [ $alt_vbge_status -eq 0 ]; then
            print_info "找到替代visual_bge目录，可以创建符号链接解决问题"
            print_info "建议执行: ln -s $alt_visual_bge_path $visual_bge_path"
            
            # 检查重要文件
            check_file "$alt_visual_bge_path/modeling.py" "替代visual_bge/modeling.py文件"
            
            # 检查eva_clip子模块
            check_dir "$alt_visual_bge_path/eva_clip" "替代eva_clip子模块目录"
        fi
    fi
    
    # 检查Python模块导入
    print_info "检查Python模块导入..."
    
    # 尝试设置PYTHONPATH
    export PYTHONPATH="$DNMSR_ROOT:$PYTHONPATH"
    
    check_module "torch"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    check_module "numpy"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    check_module "PIL"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    check_module "tqdm"
    [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    
    # 尝试导入visual_bge模块
    python -c "
import sys
sys.path.insert(0, '$DNMSR_ROOT')
try:
    import visual_bge
    print('visual_bge模块位置:', visual_bge.__file__)
    exit(0)
except ImportError as e:
    print('导入visual_bge失败:', e)
    exit(1)
"
    if [ $? -eq 0 ]; then
        print_success "可以导入visual_bge模块"
        ((success_count++))
    else
        print_error "无法导入visual_bge模块，需要运行setup_module.py解决"
        ((error_count++))
    fi
    
    # 检查数据文件
    print_info "检查数据文件..."
    
    # 检查数据目录
    data_root="$DNMSR_ROOT/../changan"
    check_dir "$data_root" "数据根目录"
    data_status=$?
    
    if [ $data_status -eq 0 ]; then
        check_file "$data_root/ca_fixed.json" "商品数据JSON文件"
        [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
        
        check_dir "$data_root/pic/full" "图像目录"
        [ $? -eq 0 ] && ((success_count++)) || ((error_count++))
    else
        print_warning "数据目录不存在，请在运行评估时指定正确的数据路径"
    fi
    
    # 输出结果汇总
    echo ""
    echo "=================================================="
    echo -e "${BOLD}检查结果汇总:${NC}"
    echo -e "${GREEN}成功:${NC} $success_count 项"
    echo -e "${RED}错误:${NC} $error_count 项"
    
    if [ $error_count -eq 0 ]; then
        echo -e "${GREEN}所有必要文件都存在，可以开始运行评估${NC}"
        return 0
    elif [ $error_count -lt 5 ]; then
        echo -e "${YELLOW}存在一些问题，但可能通过运行setup_module.py解决${NC}"
        echo "建议执行:"
        echo "  cd $SCRIPT_DIR"
        echo "  python setup_module.py"
        return 1
    else
        echo -e "${RED}存在多个严重问题，请先解决后再运行评估${NC}"
        return 2
    fi
}

# 执行主函数
main 