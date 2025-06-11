#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模块导入环境设置脚本

此脚本用于解决DNMSR项目的模块导入问题，特别是'visual_bge'模块的导入问题。
它会：
1. 为所有必要的目录创建__init__.py文件
2. 将项目根目录添加到PYTHONPATH
3. 验证模块可以被正确导入
"""

import os
import sys
import importlib
import inspect
import glob
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取当前脚本的目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录 (DNMSR目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))


def create_init_files():
    """
    为所有缺少__init__.py文件的相关目录创建该文件
    """
    logger.info("创建必要的__init__.py文件...")
    
    # 需要检查的关键目录
    dirs_to_check = [
        os.path.join(PROJECT_ROOT, "visual_bge"),
        os.path.join(PROJECT_ROOT, "visual_bge", "eva_clip"),
        os.path.join(PROJECT_ROOT, "dnmsr_new"),
        os.path.join(PROJECT_ROOT, "dnmsr_new", "visual_bge"),
        os.path.join(PROJECT_ROOT, "dnmsr_new", "visual_bge", "eva_clip"),
        os.path.join(PROJECT_ROOT, "dnmsr_new", "evaluation"),
        os.path.join(PROJECT_ROOT, "dnmsr_new", "data_preprocessing"),
    ]
    
    # 创建缺失的__init__.py文件
    created_count = 0
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            init_file = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_file):
                try:
                    with open(init_file, 'w') as f:
                        f.write("# 自动生成的__init__.py文件，用于解决模块导入问题\n")
                    logger.info(f"创建了 {init_file}")
                    created_count += 1
                except Exception as e:
                    logger.error(f"创建 {init_file} 失败: {e}")
    
    logger.info(f"共创建了 {created_count} 个__init__.py文件")


def add_to_python_path():
    """
    将项目根目录添加到PYTHONPATH
    """
    logger.info("将项目根目录添加到PYTHONPATH...")
    
    # 检查项目根目录是否已在Python路径中
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        logger.info(f"已将 {PROJECT_ROOT} 添加到Python路径")
    else:
        logger.info(f"{PROJECT_ROOT} 已经在Python路径中")
    
    # 同时添加dnmsr_new目录
    dnmsr_new_dir = os.path.join(PROJECT_ROOT, "dnmsr_new")
    if os.path.exists(dnmsr_new_dir) and dnmsr_new_dir not in sys.path:
        sys.path.insert(0, dnmsr_new_dir)
        logger.info(f"已将 {dnmsr_new_dir} 添加到Python路径")
    
    # 打印当前Python路径
    logger.info("当前Python路径:")
    for path in sys.path:
        logger.info(f" - {path}")


def check_module_structure():
    """
    检查关键模块的文件结构
    """
    logger.info("检查模块文件结构...")
    
    # 检查visual_bge模块
    visual_bge_path = os.path.join(PROJECT_ROOT, "visual_bge")
    if os.path.exists(visual_bge_path) and os.path.isdir(visual_bge_path):
        files = glob.glob(os.path.join(visual_bge_path, "*.py"))
        logger.info(f"visual_bge目录包含 {len(files)} 个Python文件:")
        for file in files:
            logger.info(f" - {os.path.basename(file)}")
        
        # 检查eva_clip子模块
        eva_clip_path = os.path.join(visual_bge_path, "eva_clip")
        if os.path.exists(eva_clip_path) and os.path.isdir(eva_clip_path):
            files = glob.glob(os.path.join(eva_clip_path, "*.py"))
            logger.info(f"eva_clip子模块包含 {len(files)} 个Python文件:")
            for file in files:
                logger.info(f" - {os.path.basename(file)}")
        else:
            logger.warning(f"eva_clip子模块不存在或不是目录: {eva_clip_path}")
    else:
        logger.warning(f"visual_bge模块不存在或不是目录: {visual_bge_path}")
        
        # 检查dnmsr_new/visual_bge目录
        alt_visual_bge_path = os.path.join(PROJECT_ROOT, "dnmsr_new", "visual_bge")
        if os.path.exists(alt_visual_bge_path) and os.path.isdir(alt_visual_bge_path):
            logger.info(f"发现替代visual_bge目录: {alt_visual_bge_path}")
            files = glob.glob(os.path.join(alt_visual_bge_path, "*.py"))
            logger.info(f"替代visual_bge目录包含 {len(files)} 个Python文件:")
            for file in files:
                logger.info(f" - {os.path.basename(file)}")
            
            # 创建符号链接或复制文件
            logger.info("创建从dnmsr_new/visual_bge到项目根目录的链接...")
            try:
                os.symlink(alt_visual_bge_path, visual_bge_path)
                logger.info(f"成功创建符号链接: {alt_visual_bge_path} -> {visual_bge_path}")
            except Exception as e:
                logger.error(f"创建符号链接失败: {e}")
                logger.info("尝试复制文件...")
                
                # 如果无法创建符号链接，尝试复制文件
                import shutil
                try:
                    os.makedirs(visual_bge_path, exist_ok=True)
                    for file in glob.glob(os.path.join(alt_visual_bge_path, "*")):
                        if os.path.isfile(file):
                            shutil.copy2(file, visual_bge_path)
                        elif os.path.isdir(file):
                            dir_name = os.path.basename(file)
                            shutil.copytree(file, os.path.join(visual_bge_path, dir_name), dirs_exist_ok=True)
                    logger.info(f"成功复制文件到 {visual_bge_path}")
                except Exception as e:
                    logger.error(f"复制文件失败: {e}")


def create_env_file():
    """
    创建环境变量设置脚本
    """
    logger.info("创建环境变量设置脚本...")
    
    # 创建shell脚本
    env_sh_path = os.path.join(CURRENT_DIR, "set_env.sh")
    with open(env_sh_path, 'w') as f:
        f.write(f"""#!/bin/bash
# 环境变量设置脚本 - 自动生成

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="{PROJECT_ROOT}:$PYTHONPATH"

# 告知用户如何使用
echo "环境变量已设置。"
echo "项目根目录 {PROJECT_ROOT} 已添加到PYTHONPATH。"
echo "请使用 source {env_sh_path} 来激活这些设置。"
""")
    
    # 设置可执行权限
    os.chmod(env_sh_path, 0o755)
    logger.info(f"环境变量设置脚本已创建: {env_sh_path}")
    logger.info(f"请使用 'source {env_sh_path}' 来激活环境设置")


def verify_imports():
    """
    验证关键模块是否可以正确导入
    """
    logger.info("验证模块导入...")
    
    # 尝试导入visual_bge模块
    try:
        import visual_bge
        logger.info(f"成功导入visual_bge模块: {visual_bge.__file__}")
        
        try:
            from visual_bge.modeling import Visualized_BGE
            logger.info("成功导入Visualized_BGE类")
            
            # 打印类结构
            logger.info("Visualized_BGE类成员:")
            for name, member in inspect.getmembers(Visualized_BGE):
                if not name.startswith('_'):  # 排除私有成员
                    logger.info(f" - {name}: {type(member).__name__}")
            
        except ImportError as e:
            logger.error(f"无法导入Visualized_BGE类: {e}")
        
        try:
            from visual_bge.eva_clip import create_eva_vision_and_transforms
            logger.info("成功导入eva_clip子模块")
        except ImportError as e:
            logger.error(f"无法导入eva_clip子模块: {e}")
            
    except ImportError as e:
        logger.error(f"无法导入visual_bge模块: {e}")
        
        # 尝试诊断问题
        logger.info("正在诊断问题...")
        
        # 检查目录是否存在
        visual_bge_path = os.path.join(PROJECT_ROOT, "visual_bge")
        if not os.path.exists(visual_bge_path):
            logger.error(f"visual_bge目录不存在: {visual_bge_path}")
        elif not os.path.isdir(visual_bge_path):
            logger.error(f"visual_bge不是一个目录: {visual_bge_path}")
        else:
            # 检查__init__.py是否存在
            init_file = os.path.join(visual_bge_path, "__init__.py")
            if not os.path.exists(init_file):
                logger.error(f"visual_bge/__init__.py文件不存在: {init_file}")
            
            # 检查关键文件是否存在
            modeling_file = os.path.join(visual_bge_path, "modeling.py")
            if not os.path.exists(modeling_file):
                logger.error(f"visual_bge/modeling.py文件不存在: {modeling_file}")
        
        return False
    
    return True


def create_symlinks():
    """
    创建必要的符号链接，解决模块导入问题
    """
    logger.info("创建必要的符号链接...")
    
    # 从dnmsr_new/visual_bge到项目根目录
    src_dir = os.path.join(PROJECT_ROOT, "dnmsr_new", "visual_bge")
    dst_dir = os.path.join(PROJECT_ROOT, "visual_bge")
    
    if os.path.exists(src_dir) and os.path.isdir(src_dir) and not os.path.exists(dst_dir):
        try:
            os.symlink(src_dir, dst_dir)
            logger.info(f"创建符号链接: {src_dir} -> {dst_dir}")
        except Exception as e:
            logger.error(f"创建符号链接失败: {e}")
            
            # 如果无法创建符号链接，创建包含导入语句的__init__.py文件
            try:
                os.makedirs(dst_dir, exist_ok=True)
                with open(os.path.join(dst_dir, "__init__.py"), 'w') as f:
                    f.write(f"""# 自动生成的重定向模块
import sys
import os

# 重定向导入到实际模块位置
sys.path.insert(0, "{os.path.dirname(src_dir)}")
from dnmsr_new.visual_bge import *
""")
                logger.info(f"创建了重定向模块: {dst_dir}/__init__.py")
            except Exception as e:
                logger.error(f"创建重定向模块失败: {e}")


def main():
    """
    主函数
    """
    logger.info("=" * 50)
    logger.info("DNMSR模块导入环境设置工具")
    logger.info("=" * 50)
    
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"当前目录: {CURRENT_DIR}")
    
    # 1. 为所有相关目录创建__init__.py文件
    create_init_files()
    
    # 2. 检查模块结构
    check_module_structure()
    
    # 3. 创建必要的符号链接
    create_symlinks()
    
    # 4. 添加到Python路径
    add_to_python_path()
    
    # 5. 创建环境变量设置脚本
    create_env_file()
    
    # 6. 验证导入
    success = verify_imports()
    
    if success:
        logger.info("=" * 50)
        logger.info("设置成功！模块可以正确导入。")
        logger.info("=" * 50)
        return 0
    else:
        logger.error("=" * 50)
        logger.error("设置失败！模块无法正确导入。")
        logger.error("请按照上述错误信息修复问题，或手动调整Python路径。")
        logger.error("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 