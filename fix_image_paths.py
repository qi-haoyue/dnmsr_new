#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复JSON文件中的图像路径，将数字类别ID替换为对应的类别名称
"""

import json
import os
import re
import sys
from tqdm import tqdm

# 类别映射关系
class_map = {
    1: '数据资源',
    2: '服务业务',
    3: '虚拟物品',
    4: '私人专拍',
    5: '卡料CVV',
    6: '影视音像',
    7: '其他类别',
    8: '技术技能',
    9: '实体物品'
}

def fix_image_paths(json_file, output_file=None):
    """
    修复JSON文件中的图像路径，将数字类别替换为类别名称
    
    Args:
        json_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径，如果为None则在原文件名后添加_fixed
    """
    if output_file is None:
        # 如果未指定输出文件，生成默认名称
        base_name, ext = os.path.splitext(json_file)
        output_file = f"{base_name}_fixed{ext}"
    
    print(f"正在读取JSON文件: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return
    
    print(f"共读取{len(data)}条记录")
    
    # 记录修改统计
    stats = {
        "total_records": len(data),
        "records_with_image_paths": 0,
        "paths_modified": 0,
        "paths_with_category_id": 0
    }
    
    # 正则表达式用于匹配full/数字/路径格式
    path_pattern = re.compile(r'full/(\d+)/')
    
    # 处理每条记录
    for record in tqdm(data, desc="修复图像路径"):
        # 处理image_path字段
        if "image_path" in record and isinstance(record["image_path"], list):
            stats["records_with_image_paths"] += 1
            modified_paths = []
            
            for path in record["image_path"]:
                if not isinstance(path, str):
                    modified_paths.append(path)
                    continue
                    
                # 查找数字类别ID并替换
                match = path_pattern.search(path)
                if match:
                    category_id = int(match.group(1))
                    stats["paths_with_category_id"] += 1
                    
                    if category_id in class_map:
                        # 替换数字ID为类别名称
                        new_path = path.replace(f"full/{category_id}/", f"full/{class_map[category_id]}/")
                        modified_paths.append(new_path)
                        stats["paths_modified"] += 1
                    else:
                        modified_paths.append(path)
                else:
                    modified_paths.append(path)
            
            # 更新记录中的路径
            record["image_path"] = modified_paths
    
    # 保存修复后的JSON文件
    print(f"正在保存修复后的JSON文件: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"保存成功!")
    except Exception as e:
        print(f"保存失败: {e}")
        return
    
    # 打印统计信息
    print("\n修复统计:")
    print(f"总记录数: {stats['total_records']}")
    print(f"包含image_path字段的记录数: {stats['records_with_image_paths']}")
    print(f"包含类别ID的路径数: {stats['paths_with_category_id']}")
    print(f"已修改的路径数: {stats['paths_modified']}")

if __name__ == "__main__":
    # 默认输入文件
    input_file = "/home/qhy/MML/DNM_dataset/changan/ca.json"
    
    # 默认输出文件
    output_file = "/home/qhy/MML/DNM_dataset/changan/ca_fixed.json"
    
    # 如果命令行提供了参数，使用命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # 执行修复
    fix_image_paths(input_file, output_file) 