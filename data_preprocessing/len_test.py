import json
from typing import List, Dict, Any, Set
from collections import Counter

def analyze_json_file(file_path: str) -> None:
    """
    分析JSON文件的结构和内容
    
    :param file_path: JSON文件路径
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 判断数据类型
        if isinstance(data, dict):
            print(f"JSON文件包含字典对象，共有 {len(data)} 个顶级键")
            # 分析每个键
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  - 键 '{key}' 包含列表，长度为 {len(value)}")
                    if value:
                        # 检查列表项类型
                        if isinstance(value[0], dict) and 'product_id' in value[0]:
                            # 提取所有产品ID
                            product_ids = [item['product_id'] for item in value if isinstance(item, dict) and 'product_id' in item]
                            unique_ids = set(product_ids)
                            print(f"    - 包含 {len(unique_ids)} 个唯一产品ID (总共 {len(product_ids)} 个项)")
                            
                            # 检查是否有重复ID
                            if len(unique_ids) < len(product_ids):
                                id_counts = Counter(product_ids)
                                duplicate_ids = {id: count for id, count in id_counts.items() if count > 1}
                                print(f"    - 发现 {len(duplicate_ids)} 个重复ID，例如:")
                                for id, count in list(duplicate_ids.items())[:3]:
                                    print(f"      * ID '{id}' 出现 {count} 次")
                elif isinstance(value, dict):
                    print(f"  - 键 '{key}' 包含字典，共有 {len(value)} 个子键")
                else:
                    print(f"  - 键 '{key}' 包含类型为 {type(value).__name__} 的值")
        
        elif isinstance(data, list):
            print(f"JSON文件包含列表，长度为 {len(data)}")
            # 分析列表内容
            if data:
                if all(isinstance(item, dict) for item in data):
                    # 检查是否所有字典都有相同的键
                    all_keys = set()
                    for item in data:
                        all_keys.update(item.keys())
                    print(f"  - 列表包含字典项，共有 {len(all_keys)} 种不同的键")
                    print(f"  - 键集合: {', '.join(sorted(all_keys))}")
                    
                    # 如果有product_id，分析ID分布
                    if 'product_id' in all_keys:
                        product_ids = [item.get('product_id') for item in data if 'product_id' in item]
                        unique_ids = set(product_ids)
                        print(f"  - 包含 {len(unique_ids)} 个唯一产品ID (总共 {len(product_ids)} 个项)")
                else:
                    # 分析各种类型的分布
                    type_counts = Counter(type(item).__name__ for item in data)
                    print(f"  - 列表包含多种类型: {dict(type_counts)}")
        else:
            print(f"JSON文件包含单个 {type(data).__name__} 类型的值")
            
    except json.JSONDecodeError:
        print(f"错误: 无效的JSON格式")
    except FileNotFoundError:
        print(f"错误: 文件不存在: {file_path}")
    except Exception as e:
        print(f"错误: {str(e)}")

def compare_product_files(samples_file: str, candidates_file: str) -> None:
    """
    比较采样产品和候选文档文件，检查产品ID匹配情况
    
    :param samples_file: 采样产品文件路径
    :param candidates_file: 候选文档文件路径
    """
    try:
        # 读取采样产品文件
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        # 读取候选文档文件
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        
        # 提取采样产品ID
        if isinstance(samples, list):
            sample_ids = set(item.get('product_id') for item in samples if isinstance(item, dict) and 'product_id' in item)
            print(f"采样产品文件包含 {len(sample_ids)} 个唯一产品ID")
        else:
            print(f"错误: 采样产品文件应包含列表，但实际是 {type(samples).__name__}")
            return
        
        # 从候选文档中提取产品ID
        if isinstance(candidates, dict):
            all_candidate_ids = set()
            
            for key, items in candidates.items():
                if isinstance(items, list):
                    modality_ids = set(item.get('product_id') for item in items if isinstance(item, dict) and 'product_id' in item)
                    all_candidate_ids.update(modality_ids)
                    print(f"候选文档 '{key}' 模态包含 {len(modality_ids)} 个唯一产品ID")
                    
                    # 检查不在采样中的ID
                    non_sample_ids = modality_ids - sample_ids
                    if non_sample_ids:
                        print(f"  - 警告: 发现 {len(non_sample_ids)} 个不在采样中的产品ID")
                        print(f"  - 不在采样中的ID示例: {list(non_sample_ids)[:3]}")
            
            # 检查所有产品ID的匹配情况
            print(f"\n总计: 候选文档包含 {len(all_candidate_ids)} 个唯一产品ID")
            non_sample_ids = all_candidate_ids - sample_ids
            if non_sample_ids:
                print(f"警告: 发现 {len(non_sample_ids)} 个不在采样中的产品ID")
                print(f"不在采样中的ID示例: {list(non_sample_ids)[:3]}")
            
            missing_ids = sample_ids - all_candidate_ids
            if missing_ids:
                print(f"警告: 有 {len(missing_ids)} 个采样产品没有生成候选文档")
                print(f"缺失ID示例: {list(missing_ids)[:3]}")
        else:
            print(f"错误: 候选文档文件应包含字典，但实际是 {type(candidates).__name__}")
            
    except json.JSONDecodeError:
        print(f"错误: 无效的JSON格式")
    except FileNotFoundError as e:
        print(f"错误: 文件不存在: {e.filename}")
    except Exception as e:
        print(f"错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 默认文件路径
    samples_file = "./samples/sampled_products.json"
    candidates_file = "./samples/candidate_documents.json"
    
    # 远程服务器路径
    remote_samples_file = "/home/qhy/MML/dnmsr/dnmsr_new/data_preprocessing/samples/sampled_products.json"
    remote_candidates_file = "/home/qhy/MML/dnmsr/dnmsr_new/data_preprocessing/samples/candidate_documents.json"
    
    # 尝试两种路径
    for path_set in [
        (samples_file, candidates_file),
        (remote_samples_file, remote_candidates_file)
    ]:
        try:
            s_file, c_file = path_set
            print(f"\n===== 分析文件 =====")
            print(f"采样文件: {s_file}")
            print(f"候选文档: {c_file}")
            print("-" * 50)
            
            # 检查文件是否存在
            import os
            if not os.path.exists(s_file):
                print(f"采样文件不存在: {s_file}")
                continue
            if not os.path.exists(c_file):
                print(f"候选文档文件不存在: {c_file}")
                continue
                
            # 分析采样文件
            print("\n1. 分析采样产品文件:")
            analyze_json_file(s_file)
            
            # 分析候选文档文件
            print("\n2. 分析候选文档文件:")
            analyze_json_file(c_file)
            
            # 比较两个文件
            print("\n3. 比较产品ID匹配情况:")
            compare_product_files(s_file, c_file)
            
            # 找到一组有效路径，退出循环
            break
            
        except Exception as e:
            print(f"尝试路径集合时出错: {str(e)}")
            continue