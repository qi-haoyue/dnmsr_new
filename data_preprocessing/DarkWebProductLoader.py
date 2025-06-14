import torch
import torch.nn.functional as F
from PIL import Image
import os
import json
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
import argparse


# (模型加载代码，同上一方案，此处省略)
# ...
# from visual_bge.modeling import Visualized_BGE
# import open_clip
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dnmSR_model = ...
# eva_clip_model, eva_clip_preprocess, eva_clip_tokenizer = ...
# ...

class DarkWebProductLoader:
    def __init__(self, json_file_path, image_root_path, random_seed=42):
        """
        初始化暗网商品数据加载器
        
        Args:
            json_file_path: 包含商品信息的JSON文件路径
            image_root_path: 图像文件的根目录
            random_seed: 随机种子，用于抽样
            use_image_urls: 当本地找不到图像时，是否使用图像URL
        """
        self.json_file_path = json_file_path
        self.image_root_path = image_root_path

        # 设置随机种子，确保可重现性
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # 加载所有商品数据
        self.all_product_items = self._load_all_data()
        print(f"Successfully loaded {len(self.all_product_items)} total product items from JSON.")

        # 用于存储生成的候选文档
        self.candidate_documents = []

    def _load_all_data(self):
        """
        从JSON文件加载所有商品数据
        
        Returns:
            list: 包含所有商品信息的列表
        """
        print(f"Loading all product data from {self.json_file_path}...")
        processed_items = []
        
        # 用于统计图像路径情况
        image_stats = {
            "total_records": 0,
            "records_with_image_paths": 0,
            "records_with_valid_images": 0,
            "total_image_paths_found": 0,
            "valid_image_paths_found": 0
        }
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                raw_data_list = json.load(f)
        except Exception as e:
            print(f"Error reading or parsing JSON file {self.json_file_path}: {e}")
            return []

        # 预扫描图像目录，构建图片文件名到路径的映射，提高查找效率
        print("预扫描图像目录，构建索引...")
        filename_to_path_map = {}
        if os.path.exists(self.image_root_path):
            for root, dirs, files in os.walk(self.image_root_path):
                for filename in files:
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        # 存储原始文件名、清理后的文件名和无空格文件名的映射
                        clean_name = filename.strip()
                        no_space_name = filename.replace(" ", "")
                        filename_to_path_map[filename] = os.path.join(root, filename)
                        if clean_name != filename:
                            filename_to_path_map[clean_name] = os.path.join(root, filename)
                        if no_space_name != filename and no_space_name != clean_name:
                            filename_to_path_map[no_space_name] = os.path.join(root, filename)
        
        print(f"索引构建完成，共找到 {len(filename_to_path_map)} 个图像文件。")

        for idx, record in enumerate(tqdm(raw_data_list, desc="Processing raw product data")):
            image_stats["total_records"] += 1
            
            # 确保每个商品有唯一ID
            product_id = record.get("id")
            if product_id is None:
                product_id = f"item_{idx}_{random.randint(10000, 99999)}"
                
            # 获取标题和描述 - 使用数据集中的实际字段名
            title = record.get("title")  # 商品标题字段
            description = record.get("intro", "")  # 商品描述字段
            
            # 也可以使用name作为备用标题
            if not title and record.get("name"):
                title = record.get("name")

            if not title:  # 标题是必须的
                print(f"Warning: Record at index {idx} skipped due to missing 'title'.")
                continue

            # 处理图像路径，仅处理image_path字段
            image_paths = []
            
            # 处理image_path字段
            img_paths = record.get("image_path", [])
            if isinstance(img_paths, list) and img_paths:
                image_stats["records_with_image_paths"] += 1
                for rel_path in img_paths:
                    if not isinstance(rel_path, str) or not rel_path.strip():
                        continue
                    
                    # 1. 首先尝试通过文件名直接在索引中查找
                    basename = os.path.basename(rel_path).strip()
                    if basename in filename_to_path_map:
                        image_paths.append(filename_to_path_map[basename])
                        image_stats["total_image_paths_found"] += 1
                        continue
                    
                    # 2. 如果是"full/数据资源/目录名/文件名"格式，提取各部分并尝试构建路径
                    parts = rel_path.split('/')
                    parts = [p.strip() for p in parts if p.strip()]
                    
                    if len(parts) >= 3 and parts[0].lower() == "full":
                        category = parts[1]
                        if len(parts) >= 4:  # full/类别/目录/文件名
                            directory = parts[2]
                            filename = parts[-1].strip()
                            # 尝试构建完整路径
                            potential_path = os.path.join(self.image_root_path, category, directory, filename)
                            if os.path.exists(potential_path):
                                image_paths.append(potential_path)
                                image_stats["total_image_paths_found"] += 1
                                continue
                        elif len(parts) == 3:  # full/类别/文件名
                            filename = parts[-1].strip()
                            # 尝试构建完整路径
                            potential_path = os.path.join(self.image_root_path, category, filename)
                            if os.path.exists(potential_path):
                                image_paths.append(potential_path)
                                image_stats["total_image_paths_found"] += 1
                                continue
                    
                    # 3. 否则，清理路径并添加到待处理列表
                    clean_path = rel_path.replace(" ", "")
                    
                    # 尝试使用辅助函数查找文件
                    clean_basename = os.path.basename(clean_path).strip()
                    # 在索引中查找
                    if clean_basename in filename_to_path_map:
                        image_paths.append(filename_to_path_map[clean_basename])
                    else:
                        # 尝试在图像根目录下递归查找
                        for root, dirs, files in os.walk(self.image_root_path):
                            if clean_basename in files:
                                image_paths.append(os.path.join(root, clean_basename))
                                break
                            # 尝试去除空格后的文件名
                            no_space_basename = clean_basename.replace(" ", "")
                            if no_space_basename != clean_basename and no_space_basename in files:
                                image_paths.append(os.path.join(root, no_space_basename))
                                break
                    
                    image_stats["total_image_paths_found"] += 1
            
            # 去除重复路径
            image_paths = list(set(image_paths))
            
            if image_paths:
                image_stats["records_with_valid_images"] += 1
                image_stats["valid_image_paths_found"] += len(image_paths)
                
            # 构建商品信息结构
            processed_items.append({
                "product_id": product_id,
                "query_title": title,
                "text_description": description,
                "image_paths": image_paths
            })
        
        # 打印图像统计信息
        print("\nImage Path Statistics:")
        print(f"  Total records: {image_stats['total_records']}")
        print(f"  Records with image_path field: {image_stats['records_with_image_paths']}")
        print(f"  Total image paths found: {image_stats['total_image_paths_found']}")
        print(f"  Valid image paths found: {image_stats['valid_image_paths_found']}")
        print(f"  Records with valid images: {image_stats['records_with_valid_images']}")
        
        return processed_items

    def get_sampled_products(self, sample_size=100):
        """
        随机抽样指定数量的商品记录
        
        Args:
            sample_size: 要抽样的商品数量
            
        Returns:
            list: 抽样的商品列表
        """
        if not self.all_product_items:
            print("No items loaded to sample from.")
            return []

        if len(self.all_product_items) < sample_size:
            print(f"Warning: Total items ({len(self.all_product_items)}) is less than requested sample_size ({sample_size}). Using all items.")
            return self.all_product_items
        else:
            return random.sample(self.all_product_items, sample_size)

    def generate_multimodal_candidates(self, products=None):
        """
        为指定的商品生成多模态候选文档
        
        Args:
            products: 要处理的商品列表，如果为None则处理所有商品
            
        Returns:
            dict: 包含所有生成的候选文档的字典，按模态类型分组
        """
        if products is None:
            products = self.all_product_items
            
        candidates = {
            "text_only": [],  # 仅文本候选
            "image_only": [],  # 仅图像候选
            "multimodal": []   # 多模态候选（图像+文本）
        }
        
        print(f"Generating multimodal candidates for {len(products)} products...")
        
        # 跟踪有效图像数量
        total_valid_images = 0
        processed_ids = set()  # 跟踪已处理的产品ID
        
        for product in tqdm(products, desc="Generating candidates"):
            product_id = product["product_id"]
            
            # 跟踪已处理的产品ID，避免重复处理
            if product_id in processed_ids:
                print(f"Warning: Duplicate product ID found: {product_id}. Skipping...")
                continue
                
            processed_ids.add(product_id)
            
            description = product["text_description"]
            image_paths = product["image_paths"]
            
            # 1. 生成仅文本候选（无论有无图片，都生成）
            if description.strip():  # 确保描述不为空
                candidates["text_only"].append({
                    "product_id": product_id,
                    "content": description,
                    "modality": "text",
                    "source_type": "description"
                })
            
            # 2. 如果有图片，生成仅图像候选和多模态候选
            if image_paths:
                total_valid_images += len(image_paths)
                for img_idx, img_path in enumerate(image_paths):
                    # 仅图像候选
                    candidates["image_only"].append({
                        "product_id": product_id,
                        "content": img_path,
                        "modality": "image",
                        "source_type": f"image_{img_idx}"
                    })
                    
                    # 多模态候选（图像+文本）
                    if description.strip():
                        candidates["multimodal"].append({
                            "product_id": product_id,
                            "content": {
                                "image": img_path,
                                "text": description
                            },
                            "modality": "multimodal",
                            "source_type": f"image_{img_idx}_plus_text"
                        })
        
        # 存储生成的候选文档
        self.candidate_documents = candidates
        
        # 打印候选文档统计信息
        total_candidates = sum(len(cands) for cands in candidates.values())
        print(f"Generated {total_candidates} total candidate documents for {len(processed_ids)} unique products:")
        for modality, cands in candidates.items():
            unique_products = len(set(cand["product_id"] for cand in cands if isinstance(cand, dict) and "product_id" in cand))
            print(f"  - {modality}: {len(cands)} candidates from {unique_products} products")
            
        # 如果没有有效图像，输出明确的警告
        if total_valid_images == 0:
            print("\n警告: 未找到有效图像文件。可能的原因:")
            print("  1. 图像文件未正确解压或不存在")
            print("  2. 图像路径与JSON中记录的路径不匹配")
            print("  3. 图像存储在不同的位置或使用了不同的命名方式")
            print("\n建议:")
            print("  - 检查图像文件是否已正确解压")
            print("  - 检查实际图像路径与JSON中记录的路径是否一致")
            print("  - 如需继续评估，可以考虑使用文本模态进行评估")
            
        return candidates
    
    def get_ground_truth_mapping(self, query_products):
        """
        为查询商品生成真实答案映射
        
        Args:
            query_products: 作为查询的商品列表
            
        Returns:
            dict: 查询ID到相关候选文档ID的映射
        """
        if not self.candidate_documents:
            print("No candidate documents generated. Please call generate_multimodal_candidates() first.")
            return {}
            
        # 构建查询ID到候选文档的映射
        query_to_relevant_docs = {}
        
        # 从候选文档中提取所有产品ID
        candidate_product_ids = set()
        for modality in self.candidate_documents.keys():
            for candidate in self.candidate_documents[modality]:
                if isinstance(candidate, dict) and "product_id" in candidate:
                    candidate_product_ids.add(candidate["product_id"])
        
        # 处理每个查询产品
        for product in query_products:
            product_id = product["product_id"]
            
            # 验证该产品是否存在候选文档
            if product_id not in candidate_product_ids:
                print(f"Warning: No candidate documents found for product ID {product_id}")
                continue
                
            # 收集该商品生成的所有候选文档
            relevant_docs = []
            for modality in self.candidate_documents.keys():
                for candidate in self.candidate_documents[modality]:
                    if isinstance(candidate, dict) and "product_id" in candidate and candidate["product_id"] == product_id:
                        relevant_docs.append(candidate)
            
            # 如果找到相关候选，将标题作为查询，相关候选作为真实答案
            if relevant_docs:
                query_to_relevant_docs[product_id] = {
                    "query": product["query_title"],
                    "relevant_docs": relevant_docs
                }
        
        # 打印统计信息
        print(f"生成了 {len(query_to_relevant_docs)} 个查询的真实答案映射")
        if len(query_to_relevant_docs) < len(query_products):
            print(f"警告: {len(query_products) - len(query_to_relevant_docs)} 个查询产品没有对应的候选文档")
            
        return query_to_relevant_docs

    def save_samples(self, products, output_file):
        """
        将采样的商品保存到文件
        
        Args:
            products: 要保存的商品列表
            output_file: 输出文件路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(products, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved {len(products)} sampled products to {output_file}")
        except Exception as e:
            print(f"Error saving samples to {output_file}: {e}")
            
    def save_candidate_documents(self, output_file):
        """
        将生成的候选文档保存到文件
        
        Args:
            output_file: 输出文件路径
        """
        if not self.candidate_documents:
            print("No candidate documents to save. Please call generate_multimodal_candidates() first.")
            return
            
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.candidate_documents, f, ensure_ascii=False, indent=2)
            
            # 计算各模态候选文档数量
            text_cands = len(self.candidate_documents.get("text_only", []))
            img_cands = len(self.candidate_documents.get("image_only", []))
            mm_cands = len(self.candidate_documents.get("multimodal", []))
            total_cands = text_cands + img_cands + mm_cands
            
            print(f"Successfully saved {total_cands} candidate documents to {output_file}:")
            print(f"  - Text only: {text_cands}")
            print(f"  - Image only: {img_cands}")
            print(f"  - Multimodal: {mm_cands}")
        except Exception as e:
            print(f"Error saving candidate documents to {output_file}: {e}")

    def _debug_image_paths(self):
        """调试图像路径，打印出图像目录的结构"""
        print(f"\nDebug: Exploring image directory structure: {self.image_root_path}")
        
        if not os.path.exists(self.image_root_path):
            print(f"Error: Image root path does not exist: {self.image_root_path}")
            return
            
        # 检查直接子目录
        subdirs = [d for d in os.listdir(self.image_root_path) 
                  if os.path.isdir(os.path.join(self.image_root_path, d))]
        print(f"Found {len(subdirs)} subdirectories: {subdirs}")
        
        # 检查特定的'full'目录
        full_dir = os.path.join(self.image_root_path, "full")
        if os.path.exists(full_dir) and os.path.isdir(full_dir):
            full_subdirs = [d for d in os.listdir(full_dir) 
                           if os.path.isdir(os.path.join(full_dir, d))]
            print(f"Found {len(full_subdirs)} subdirectories in 'full': {full_subdirs[:10]}...")
            
            # 随机抽取一些子目录并列出内容
            if full_subdirs:
                sample_dirs = random.sample(full_subdirs, min(3, len(full_subdirs)))
                for sample_dir in sample_dirs:
                    sample_path = os.path.join(full_dir, sample_dir)
                    files = os.listdir(sample_path)[:10]  # 只取前10个文件
                    print(f"Sample of files in {sample_path}: {files}")
        else:
            print("No 'full' subdirectory found")
            
        # 检查图像文件类型
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        image_files = []
        
        # 在根目录中查找图像
        for ext in image_extensions:
            root_images = [f for f in os.listdir(self.image_root_path) 
                          if f.lower().endswith(ext) and os.path.isfile(os.path.join(self.image_root_path, f))]
            image_files.extend(root_images)
            
        print(f"Found {len(image_files)} image files directly in root directory. Sample: {image_files[:5]}")
        
    def _find_image_recursively(self, base_dir, target_filename, max_depth=4):
        """
        递归查找图像文件
        
        Args:
            base_dir: 开始搜索的目录
            target_filename: 要查找的文件名
            max_depth: 最大递归深度
            
        Returns:
            找到的文件路径，如果未找到则返回None
        """
        if max_depth <= 0:
            return None
            
        if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
            return None
            
        # 首先检查当前目录
        target_path = os.path.join(base_dir, target_filename)
        if os.path.exists(target_path) and os.path.isfile(target_path):
            return target_path
            
        # 然后递归检查子目录
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                found_path = self._find_image_recursively(item_path, target_filename, max_depth - 1)
                if found_path:
                    return found_path
                    
        return None
        
    def _parse_image_path(self, rel_path):
        """
        解析图像路径，处理特殊格式和转义字符
        
        Args:
            rel_path: 相对路径字符串
            
        Returns:
            tuple: (清理后的路径, 基本文件名, 类别名称, 目录名称)
        """
        # 特殊情况处理：如果路径中有文件名前的空格
        # 例如 "full/数据资源/Weee！支付数据库/ 00b827ade9cba8693d920ad8b5d4760d3ea7fb82.jpg"
        if " /" in rel_path or "/ " in rel_path:
            rel_path = rel_path.replace(" /", "/").replace("/ ", "/")
        
        # 处理路径末尾可能有文件名前的空格
        parts = rel_path.split("/")
        if len(parts) > 1 and parts[-1].startswith(" "):
            parts[-1] = parts[-1].strip()
            rel_path = "/".join(parts)
        
        # 清理路径中的转义字符和多余空格
        clean_path = rel_path.replace('\\r\\n', '').replace('\\n', '').strip()
        
        # 处理路径中的连续空格和空格前后的斜杠问题
        clean_path = clean_path.replace(' / ', '/').replace('/ ', '/').replace(' /', '/')
        
        # 确保没有多余的斜杠
        while '//' in clean_path:
            clean_path = clean_path.replace('//', '/')
        
        # 路径映射：在ca.json中可能使用数字代替类别名称，例如"full/1/..."
        # 数字到类别名称的映射
        category_mapping = {
            "1": "数据资源",
            "2": "影视音像",
            "3": "虚拟物品",
            "4": "卡料CVV",
            "5": "技术技能",
            "6": "其他类别",
            "7": "数据业务",
            "8": "私人专拍",
            "9": "实体物品"
        }
        
        # 分割路径以提取各部分
        parts = clean_path.split('/')
        parts = [p.strip() for p in parts if p.strip()]  # 移除空部分和前后空格
        
        # 提取基本文件名（处理文件名前的空格）
        basename = parts[-1].strip() if parts else ""
        
        # 初始化类别和目录名称
        category_name = None
        directory_name = None
        
        # 检查是否符合 full/[类别]/[目录]/[文件名] 格式
        if len(parts) >= 3 and parts[0].lower() == "full":
            # 获取类别名称（考虑可能的映射）
            category_id = parts[1]
            category_name = category_mapping.get(category_id, category_id)  # 如果是数字，则映射；否则使用原值
            
            # 获取目录名称（如果存在）
            if len(parts) >= 4:
                # 目录名称可能包含多个部分，需要合并
                dir_parts = parts[2:-1]
                if dir_parts:
                    directory_name = '/'.join(dir_parts)
            elif len(parts) == 3:
                # 如果只有三部分（full/类别/文件名），则没有目录
                directory_name = None
        
        return clean_path, basename, category_name, directory_name

    def explore_image_directory(self, max_depth=2):
        """
        探索图像目录结构，用于诊断图像查找问题
        
        Args:
            max_depth: 最大递归深度
            
        Returns:
            dict: 目录结构信息
        """
        def explore_dir(current_dir, depth=0):
            if depth > max_depth:
                return {"type": "dir", "name": os.path.basename(current_dir), "note": "达到最大深度"}
            
            try:
                items = os.listdir(current_dir)
                dirs = []
                files = []
                
                for item in items:
                    item_path = os.path.join(current_dir, item)
                    if os.path.isdir(item_path):
                        if depth < max_depth:
                            dirs.append(explore_dir(item_path, depth + 1))
                        else:
                            dirs.append({"type": "dir", "name": item})
                    elif os.path.isfile(item_path):
                        # 只记录前5个文件和文件总数
                        if len(files) < 5:
                            files.append({"type": "file", "name": item})
                
                return {
                    "type": "dir", 
                    "name": os.path.basename(current_dir),
                    "total_dirs": len([i for i in items if os.path.isdir(os.path.join(current_dir, i))]),
                    "total_files": len([i for i in items if os.path.isfile(os.path.join(current_dir, i))]),
                    "dirs": dirs,
                    "files_sample": files,
                }
            except Exception as e:
                return {"type": "error", "path": current_dir, "error": str(e)}
        
        print(f"\n探索图像目录结构 (最大深度: {max_depth}):")
        
        if not os.path.exists(self.image_root_path):
            print(f"错误: 图像根目录不存在: {self.image_root_path}")
            return None
            
        result = explore_dir(self.image_root_path)
        
        # 打印摘要信息
        self._print_dir_summary(result)
        
        return result
        
    def _print_dir_summary(self, dir_info, indent=0):
        """打印目录摘要信息"""
        if dir_info["type"] == "error":
            print(f"{' ' * indent}错误 ({dir_info['path']}): {dir_info['error']}")
            return
            
        files_str = f", {dir_info.get('total_files', 0)} 个文件" if 'total_files' in dir_info else ""
        print(f"{' ' * indent}{dir_info['name']} ({dir_info.get('total_dirs', 0)} 个子目录{files_str})")
        
        # 打印文件样本
        if 'files_sample' in dir_info and dir_info['files_sample']:
            file_names = [f["name"] for f in dir_info['files_sample']]
            if file_names:
                print(f"{' ' * (indent+2)}文件样本: {', '.join(file_names)}" + 
                      ("..." if dir_info.get('total_files', 0) > len(file_names) else ""))
        
        # 递归打印子目录
        if 'dirs' in dir_info:
            for sub_dir in dir_info['dirs']:
                self._print_dir_summary(sub_dir, indent + 2)

    def _find_file_variations(self, directory, filename):
        """
        在指定目录中查找文件的各种变体形式
        
        Args:
            directory: 要搜索的目录路径
            filename: 要查找的文件名
            
        Returns:
            找到的文件路径，如果未找到则返回None
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            return None
            
        # 生成可能的文件名变体
        variations = [
            filename,                  # 原始文件名
            filename.strip(),          # 去除前后空格
            filename.replace(" ", "")  # 移除所有空格
        ]
        
        # 检查文件是否存在
        for var in variations:
            file_path = os.path.join(directory, var)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                return file_path
                
# 示例用法
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="暗网商品数据加载与处理工具")
    parser.add_argument("--json_file", required=True, help="包含商品信息的JSON文件路径")
    parser.add_argument("--image_dir", required=True, help="图像文件的根目录")
    parser.add_argument("--samples_output", required=True, help="采样商品输出文件路径")
    parser.add_argument("--candidates_output", required=True, help="候选文档输出文件路径")
    parser.add_argument("--sample_size", type=int, default=100, help="要抽样的商品数量")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.samples_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.candidates_output), exist_ok=True)


    print("\n=== 初始化暗网商品数据加载器 ===")
    print(f"输出路径{args.samples_output} 和 {args.candidates_output} 已创建")
    waiting_for_user = input("请确保图像目录已准备好，按回车继续...")


    # 初始化数据加载器
    loader = DarkWebProductLoader(
        json_file_path=args.json_file,
        image_root_path=args.image_dir,
        random_seed=42
    )
    
    # 探索图像目录结构
    print("\n=== 探索图像目录结构 ===")
    loader.explore_image_directory(max_depth=3)
    
    # 随机抽样商品
    sampled_products = loader.get_sampled_products(sample_size=args.sample_size)
    print(f"\n=== 抽样了 {len(sampled_products)} 个商品进行评估 ===")
    
    # 确认所有采样商品有唯一ID
    product_ids = [p["product_id"] for p in sampled_products]
    unique_ids = set(product_ids)
    if len(unique_ids) < len(product_ids):
        print(f"警告: 采样中存在 {len(product_ids) - len(unique_ids)} 个重复的产品ID")
        # 去除重复ID的商品
        unique_products = []
        seen_ids = set()
        for product in sampled_products:
            if product["product_id"] not in seen_ids:
                unique_products.append(product)
                seen_ids.add(product["product_id"])
        print(f"已去除重复项，剩余 {len(unique_products)} 个唯一商品")
        sampled_products = unique_products
    
    # 保存采样的商品
    loader.save_samples(sampled_products, args.samples_output)
    print(f"保存了 {len(sampled_products)} 个采样商品到 {args.samples_output}")
    
    # 为采样的商品生成多模态候选文档
    print(f"\n=== 为 {len(sampled_products)} 个采样商品生成多模态候选文档 ===")
    loader.candidate_documents = {"text_only": [], "image_only": [], "multimodal": []}
    all_candidates = loader.generate_multimodal_candidates(products=sampled_products)
    
    # 验证生成的候选文档
    all_product_ids = set()
    for modality, candidates in all_candidates.items():
        modality_ids = set(c["product_id"] for c in candidates if isinstance(c, dict) and "product_id" in c)
        all_product_ids.update(modality_ids)
        non_sampled_ids = modality_ids - unique_ids
        if non_sampled_ids:
            print(f"错误: 在 {modality} 模态中发现 {len(non_sampled_ids)} 个不在采样中的产品ID")
    
    missing_ids = unique_ids - all_product_ids
    if missing_ids:
        print(f"警告: {len(missing_ids)} 个采样商品未生成任何候选文档")
    
    # 保存生成的候选文档
    loader.save_candidate_documents(args.candidates_output)
    
    # 总结检索评估准备情况
    print("\n=== 检索评估准备就绪情况 ===")
    text_cands = len(all_candidates.get("text_only", []))
    img_cands = len(all_candidates.get("image_only", []))
    mm_cands = len(all_candidates.get("multimodal", []))
    
    print(f"  - 文本候选文档: {text_cands} 个 {'✓' if text_cands > 0 else '✗'}")
    print(f"  - 图像候选文档: {img_cands} 个 {'✓' if img_cands > 0 else '✗'}")
    print(f"  - 多模态候选文档: {mm_cands} 个 {'✓' if mm_cands > 0 else '✗'}")
    
    total_cands = text_cands + img_cands + mm_cands
    if total_cands > 0:
        print(f"\n总共生成了 {total_cands} 个候选文档，可以进行检索评估。")
        print(f"注意: 这些候选文档仅来自于 {len(all_product_ids)} 个采样商品，而不是全部 {len(loader.all_product_items)} 个商品。")
    else:
        print("\n错误: 没有生成任何候选文档，无法进行检索评估。")

if __name__ == "__main__":
    main()