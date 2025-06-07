import torch
import torch.nn.functional as F
from PIL import Image
import os
import json
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict


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
            "records_with_image_urls": 0,
            "records_with_pics": 0,
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
                
            # 处理图像路径，同时检查image_path和image_urls字段
            image_paths = []
            
            # 处理image_path字段
            img_paths = record.get("image_path", [])
            if isinstance(img_paths, list) and img_paths:
                image_stats["records_with_image_paths"] += 1
                for rel_path in img_paths:
                    if not isinstance(rel_path, str) or not rel_path.strip():
                        continue
                    # 简化路径，移除可能导致路径解析问题的空格
                    clean_path = rel_path.replace(" ", "")
                    image_paths.append(clean_path)
                    image_stats["total_image_paths_found"] += 1
            
            # 处理image_urls字段
            img_urls = record.get("image_urls", [])
            if isinstance(img_urls, list) and img_urls:
                image_stats["records_with_image_urls"] += 1
                for url in img_urls:
                    if not isinstance(url, str) or not url.strip():
                        continue
                    # 从URL中提取文件名
                    filename = os.path.basename(url).strip()
                    if filename:
                        image_paths.append(filename)
                        image_stats["total_image_paths_found"] += 1
            
            # 处理pics字段（如果存在）
            pics = record.get("pics", [])
            if isinstance(pics, list) and pics:
                image_stats["records_with_pics"] += 1
                for pic_obj in pics:
                    if isinstance(pic_obj, dict) and "pic" in pic_obj:
                        pic_path = pic_obj["pic"]
                        if pic_path and isinstance(pic_path, str):
                            image_paths.append(pic_path)
                            image_stats["total_image_paths_found"] += 1
            
            # 去除重复路径
            image_paths = list(set(image_paths))
            
            # 构建完整路径并验证文件存在性
            valid_full_image_paths = []
            for rel_path in image_paths:
                # 尝试多种可能的路径组合
                potential_paths = [
                    os.path.join(self.image_root_path, rel_path),
                    os.path.join(self.image_root_path, os.path.basename(rel_path)),
                    os.path.join(self.image_root_path, "full", rel_path),
                    os.path.join(self.image_root_path, "full", os.path.basename(rel_path))
                ]
                
                found = False
                for path in potential_paths:
                    if os.path.exists(path):
                        valid_full_image_paths.append(path)
                        image_stats["valid_image_paths_found"] += 1
                        found = True
                        break
                
                if not found and idx < 5:  # 只打印前5个找不到的图像，避免输出过多
                    print(f"Warning: Could not find image for path '{rel_path}'. Tried: {potential_paths}")
            
            if valid_full_image_paths:
                image_stats["records_with_valid_images"] += 1
                
            # 构建商品信息结构
            processed_items.append({
                "product_id": product_id,
                "query_title": title,
                "text_description": description,
                "image_paths": valid_full_image_paths
            })
        
        # 打印图像统计信息
        print("\nImage Path Statistics:")
        print(f"  Total records: {image_stats['total_records']}")
        print(f"  Records with image_path field: {image_stats['records_with_image_paths']}")
        print(f"  Records with image_urls field: {image_stats['records_with_image_urls']}")
        print(f"  Records with pics field: {image_stats['records_with_pics']}")
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
        
        for product in tqdm(products, desc="Generating candidates"):
            product_id = product["product_id"]
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
        print(f"Generated {total_candidates} total candidate documents:")
        for modality, cands in candidates.items():
            print(f"  - {modality}: {len(cands)} candidates")
            
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
        
        for product in query_products:
            product_id = product["product_id"]
            relevant_docs = []
            
            # 收集该商品生成的所有候选文档
            for modality in self.candidate_documents.keys():
                for candidate in self.candidate_documents[modality]:
                    if candidate["product_id"] == product_id:
                        relevant_docs.append(candidate)
            
            # 将标题作为查询，相关候选作为真实答案
            query_to_relevant_docs[product_id] = {
                "query": product["query_title"],
                "relevant_docs": relevant_docs
            }
            
        return query_to_relevant_docs

    def save_samples(self, products, output_file):
        """
        将采样的商品保存到文件
        
        Args:
            products: 要保存的商品列表
            output_file: 输出文件路径
        """
        try:
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
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.candidate_documents, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved candidate documents to {output_file}")
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

# 示例用法
def main():
    # 配置路径
    JSON_FILE_PATH = "/home/qhy/MML/DNM_dataset/changan/ca.json"
    IMAGE_ROOT_PATH = "/home/qhy/MML/DNM_dataset/changan/pic"
    
    # 输出文件路径
    SAMPLES_OUTPUT_FILE = "/home/qhy/MML/DNM_dataset/changan/sampled_products.json"
    CANDIDATES_OUTPUT_FILE = "/home/qhy/MML/DNM_dataset/changan/candidate_documents.json"
    
    # 初始化数据加载器
    loader = DarkWebProductLoader(
        json_file_path=JSON_FILE_PATH,
        image_root_path=IMAGE_ROOT_PATH,
        random_seed=42
    )
    
    # 随机抽样100个商品
    sampled_products = loader.get_sampled_products(sample_size=100)
    print(f"Sampled {len(sampled_products)} products for evaluation.")
    
    # 保存采样的商品
    loader.save_samples(sampled_products, SAMPLES_OUTPUT_FILE)
    
    # 为所有商品生成多模态候选文档
    all_candidates = loader.generate_multimodal_candidates()
    
    # 保存生成的候选文档
    loader.save_candidate_documents(CANDIDATES_OUTPUT_FILE)
    
    # 获取查询到真实答案的映射
    ground_truth_mapping = loader.get_ground_truth_mapping(sampled_products)
    
    print(f"Generated ground truth mapping for {len(ground_truth_mapping)} queries.")

if __name__ == "__main__":
    main()