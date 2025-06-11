#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
嵌入构建器 - 用于构建查询嵌入和候选文档库

此脚本使用DNMSR（Visualized_BGE）和EVA-CLIP模型为暗网商品数据生成嵌入向量，
包括查询嵌入（基于标题）和候选文档嵌入（基于描述文本、图像或两者结合）。
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
from typing import Dict, List, Any, Optional, Union
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

try:
    # 导入Visualized_BGE模型
    from visual_bge.modeling import Visualized_BGE
    import open_clip
    logger.info("成功导入模型模块")
except ImportError as e:
    logger.error(f"导入模型模块失败: {e}")
    sys.exit(1)

# 定义设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")

class EmbeddingBuilder:
    """
    嵌入构建器 - 用于构建查询嵌入和候选文档库
    """
    def __init__(
        self,
        dnmsr_model_path: str,
        samples_file: str,
        candidates_file: str,
        output_dir: str = "./embeddings",
        device: torch.device = DEVICE
    ):
        """
        初始化嵌入构建器
        
        Args:
            dnmsr_model_path: DNMSR模型权重文件路径
            samples_file: 采样商品数据文件路径
            candidates_file: 候选文档数据文件路径
            output_dir: 输出目录
            device: 计算设备
        """
        self.dnmsr_model_path = dnmsr_model_path
        self.samples_file = samples_file
        self.candidates_file = candidates_file
        self.output_dir = output_dir
        self.device = device
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 模型相关
        self.dnmsr_model = None
        self.eva_clip_model = None
        self.eva_clip_preprocess = None
        self.eva_clip_tokenizer = None
        
        # 数据存储
        self.samples = []
        self.candidates = {}
        
        # 嵌入向量
        self.query_embeddings_dnmsr = []
        self.query_embeddings_clip = []
        self.query_product_ids = []
        
        self.gallery_embeddings_dnmsr = []
        self.gallery_product_ids_dnmsr = []
        self.gallery_modality_dnmsr = []
        
        self.gallery_embeddings_clip = []
        self.gallery_product_ids_clip = []
    
    def load_models(self):
        """加载DNMSR和EVA-CLIP模型"""
        logger.info("加载模型...")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.dnmsr_model_path):
            logger.error(f"DNMSR模型文件不存在: {self.dnmsr_model_path}")
            return False
        
        try:
            # 1. 加载DNMSR模型 (Visualized_BGE)
            logger.info(f"加载DNMSR模型: {self.dnmsr_model_path}")
            # 使用正确的参数初始化Visualized_BGE模型
            self.dnmsr_model = Visualized_BGE(
                model_name_bge="BAAI/bge-m3",  # 使用bge-m3模型
                model_weight=self.dnmsr_model_path
            )
            self.dnmsr_model.eval()  # 设置为评估模式
            logger.info("DNMSR模型加载成功")

            # 2. 准备EVA-CLIP模型 (用于独立图像嵌入)
            # 这里使用DNMSR模型内部的EVA-CLIP组件
            self.eva_clip_model = self.dnmsr_model.model_visual
            self.eva_clip_preprocess = self.dnmsr_model.preprocess_val
            self.eva_clip_tokenizer = self.dnmsr_model.tokenizer
            logger.info("EVA-CLIP组件准备完成")
            
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def load_data(self):
        """加载采样商品和候选文档数据"""
        try:
            # 加载采样商品数据
            logger.info(f"加载采样商品数据: {self.samples_file}")
            if not os.path.exists(self.samples_file):
                logger.error(f"采样商品文件不存在: {self.samples_file}")
                return False
                
            with open(self.samples_file, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)
            
            if not isinstance(self.samples, list):
                logger.error(f"采样商品数据格式错误，应为列表，实际是 {type(self.samples).__name__}")
                return False
            
            logger.info(f"加载了 {len(self.samples)} 个采样商品")
            
            # 加载候选文档数据
            logger.info(f"加载候选文档数据: {self.candidates_file}")
            if not os.path.exists(self.candidates_file):
                logger.error(f"候选文档文件不存在: {self.candidates_file}")
                return False
                
            with open(self.candidates_file, 'r', encoding='utf-8') as f:
                self.candidates = json.load(f)
            
            if not isinstance(self.candidates, dict):
                logger.error(f"候选文档数据格式错误，应为字典，实际是 {type(self.candidates).__name__}")
                return False
            
            # 检查候选文档的模态类型
            expected_modalities = ["text_only", "image_only", "multimodal"]
            for modality in expected_modalities:
                if modality not in self.candidates:
                    logger.warning(f"候选文档中缺少 {modality} 模态数据")
            
            # 统计候选文档数量
            total_candidates = sum(len(candidates) for modality, candidates in self.candidates.items() if isinstance(candidates, list))
            logger.info(f"加载了 {total_candidates} 个候选文档")
            
            return True
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return False
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return False
    
    def build_query_embeddings(self):
        """构建查询嵌入 (基于商品标题)"""
        logger.info("构建查询嵌入...")
        
        # 清空现有数据
        self.query_embeddings_dnmsr = []
        self.query_embeddings_clip = []
        self.query_product_ids = []
        
        with torch.no_grad():
            for sample in tqdm(self.samples, desc="构建查询嵌入"):
                product_id = sample.get("product_id")
                title = sample.get("query_title")
                
                if not product_id or not title:
                    logger.warning(f"样本缺少product_id或title字段，跳过")
                    continue
                
                # 1. 使用DNMSR模型生成标题嵌入
                # 通过encode函数直接获取文本嵌入
                dnmsr_embedding = self.dnmsr_model.encode(text=title)
                
                # 2. 使用EVA-CLIP模型生成标题嵌入
                # 使用CLIP文本编码器处理标题
                clip_text = self.eva_clip_tokenizer(title, return_tensors="pt", padding=True).to(self.device)
                clip_embedding = self.dnmsr_model.encode_text(clip_text)
                
                # 转换为NumPy数组
                dnmsr_embedding_np = dnmsr_embedding.cpu().numpy()
                clip_embedding_np = clip_embedding.cpu().numpy()
                
                # 存储嵌入和产品ID
                self.query_embeddings_dnmsr.append(dnmsr_embedding_np)
                self.query_embeddings_clip.append(clip_embedding_np)
                self.query_product_ids.append(product_id)
        
        logger.info(f"生成了 {len(self.query_product_ids)} 个查询嵌入")
    
    def build_dnmsr_gallery(self):
        """构建DNMSR候选文档库"""
        logger.info("构建DNMSR候选文档库...")
        
        # 清空现有数据
        self.gallery_embeddings_dnmsr = []
        self.gallery_product_ids_dnmsr = []
        self.gallery_modality_dnmsr = []
        
        # 1. 处理文本候选 (Text-Only)
        if "text_only" in self.candidates:
            text_candidates = self.candidates["text_only"]
            logger.info(f"处理 {len(text_candidates)} 个文本候选...")
            
            with torch.no_grad():
                for candidate in tqdm(text_candidates, desc="处理文本候选"):
                    product_id = candidate.get("product_id")
                    text = candidate.get("content")
                    
                    if not product_id or not text:
                        continue
                    
                    # 使用DNMSR模型encode函数生成文本嵌入
                    embedding = self.dnmsr_model.encode(text=text)
                    
                    # 转换为NumPy数组并存储
                    embedding_np = embedding.cpu().numpy()
                    self.gallery_embeddings_dnmsr.append(embedding_np)
                    self.gallery_product_ids_dnmsr.append(product_id)
                    self.gallery_modality_dnmsr.append("text")
        
        # 2. 处理图像候选 (Image-Only)
        if "image_only" in self.candidates:
            image_candidates = self.candidates["image_only"]
            logger.info(f"处理 {len(image_candidates)} 个图像候选...")
            
            with torch.no_grad():
                for candidate in tqdm(image_candidates, desc="处理图像候选"):
                    product_id = candidate.get("product_id")
                    image_path = candidate.get("content")
                    
                    if not product_id or not image_path:
                        continue
                    
                    # 检查图像是否存在
                    if not os.path.exists(image_path) or not os.path.isfile(image_path):
                        logger.warning(f"图像文件不存在: {image_path}")
                        continue
                    
                    try:
                        # 使用DNMSR模型encode函数生成图像嵌入
                        embedding = self.dnmsr_model.encode(image=image_path)
                        
                        # 转换为NumPy数组并存储
                        embedding_np = embedding.cpu().numpy()
                        self.gallery_embeddings_dnmsr.append(embedding_np)
                        self.gallery_product_ids_dnmsr.append(product_id)
                        self.gallery_modality_dnmsr.append("image")
                    except Exception as e:
                        logger.error(f"处理图像时出错: {image_path}, 错误: {e}")
        
        # 3. 处理多模态候选 (Image+Text)
        if "multimodal" in self.candidates:
            multimodal_candidates = self.candidates["multimodal"]
            logger.info(f"处理 {len(multimodal_candidates)} 个多模态候选...")
            
            with torch.no_grad():
                for candidate in tqdm(multimodal_candidates, desc="处理多模态候选"):
                    product_id = candidate.get("product_id")
                    content = candidate.get("content")
                    
                    if not product_id or not content or not isinstance(content, dict):
                        continue
                    
                    image_path = content.get("image")
                    text = content.get("text")
                    
                    if not image_path or not text:
                        continue
                    
                    # 检查图像是否存在
                    if not os.path.exists(image_path) or not os.path.isfile(image_path):
                        logger.warning(f"图像文件不存在: {image_path}")
                        continue
                    
                    try:
                        # 使用DNMSR模型encode函数生成多模态嵌入
                        embedding = self.dnmsr_model.encode(image=image_path, text=text)
                        
                        # 转换为NumPy数组并存储
                        embedding_np = embedding.cpu().numpy()
                        self.gallery_embeddings_dnmsr.append(embedding_np)
                        self.gallery_product_ids_dnmsr.append(product_id)
                        self.gallery_modality_dnmsr.append("multimodal")
                    except Exception as e:
                        logger.error(f"处理多模态数据时出错: {image_path}, 错误: {e}")
        
        logger.info(f"生成了 {len(self.gallery_embeddings_dnmsr)} 个DNMSR候选文档嵌入")
    
    def build_clip_gallery(self):
        """构建EVA-CLIP候选文档库 (仅图像)"""
        logger.info("构建EVA-CLIP候选文档库...")
        
        # 清空现有数据
        self.gallery_embeddings_clip = []
        self.gallery_product_ids_clip = []
        
        # 只处理图像候选
        if "image_only" in self.candidates:
            image_candidates = self.candidates["image_only"]
            logger.info(f"处理 {len(image_candidates)} 个EVA-CLIP图像候选...")
            
            with torch.no_grad():
                for candidate in tqdm(image_candidates, desc="处理EVA-CLIP图像候选"):
                    product_id = candidate.get("product_id")
                    image_path = candidate.get("content")
                    
                    if not product_id or not image_path:
                        continue
                    
                    # 检查图像是否存在
                    if not os.path.exists(image_path) or not os.path.isfile(image_path):
                        logger.warning(f"图像文件不存在: {image_path}")
                        continue
                    
                    try:
                        # 读取并预处理图像
                        image = Image.open(image_path).convert('RGB')
                        processed_image = self.eva_clip_preprocess(image).unsqueeze(0).to(self.device)
                        
                        # 使用EVA-CLIP模型生成图像嵌入
                        embedding = self.eva_clip_model.encode_image(processed_image, normalize=True)
                        
                        # 转换为NumPy数组并存储
                        embedding_np = embedding.cpu().numpy()
                        self.gallery_embeddings_clip.append(embedding_np)
                        self.gallery_product_ids_clip.append(product_id)
                    except Exception as e:
                        logger.error(f"处理EVA-CLIP图像时出错: {image_path}, 错误: {e}")
        
        logger.info(f"生成了 {len(self.gallery_embeddings_clip)} 个EVA-CLIP候选文档嵌入")
    
    def save_embeddings(self):
        """保存嵌入数据到文件"""
        logger.info(f"保存嵌入到目录: {self.output_dir}")
        
        # 1. 保存查询嵌入
        if self.query_embeddings_dnmsr and self.query_product_ids:
            query_data = {
                "dnmsr_embeddings": np.array(self.query_embeddings_dnmsr),
                "clip_embeddings": np.array(self.query_embeddings_clip) if self.query_embeddings_clip else None,
                "product_ids": np.array(self.query_product_ids)
            }
            query_file = os.path.join(self.output_dir, "query_embeddings.npz")
            np.savez_compressed(query_file, **query_data)
            logger.info(f"保存查询嵌入到: {query_file}")
        
        # 2. 保存DNMSR候选文档嵌入
        if self.gallery_embeddings_dnmsr and self.gallery_product_ids_dnmsr:
            dnmsr_gallery_data = {
                "embeddings": np.array(self.gallery_embeddings_dnmsr),
                "product_ids": np.array(self.gallery_product_ids_dnmsr),
                "modalities": np.array(self.gallery_modality_dnmsr)
            }
            dnmsr_gallery_file = os.path.join(self.output_dir, "dnmsr_gallery_embeddings.npz")
            np.savez_compressed(dnmsr_gallery_file, **dnmsr_gallery_data)
            logger.info(f"保存DNMSR候选文档嵌入到: {dnmsr_gallery_file}")
        
        # 3. 保存EVA-CLIP候选文档嵌入
        if self.gallery_embeddings_clip and self.gallery_product_ids_clip:
            clip_gallery_data = {
                "embeddings": np.array(self.gallery_embeddings_clip),
                "product_ids": np.array(self.gallery_product_ids_clip)
            }
            clip_gallery_file = os.path.join(self.output_dir, "clip_gallery_embeddings.npz")
            np.savez_compressed(clip_gallery_file, **clip_gallery_data)
            logger.info(f"保存EVA-CLIP候选文档嵌入到: {clip_gallery_file}")
    
    def run(self):
        """运行完整的嵌入构建流程"""
        # 1. 加载模型
        if not self.load_models():
            logger.error("模型加载失败，退出嵌入构建")
            return False
        
        # 2. 加载数据
        if not self.load_data():
            logger.error("数据加载失败，退出嵌入构建")
            return False
        
        # 3. 构建查询嵌入
        self.build_query_embeddings()
        
        # 4. 构建DNMSR候选文档库
        self.build_dnmsr_gallery()
        
        # 5. 构建EVA-CLIP候选文档库
        self.build_clip_gallery()
        
        # 6. 保存嵌入数据
        self.save_embeddings()
        
        logger.info("嵌入构建完成")
        return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="构建查询嵌入和候选文档库")
    parser.add_argument("--dnmsr_model_path", type=str, required=True, 
                      help="DNMSR模型权重文件路径 (Visualized_m3.pth)")
    parser.add_argument("--samples_file", type=str, required=True, 
                      help="采样商品数据文件路径")
    parser.add_argument("--candidates_file", type=str, required=True, 
                      help="候选文档数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./embeddings", 
                      help="输出目录")
    args = parser.parse_args()
    
    # 显示参数
    logger.info("参数配置:")
    logger.info(f"  DNMSR模型路径: {args.dnmsr_model_path}")
    logger.info(f"  采样商品文件: {args.samples_file}")
    logger.info(f"  候选文档文件: {args.candidates_file}")
    logger.info(f"  输出目录: {args.output_dir}")
    
    # 创建嵌入构建器并运行
    builder = EmbeddingBuilder(
        dnmsr_model_path=args.dnmsr_model_path,
        samples_file=args.samples_file,
        candidates_file=args.candidates_file,
        output_dir=args.output_dir
    )
    
    success = builder.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 