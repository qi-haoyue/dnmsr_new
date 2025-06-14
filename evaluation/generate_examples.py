#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成检索示例分析

此脚本从现有的评估结果JSON文件中读取数据，为查询生成检索结果示例，
展示每个查询的前K个检索结果，包括相似度和模态信息。
"""

import os
import json
import numpy as np
import datetime
import argparse
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import random
import sys

def load_embeddings(embedding_dir, model_name, mode="multimodal"):
    """
    加载嵌入向量
    
    Args:
        embedding_dir: 嵌入向量目录
        model_name: 模型名称
        mode: 检索模式，可选值为 'multimodal', 'text_to_image', 'image_to_text'
        
    Returns:
        tuple: (查询嵌入，查询ID，候选文档嵌入，候选文档ID，候选文档模态)
    """
    print(f"Loading embeddings for {model_name}...")
    print(f"嵌入目录: {embedding_dir}")
    print(f"检索模式: {mode}")
    
    # 检查目录是否存在
    if not os.path.exists(embedding_dir):
        print(f"错误: 嵌入目录不存在: {embedding_dir}")
        return None, None, None, None, None
    
    # 列出目录内容，帮助调试
    print(f"目录内容: {os.listdir(embedding_dir)}")
    
    try:
        # 定义文件路径
        query_file = os.path.join(embedding_dir, "query_embeddings.npz")
        if not os.path.exists(query_file):
            print(f"Warning: Query embedding file not found: {query_file}")
            return None, None, None, None, None
            
        # 确定gallery文件路径
        if model_name == "dnmsr":
            gallery_file = os.path.join(embedding_dir, "dnmsr_gallery_embeddings.npz")
        elif model_name == "clip":
            gallery_file = os.path.join(embedding_dir, "clip_gallery_embeddings.npz")
        else:
            print(f"未知模型名称: {model_name}")
            return None, None, None, None, None
            
        # 检查gallery文件是否存在
        if not os.path.exists(gallery_file):
            print(f"Warning: Embedding files for {model_name} not found: {gallery_file}")
            return None, None, None, None, None
            
        # 加载查询嵌入
        print(f"加载查询嵌入: {query_file}")
        query_data = np.load(query_file, allow_pickle=True)
        
        # 列出npz文件中的键
        print(f"Query npz文件包含的键: {list(query_data.keys())}")
        
        # 根据模型提取嵌入向量
        if model_name == "dnmsr":
            if "dnmsr_embeddings" in query_data:
                query_embeddings = query_data["dnmsr_embeddings"]
            else:
                print(f"错误: 查询npz文件中没有dnmsr_embeddings键")
                return None, None, None, None, None
        elif model_name == "clip":
            if "clip_embeddings" in query_data:
                query_embeddings = query_data["clip_embeddings"]
            else:
                print(f"错误: 查询npz文件中没有clip_embeddings键")
                return None, None, None, None, None
        else:
            print(f"未知模型名称: {model_name}")
            return None, None, None, None, None
            
        # 获取查询产品ID
        if "product_ids" in query_data:
            query_product_ids = query_data["product_ids"]
        else:
            print(f"错误: 查询npz文件中没有product_ids键")
            return None, None, None, None, None
            
        # 获取查询模态信息（如果存在）
        query_modalities = None
        if "modalities" in query_data:
            query_modalities = query_data["modalities"]
            print(f"获取到查询模态信息: {len(set(query_modalities))}种模态")
        
        # 加载候选文档嵌入
        print(f"加载{model_name}候选文档嵌入: {gallery_file}")
        gallery_data = np.load(gallery_file, allow_pickle=True)
        
        # 列出npz文件中的键
        print(f"Gallery npz文件包含的键: {list(gallery_data.keys())}")
        
        # 获取嵌入向量和ID
        if "embeddings" in gallery_data:
            gallery_embeddings = gallery_data["embeddings"]
        else:
            print(f"错误: 候选文档npz文件中没有embeddings键")
            return None, None, None, None, None
            
        if "product_ids" in gallery_data:
            gallery_product_ids = gallery_data["product_ids"]
        else:
            print(f"错误: 候选文档npz文件中没有product_ids键")
            return None, None, None, None, None
            
        # 获取模态信息（如果有）
        gallery_modalities = gallery_data.get("modalities", None)
        
        # 根据检索模式过滤查询和候选文档
        if mode == "text_to_image" and query_modalities is not None and gallery_modalities is not None:
            # 文搜图模式：查询为文本，候选为图像
            print("应用文搜图模式过滤...")
            
            # 过滤查询为文本模态
            text_indices = np.array([i for i, mod in enumerate(query_modalities) if mod == 'text'])
            if len(text_indices) > 0:
                print(f"过滤查询：保留 {len(text_indices)}/{len(query_modalities)} 个文本查询")
                query_embeddings = query_embeddings[text_indices]
                query_product_ids = query_product_ids[text_indices]
                if query_modalities is not None:
                    query_modalities = query_modalities[text_indices]
            else:
                print("警告：未找到文本模态的查询")
                
            # 过滤候选仅为图像模态
            image_indices = np.array([i for i, mod in enumerate(gallery_modalities) if mod == 'image'])
            if len(image_indices) > 0:
                print(f"过滤候选：保留 {len(image_indices)}/{len(gallery_modalities)} 个图像候选")
                gallery_embeddings = gallery_embeddings[image_indices]
                gallery_product_ids = gallery_product_ids[image_indices]
                gallery_modalities = gallery_modalities[image_indices]
            else:
                print("警告：未找到图像模态的候选")
                
        elif mode == "image_to_text" and query_modalities is not None and gallery_modalities is not None:
            # 图搜文模式：查询为图像，候选为文本
            print("应用图搜文模式过滤...")
            
            # 过滤查询为图像模态
            image_indices = np.array([i for i, mod in enumerate(query_modalities) if mod == 'image'])
            if len(image_indices) > 0:
                print(f"过滤查询：保留 {len(image_indices)}/{len(query_modalities)} 个图像查询")
                query_embeddings = query_embeddings[image_indices]
                query_product_ids = query_product_ids[image_indices]
                if query_modalities is not None:
                    query_modalities = query_modalities[image_indices]
            else:
                print("警告：未找到图像模态的查询")
                
            # 过滤候选仅为文本模态
            text_indices = np.array([i for i, mod in enumerate(gallery_modalities) if mod == 'text'])
            if len(text_indices) > 0:
                print(f"过滤候选：保留 {len(text_indices)}/{len(gallery_modalities)} 个文本候选")
                gallery_embeddings = gallery_embeddings[text_indices]
                gallery_product_ids = gallery_product_ids[text_indices]
                gallery_modalities = gallery_modalities[text_indices]
            else:
                print("警告：未找到文本模态的候选")
        
        print(f"成功加载{model_name}的嵌入文件:")
        print(f"  查询嵌入形状: {query_embeddings.shape}")
        print(f"  查询ID数量: {len(query_product_ids)}")
        print(f"  候选文档嵌入形状: {gallery_embeddings.shape}")
        print(f"  候选文档ID数量: {len(gallery_product_ids)}")
        if gallery_modalities is not None:
            print(f"  候选文档模态数量: {len(gallery_modalities)}")
        
        return query_embeddings, query_product_ids, gallery_embeddings, gallery_product_ids, gallery_modalities
    except Exception as e:
        print(f"加载嵌入文件时出错: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def normalize_embeddings(embeddings):
    """
    标准化嵌入向量
    
    Args:
        embeddings: 嵌入向量数组
        
    Returns:
        np.ndarray: 标准化后的嵌入向量
    """
    # 处理形状不兼容的问题
    if len(embeddings.shape) > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    # 计算嵌入向量的范数
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # 防止除以零
    norms[norms == 0] = 1.0
    # 标准化
    return embeddings / norms

def compute_similarity_matrix(query_embeddings, gallery_embeddings):
    """
    计算查询和候选文档之间的相似度矩阵，按照visual_bge模型的实现方式
    
    Args:
        query_embeddings: 查询嵌入向量
        gallery_embeddings: 候选文档嵌入向量
        
    Returns:
        np.ndarray: 相似度矩阵
    """
    # 打印形状信息
    print(f"查询嵌入形状: {query_embeddings.shape}, 类型: {query_embeddings.dtype}")
    print(f"候选文档嵌入形状: {gallery_embeddings.shape}, 类型: {gallery_embeddings.dtype}")
    
    # 对于EVA-CLIP嵌入的特殊处理
    if len(gallery_embeddings.shape) > 3:  # 如果是形如(273, 1, 257, 1024)的EVA-CLIP嵌入
        print("检测到EVA-CLIP格式的嵌入，执行特殊处理...")
        # 先取第一个维度的嵌入，通常为图像嵌入
        gallery_embeddings = gallery_embeddings[:, 0, 0, :]
        print(f"EVA-CLIP嵌入处理后形状: {gallery_embeddings.shape}")
    
    # 处理一般情况的嵌入维度
    if len(query_embeddings.shape) > 2:
        query_embeddings = query_embeddings.reshape(query_embeddings.shape[0], -1)
        print(f"查询嵌入形状: {query_embeddings.shape}")
        
    if len(gallery_embeddings.shape) > 2:
        # 对于一般的三维嵌入，保留最后一个维度
        if gallery_embeddings.shape[-1] == query_embeddings.shape[-1]:
            gallery_embeddings = gallery_embeddings.reshape(gallery_embeddings.shape[0], -1, gallery_embeddings.shape[-1])
            # 取第一个token的嵌入
            gallery_embeddings = gallery_embeddings[:, 0, :]
        else:
            # 如果最后维度不匹配，则尝试展平
            gallery_embeddings = gallery_embeddings.reshape(gallery_embeddings.shape[0], -1)
        print(f"候选文档嵌入形状: {gallery_embeddings.shape}")
    
    # 确保维度匹配
    if query_embeddings.shape[-1] != gallery_embeddings.shape[-1]:
        min_dim = min(query_embeddings.shape[-1], gallery_embeddings.shape[-1])
        query_embeddings = query_embeddings[..., :min_dim]
        gallery_embeddings = gallery_embeddings[..., :min_dim]
        print(f"截断后 - 查询形状: {query_embeddings.shape}, 候选形状: {gallery_embeddings.shape}")
    
    # 对嵌入向量进行归一化
    # 在visual_bge模型中，归一化是在encode方法中完成的
    # 确保向量已经归一化
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    gallery_norms = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    
    # 检查是否需要归一化
    if np.any(np.abs(query_norms - 1.0) > 1e-5) or np.any(np.abs(gallery_norms - 1.0) > 1e-5):
        print("检测到向量未归一化，进行归一化处理...")
        # 避免除以零
        query_norms = np.maximum(query_norms, 1e-8)
        gallery_norms = np.maximum(gallery_norms, 1e-8)
        
        # 标准化向量
        query_embeddings = query_embeddings / query_norms
        gallery_embeddings = gallery_embeddings / gallery_norms
    else:
        print("向量已经归一化，无需再次处理")
    
    # 计算余弦相似度 - 与visual_bge.compute_similarity保持一致
    # 余弦相似度范围为[-1, 1]，但我们保持原始值，不再映射到[0, 1]
    similarity_matrix = np.matmul(query_embeddings, gallery_embeddings.T)
    
    # 将相似度值缩放并转换为整数，以保持与原检索示例文件格式一致
    similarity_matrix = np.round(similarity_matrix * 1000) / 1000
    
    # 输出相似度范围（调试用）
    min_sim = np.min(similarity_matrix)
    max_sim = np.max(similarity_matrix)
    print(f"相似度值范围: [{min_sim:.4f}, {max_sim:.4f}]")
    
    return similarity_matrix

def generate_retrieval_examples(results_file, embedding_dir, output_dir, num_examples=10, top_k=10, candidates_file=None, mode="multimodal"):
    """
    生成检索示例分析
    
    Args:
        results_file: 检索结果JSON文件路径
        embedding_dir: 嵌入向量目录
        output_dir: 输出目录
        num_examples: 示例数量
        top_k: 每个查询保留的前K个结果
        candidates_file: 候选文档JSON文件路径
        mode: 检索模式，可选值为 'multimodal'(多模态混合检索), 'text_to_image'(文搜图), 'image_to_text'(图搜文)
    """
    print(f"生成检索示例分析，模式: {mode}, 示例数量: {num_examples}, top_k: {top_k}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据模式设置文件名
    output_filename = "retrieval_examples.txt"
    if mode == "text_to_image":
        output_filename = "text_to_image_examples.txt"
    elif mode == "image_to_text":
        output_filename = "image_to_text_examples.txt"
    
    # 加载检索结果
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            print(f"加载检索结果文件: {results_file}")
            results_data = json.load(f)
            
        # 检查结果文件结构，支持新旧两种格式
        if "metadata" in results_data and "results" in results_data:
            # 新格式: {"metadata": {...}, "results": {...}}
            metadata = results_data.get("metadata", {})
            file_mode = metadata.get("mode", "multimodal")
            
            # 如果文件中的模式与指定的模式不一致，发出警告
            if file_mode != mode and file_mode != "unknown":
                print(f"警告: 检索结果文件的模式({file_mode})与指定的模式({mode})不一致")
                
            results = results_data.get("results", {})
        else:
            # 旧格式: 直接是结果对象 {"dnmsr": {...}, "clip": {...}}
            results = results_data
    except Exception as e:
        print(f"加载检索结果文件出错: {e}")
        return
    
    # 加载候选文档（如果提供）
    candidates_by_id = {}
    if candidates_file and os.path.exists(candidates_file):
        try:
            with open(candidates_file, 'r', encoding='utf-8') as f:
                print(f"加载候选文档文件: {candidates_file}")
                candidates_data = json.load(f)
                
            # 构建候选文档ID到内容的映射
            for modality, docs in candidates_data.items():
                for doc in docs:
                    if isinstance(doc, dict) and "product_id" in doc:
                        # 对于图像模态，只存储文件名而不是完整路径
                        if doc.get("modality") == "image" and isinstance(doc.get("content"), str):
                            img_path = doc.get("content")
                            doc["content"] = os.path.basename(img_path)
                            
                        candidates_by_id[doc["product_id"]] = doc
                        
            # 根据模式进行过滤
            if mode == "text_to_image":
                # 文搜图模式：只保留图像候选
                print(f"按模式过滤候选文档：保留图像候选用于文搜图模式")
                candidates_by_id = {k: v for k, v in candidates_by_id.items() if v.get("modality") == "image"}
            elif mode == "image_to_text":
                # 图搜文模式：只保留文本候选
                print(f"按模式过滤候选文档：保留文本候选用于图搜文模式")
                candidates_by_id = {k: v for k, v in candidates_by_id.items() if v.get("modality") == "text"}
                
            print(f"过滤后的候选文档数量: {len(candidates_by_id)}")
        except Exception as e:
            print(f"加载候选文档文件出错: {e}")
    
    # 加载查询和候选文档嵌入 - 用于重新计算相似度矩阵
    query_embeddings, query_ids, gallery_embeddings, gallery_ids, gallery_modalities = load_embeddings(embedding_dir, "dnmsr", mode)
    if query_embeddings is None:
        print("无法加载嵌入，将使用结果文件中的数据")
    
    # 提取DNMSR模型的结果
    dnmsr_results = results.get("dnmsr", {})
    
    # 相似度矩阵和相关性矩阵
    similarity_matrix = dnmsr_results.get("similarity_matrix", [])
    relevance_matrix = dnmsr_results.get("relevance_matrix", [])
    
    # 如果结果文件中没有矩阵数据，且能够加载嵌入，则自己计算
    if (len(similarity_matrix) == 0 or len(relevance_matrix) == 0) and query_embeddings is not None:
        print("从嵌入计算相似度矩阵...")
        # 标准化嵌入
        query_embeddings = normalize_embeddings(query_embeddings)
        gallery_embeddings = normalize_embeddings(gallery_embeddings)
        # 计算相似度矩阵
        similarity_matrix = compute_similarity_matrix(query_embeddings, gallery_embeddings).tolist()
        
        # 构建相关性矩阵
        relevance_matrix = [[1 if qid == gid else 0 for gid in gallery_ids] for qid in query_ids]
    
    # 检查是否有足够的数据
    if (similarity_matrix is None or len(similarity_matrix) == 0 or
        query_ids is None or len(query_ids) == 0 or
        gallery_ids is None or len(gallery_ids) == 0):
        print("错误: 数据不足，无法生成示例分析")
        return
    
    # 确保矩阵是列表类型
    if isinstance(similarity_matrix, np.ndarray):
        similarity_matrix = similarity_matrix.tolist()
    if isinstance(relevance_matrix, np.ndarray):
        relevance_matrix = relevance_matrix.tolist()
    
    # 根据模式过滤gallery_modalities，确保只包含正确的模态
    if gallery_modalities is not None and len(gallery_modalities) > 0:
        filtered_indices = []
        if mode == "text_to_image":
            # 文搜图模式：只保留图像模态的候选
            filtered_indices = [i for i, modality in enumerate(gallery_modalities) if modality == "image"]
            print(f"过滤后保留 {len(filtered_indices)}/{len(gallery_modalities)} 个图像模态候选")
        elif mode == "image_to_text":
            # 图搜文模式：只保留文本模态的候选
            filtered_indices = [i for i, modality in enumerate(gallery_modalities) if modality == "text"]
            print(f"过滤后保留 {len(filtered_indices)}/{len(gallery_modalities)} 个文本模态候选")
        
        if filtered_indices:
            # 过滤相似度矩阵、相关性矩阵和候选ID
            filtered_gallery_ids = [gallery_ids[i] for i in filtered_indices]
            filtered_gallery_modalities = [gallery_modalities[i] for i in filtered_indices]
            
            # 过滤相似度矩阵的每一行，只保留符合条件的列
            filtered_similarity_matrix = []
            for row in similarity_matrix:
                filtered_row = [row[i] for i in filtered_indices]
                filtered_similarity_matrix.append(filtered_row)
            
            # 过滤相关性矩阵
            if relevance_matrix:
                filtered_relevance_matrix = []
                for row in relevance_matrix:
                    filtered_row = [row[i] for i in filtered_indices]
                    filtered_relevance_matrix.append(filtered_row)
            else:
                filtered_relevance_matrix = None
            
            # 更新变量
            gallery_ids = filtered_gallery_ids
            gallery_modalities = filtered_gallery_modalities
            similarity_matrix = filtered_similarity_matrix
            relevance_matrix = filtered_relevance_matrix
            
            print(f"完成过滤，现在有 {len(gallery_ids)} 个符合模式的候选文档")
    
    # 随机抽取示例 (如果数量不足，则使用全部)
    num_queries = len(similarity_matrix)
    if num_examples >= num_queries:
        example_indices = list(range(num_queries))
        print(f"使用所有 {num_queries} 个查询作为示例")
    else:
        example_indices = random.sample(range(num_queries), num_examples)
        print(f"随机抽取了 {num_examples} 个查询作为示例")
    
    # 打开输出文件
    output_file = os.path.join(output_dir, output_filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入标题
        mode_name = {
            "multimodal": "多模态混合检索",
            "text_to_image": "文搜图检索",
            "image_to_text": "图搜文检索"
        }.get(mode, "未知模式")
        
        f.write(f"{mode_name} - 查询示例分析\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"嵌入目录: {embedding_dir}\n")
        f.write(f"结果文件: {results_file}\n")
        f.write(f"候选文档文件: {candidates_file if candidates_file else 'N/A'}\n")
        f.write(f"示例数量: {len(example_indices)}\n")
        f.write(f"每个查询显示前 {top_k} 个结果\n")
        f.write("=" * 80 + "\n\n")
        
        # 按相似度得分从高到低生成示例
        examples_with_scores = []
        
        for idx in example_indices:
            # 获取当前查询
            query_id = query_ids[idx]
            sim_scores = similarity_matrix[idx]
            rel_scores = relevance_matrix[idx] if relevance_matrix else None
            
            # 找到相关项的最大相似度得分，作为示例排序依据
            # 优先选择有相关项，且相关项得分高的查询
            relevant_idxs = [i for i, rel in enumerate(rel_scores) if rel > 0] if rel_scores is not None else []
            if relevant_idxs:
                max_relevant_score = max([sim_scores[i] for i in relevant_idxs])
            else:
                max_relevant_score = 0
                
            examples_with_scores.append((idx, max_relevant_score))
        
        # 按相关项得分从高到低排序示例
        examples_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 生成每个示例
        for example_idx, (query_idx, _) in enumerate(examples_with_scores):
            # 获取当前查询
            query_id = query_ids[query_idx]
            sim_scores = similarity_matrix[query_idx]
            rel_scores = relevance_matrix[query_idx] if relevance_matrix else None
            
            # 获取查询内容
            query_content = f"查询ID: {query_id}"
            query_modality = None
            if candidates_by_id and query_id in candidates_by_id:
                query_doc = candidates_by_id[query_id]
                query_modality = query_doc.get('modality', 'unknown')
                query_content += f" (模态: {query_modality})"
                
                # 根据模态显示查询内容
                if query_doc.get("modality") == "text":
                    query_content += f"\n查询文本: {query_doc.get('content', 'N/A')}"
                elif query_doc.get("modality") == "image":
                    query_content += f"\n查询图像: {query_doc.get('content', 'N/A')}"
                elif query_doc.get("modality") == "multimodal":
                    content = query_doc.get('content', {})
                    if isinstance(content, dict):
                        query_content += f"\n查询文本: {content.get('text', 'N/A')}"
                        query_content += f"\n查询图像: {os.path.basename(content.get('image', 'N/A'))}"
            
            # 写入查询信息
            f.write(f"示例 {example_idx + 1}: {query_content}\n")
            f.write("-" * 80 + "\n")
            
            # 为当前查询找到前K个结果
            top_indices = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)[:top_k]
            
            # 写入搜索结果
            f.write(f"搜索结果 (前 {top_k} 个):\n\n")
            
            for rank, idx in enumerate(top_indices):
                gallery_id = gallery_ids[idx]
                score = sim_scores[idx]
                is_relevant = rel_scores is not None and rel_scores[idx] > 0
                
                # 获取候选文档的模态
                gallery_modality = gallery_modalities[idx] if gallery_modalities is not None and idx < len(gallery_modalities) else "unknown"
                
                # 检查模态是否符合要求
                if (mode == "image_to_text" and gallery_modality != "text") or \
                   (mode == "text_to_image" and gallery_modality != "image"):
                    # 跳过不符合模式要求的候选文档
                    continue
                
                result_line = f"[{rank+1}] 得分: {score:.4f} - 候选ID: {gallery_id}"
                if is_relevant:
                    result_line += " ✓"  # 标记相关项
                
                # 添加模态信息
                result_line += f" (模态: {gallery_modality})"
                f.write(result_line + "\n")
                
                # 如果有候选文档信息，显示内容
                if candidates_by_id and gallery_id in candidates_by_id:
                    doc = candidates_by_id[gallery_id]
                    
                    # 根据模态显示不同内容
                    if doc.get("modality") == "text":
                        text_content = doc.get("content", "")
                        # 截断过长的文本
                        if len(text_content) > 200:
                            text_content = text_content[:197] + "..."
                        f.write(f"  文本: {text_content}\n\n")
                    elif doc.get("modality") == "image":
                        f.write(f"  图像: {doc.get('content', 'N/A')}\n\n")
                    elif doc.get("modality") == "multimodal":
                        content = doc.get('content', {})
                        if isinstance(content, dict):
                            text = content.get('text', '')
                            # 截断过长的文本
                            if len(text) > 200:
                                text = text[:197] + "..."
                            f.write(f"  文本: {text}\n")
                            f.write(f"  图像: {os.path.basename(content.get('image', 'N/A'))}\n\n")
                else:
                    f.write("\n")  # 确保每个结果之间有空行
            
            f.write("=" * 80 + "\n\n")
    
    print(f"检索示例分析已保存到: {output_file}")
    return output_file

def generate_image_to_text_examples(results_file, embedding_dir, output_dir, num_examples=10, top_k=10, candidates_file=None):
    """
    生成图搜文检索示例分析
    
    Args:
        results_file: 检索结果JSON文件路径
        embedding_dir: 嵌入向量目录
        output_dir: 输出目录
        num_examples: 示例数量
        top_k: 每个查询保留的前K个结果
        candidates_file: 候选文档JSON文件路径
    """
    # 调用通用函数生成示例，指定模式为图搜文
    return generate_retrieval_examples(
        results_file=results_file,
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        num_examples=num_examples,
        top_k=top_k,
        candidates_file=candidates_file,
        mode="image_to_text"
    )

def generate_text_to_image_examples(results_file, embedding_dir, output_dir, num_examples=10, top_k=10, candidates_file=None):
    """
    生成文搜图检索示例分析
    
    Args:
        results_file: 检索结果JSON文件路径
        embedding_dir: 嵌入向量目录
        output_dir: 输出目录
        num_examples: 示例数量
        top_k: 每个查询保留的前K个结果
        candidates_file: 候选文档JSON文件路径
    """
    # 调用通用函数生成示例，指定模式为文搜图
    return generate_retrieval_examples(
        results_file=results_file,
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        num_examples=num_examples,
        top_k=top_k,
        candidates_file=candidates_file,
        mode="text_to_image"
    )

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成检索示例分析")
    parser.add_argument("--results_file", required=True, help="检索结果JSON文件路径")
    parser.add_argument("--embedding_dir", required=True, help="嵌入向量目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--num_examples", type=int, default=10, help="示例数量")
    parser.add_argument("--top_k", type=int, default=10, help="每个查询保留的前K个结果")
    parser.add_argument("--candidates_file", help="候选文档JSON文件路径")
    parser.add_argument("--mode", default="multimodal", choices=["multimodal", "text_to_image", "image_to_text"], 
                       help="检索模式: multimodal(多模态混合检索), text_to_image(文搜图), image_to_text(图搜文)")
    
    args = parser.parse_args()
    
    # 根据指定的模式生成示例
    if args.mode == "text_to_image":
        generate_text_to_image_examples(
            results_file=args.results_file,
            embedding_dir=args.embedding_dir,
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            top_k=args.top_k,
            candidates_file=args.candidates_file
        )
    elif args.mode == "image_to_text":
        generate_image_to_text_examples(
            results_file=args.results_file,
            embedding_dir=args.embedding_dir,
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            top_k=args.top_k,
            candidates_file=args.candidates_file
        )
    else:  # 默认多模态混合检索
        generate_retrieval_examples(
            results_file=args.results_file,
            embedding_dir=args.embedding_dir,
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            top_k=args.top_k,
            candidates_file=args.candidates_file,
            mode="multimodal"
        )
    
    print("检索示例分析生成完成")

if __name__ == "__main__":
    main() 