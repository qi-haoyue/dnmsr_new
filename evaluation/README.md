# 暗网商品检索评估工具

本目录包含用于评估DNMSR和EVA-CLIP模型在暗网商品检索任务上性能的工具。

## 功能概述

1. **嵌入构建**：使用DNMSR和EVA-CLIP模型构建查询嵌入和候选文档库
2. **检索评估**：评估模型在检索任务上的性能，计算召回率、精确率和mAP等指标
3. **结果可视化**：生成对比图表，直观展示不同模型的性能差异

## 文件说明

- `embedding_builder.py`: 构建查询嵌入和候选文档库
- `retrieval_evaluator.py`: 评估检索性能并生成结果报告
- `run_evaluation.py`: 整合以上步骤的主运行脚本

## 使用方法

### 方法一：使用集成运行脚本

最简单的方法是使用`run_evaluation.py`脚本，它会自动处理嵌入构建和检索评估两个步骤：

```bash
python -m dnmsr.dnmsr_new.evaluation.run_evaluation \
    --dnmsr_model_path /path/to/Visualized_m3.pth \
    --samples_file /path/to/sampled_products.json \
    --candidates_file /path/to/candidate_documents.json \
    --output_dir ./evaluation_results
```

脚本会自动创建时间戳文件夹，将结果保存在指定的输出目录中。

#### 可选参数

- `--skip_embedding`: 跳过嵌入构建阶段，仅执行评估（需要之前已经生成了嵌入）
- `--skip_evaluation`: 跳过评估阶段，仅生成嵌入

### 方法二：分步执行

也可以分别执行嵌入构建和检索评估两个步骤：

#### 1. 构建嵌入

```bash
python -m dnmsr.dnmsr_new.evaluation.embedding_builder \
    --dnmsr_model_path /path/to/Visualized_m3.pth \
    --samples_file /path/to/sampled_products.json \
    --candidates_file /path/to/candidate_documents.json \
    --output_dir ./embeddings
```

#### 2. 评估检索性能

```bash
python -m dnmsr.dnmsr_new.evaluation.retrieval_evaluator \
    --embeddings_dir ./embeddings \
    --output_dir ./results
```

## 评估指标

评估脚本会计算以下指标：

1. **mAP (mean Average Precision)**: 平均精度均值，评估整体检索质量
2. **Recall@K**: K值不同时的召回率，即检索前K个结果中找到相关项的比例
3. **Precision@K**: K值不同时的精确率，即检索前K个结果中相关项的比例

## 输出结果

评估完成后，将生成以下结果：

1. **详细结果JSON文件**：包含每个查询的详细评估指标
2. **汇总报告CSV文件**：不同模型在各指标上的对比表格
3. **性能对比图表**：包括召回率对比图、精确率对比图和mAP对比图

## 注意事项

1. 确保已安装所有必要的依赖包，包括`torch`、`numpy`、`PIL`、`matplotlib`、`pandas`等
2. 模型文件、采样商品文件和候选文档文件必须存在且格式正确
3. 如果遇到图像文件路径问题，请检查候选文档中的图像路径是否正确

## 示例输出

成功执行评估后，终端将显示类似以下内容的结果摘要：

```
DNMSR检索评估完成
  mAP: 0.7846
  Recall@1: 0.5923
  Precision@1: 0.5923
  Recall@5: 0.8462
  Precision@5: 0.1692
  Recall@10: 0.9077
  Precision@10: 0.0908
  模态分布:
    text: 33.75%
    image: 48.12%
    multimodal: 18.13%

EVA-CLIP检索评估完成
  mAP: 0.6154
  Recall@1: 0.4615
  Precision@1: 0.4615
  Recall@5: 0.7385
  Precision@5: 0.1477
  Recall@10: 0.8308
  Precision@10: 0.0831
```

评估结果将保存在指定的输出目录中，包括详细的JSON结果文件、CSV汇总报告和可视化图表。 