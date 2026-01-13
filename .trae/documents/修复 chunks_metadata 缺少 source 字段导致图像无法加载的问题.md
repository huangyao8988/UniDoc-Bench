## 根本原因分析

经过深入分析，我发现了问题的**根本原因**：

### 核心问题：chunks_metadata 缺少 `source` 字段

在 [batch_retrieve_results_test.json](file:///home/project/UniDoc-Bench/batch_retrieve_results_test.json) 中，chunks 只包含 `document_keyword` 字段，**没有 `source` 字段**。

### 为什么这导致问题

在 [2_qa_synthesize.py](file:///home/project/UniDoc-Bench/src/qa_synthesize/2_qa_synthesize.py#L35-L38) 的 `chunk_match_back` 函数中：

```python
if 'source' not in chunks_metadata:
    if hasattr(args, 'debug') and args.debug:
        print(f"Warning: 'source' key not found in chunks_metadata, skipping image/table loading")
    return {}, {}

file_name = os.path.splitext(os.path.basename(chunks_metadata["source"]))[0].split("_id")[0]
elements_file_path = os.path.join(folder_elements, file_name + ".json")
```

由于 `chunks_metadata` 中没有 `source` 字段，函数会：
1. 打印警告信息（debug 模式下）
2. 直接返回 `{}, {}`
3. **无法加载任何图像或表格**

### 为什么这会影响特定类型

- **image_as_answer**：需要加载图像来生成 QA，但图像无法加载 → 跳过
- **image_plus_text_as_answer**：需要同时加载图像和文本，但图像无法加载 → 跳过
- **text_as_answer**：只需要文本，不依赖图像 → 可以正常生成
- **table_as_answer**：可能不依赖图像加载逻辑，或者有其他处理方式 → 可以正常生成

## 解决方案

### 方案1：修改 chunk_match_back 函数（推荐）

修改 [chunk_match_back](file:///home/project/UniDoc-Bench/src/qa_synthesize/2_qa_synthesize.py#L23-L46) 函数，使其能够使用 `document_keyword` 字段来构建 elements 文件路径：

```python
if 'source' not in chunks_metadata:
    if 'document_keyword' in chunks_metadata:
        document_keyword = chunks_metadata["document_keyword"]
        file_name = os.path.splitext(document_keyword)[0]
        elements_file_path = os.path.join(folder_elements, file_name + ".json")
        
        if not os.path.exists(elements_file_path):
            if hasattr(args, 'debug') and args.debug:
                print(f"Warning: Elements file not found: {elements_file_path}")
            return {}, {}
    else:
        if hasattr(args, 'debug') and args.debug:
            print(f"Warning: 'source' key not found in chunks_metadata, skipping image/table loading")
        return {}, {}
else:
    file_name = os.path.splitext(os.path.basename(chunks_metadata["source"]))[0].split("_id")[0]
    elements_file_path = os.path.join(folder_elements, file_name + ".json")
```

### 方案2：修改 load_chunks_from_json 函数

修改 [load_chunks_from_json](file:///home/project/UniDoc-Bench/src/qa_synthesize/2_qa_synthesize.py#L84-L135) 函数，确保 chunks_metadata 中包含 `source` 字段：

```python
chunk_info = {
    'content': content,
    'document_id': document_id,
    'query_question': query_question,
    'metadata': chunk,
    'source': chunk.get('document_keyword', '')  # 添加 source 字段
}
```

### 推荐执行步骤

1. **实施方案1**：修改 `chunk_match_back` 函数，使其能够使用 `document_keyword` 字段
2. **验证修复**：重新运行脚本，检查是否能够生成 `image_as_answer` 和 `image_plus_text_as_answer` 类型
3. **检查 elements 文件**：确保 `path/to/elements` 目录下存在对应的 JSON 文件（如 `186.json`、`028.json` 等）

## 预期效果

修改后：
- `image_as_answer` 类型应该能够正常生成
- `image_plus_text_as_answer` 类型应该能够正常生成
- 所有涉及图像加载的类型都能正常工作