## 修改计划

### 目标
修复 `image_plus_text_as_answer` 在 `different_files_visited=True` 时无法获取数据的问题。

### 修改内容
修改 `/home/project/UniDoc-Bench/src/qa_synthesize/2_qa_synthesize.py` 中的 `extract_chunks_from_json` 函数：

1. **修改 `else` 分支逻辑**（第170-205行）：
   - 对于 `image_plus_text_as_answer` 且 `different_files_visited=True` 的情况：
     - 分别收集所有 text chunks 和 image chunks（不按 document_id 分组）
     - 将它们组合成新的 chunk 组合
     - 确保每个组合中至少包含一个 text chunk 和一个 image chunk
     - 检查组合后的 chunks 数量是否在 1-5 范围内
   - 对于其他 answer_type，保持原有逻辑不变

2. **具体实现**：
   - 当 `answer_type == "image_plus_text_as_answer"` 时：
     - 将 filtered_chunks 分为 text_chunks 和 image_chunks
     - 从 text_chunks 和 image_chunks 中各取一个或多个 chunks 组合
     - 确保组合后的 chunks 数量在 1-5 范围内
     - 生成对应的 metadata 和 overlapped_items

### 预期效果
修改后，运行带有 `--different_files_visited` 参数的命令时，`image_plus_text_as_answer` 应该能够获取到数据。