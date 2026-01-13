## 修复 `image_plus_text_as_answer` 类型缺失问题

### 问题分析
通过分析中间结果文件发现：
- `test_image_plus_text_as_answer.json` 成功生成了1个group（包含1个text + 1个image chunks）
- 但最终 `output.jsonl` 中没有 `image_plus_text_as_answer` 类型的QA

**根本原因**：问题出在QA生成阶段，`choose_templates`函数在处理混合类型chunks时可能失败，导致这些chunks被跳过。

### 解决方案

#### 方案1：添加详细的调试输出（推荐）
在 `2_qa_synthesize.py` 中添加调试输出，帮助诊断问题：
1. 在 `choose_templates` 函数中添加详细的错误日志
2. 在QA生成循环中记录每个answer_type的处理进度
3. 记录API调用的响应内容
4. 添加 `--debug` 参数支持

#### 方案2：改进模板选择逻辑
修改 `choose_templates` 函数，使其更好地处理混合类型chunks：
1. 添加对 `image_plus_text_as_answer` 类型的特殊处理
2. 改进错误处理和重试机制
3. 添加默认模板作为fallback

#### 方案3：降低失败影响
修改QA生成逻辑，即使模板选择失败也尝试生成QA：
1. 使用默认模板
2. 或者跳过模板选择步骤

### 修改文件
- `/home/project/UniDoc-Bench/src/qa_synthesize/2_qa_synthesize.py`

### 实施步骤
1. 添加调试输出功能
2. 改进 `choose_templates` 函数的错误处理
3. 添加 `--debug` 参数支持
4. 测试运行并查看调试输出