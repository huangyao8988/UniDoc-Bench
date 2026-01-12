# 修改 2\_qa\_synthesize.py 实施计划

## 第一阶段：基础功能改造

### 步骤1：添加命令行参数

* 添加 `--chunks_json_path` 参数指定chunks库JSON文件路径

* 添加 `--test` 参数支持测试模式（mode01）

* 添加 `--QA` 参数控制生成特定answer\_type的QA

* 添加 `--different_file_image_as_answer` 参数

* 添加 `--different_file_table_as_answer` 参数

* 添加 `--different_file_text_as_answer` 参数

* 添加 `--different_file_image_plus_text_as_answer` 参数

* 修改OpenAI client初始化，支持环境变量 OPENAI\_API\_KEY 和 OPENAI\_BASE\_URL

**测试用例1**：验证所有新增参数可以正确解析

### 步骤2：实现从JSON加载chunks的函数

* 创建 `load_chunks_from_json()` 函数

* 解析JSON文件，提取所有chunks

* 根据 `doc_type_kwd` 和 `content` 分类chunks：

  * 文本：`doc_type_kwd == ""`

  * 图片：`doc_type_kwd == "image"` 且 `content` 不包含 `<table>`

  * 表格：`doc_type_kwd == "image"` 且 `content` 包含 `<table>`

* 使用 `document_id` 区分文件来源

**测试用例2**：验证从JSON正确加载和分类chunks，统计各类型chunks数量

### 步骤3：实现chunks组合函数

* 创建 `combine_chunks_by_answer_type()` 函数

* 实现四种answer\_type的组合规则：

  * `image_as_answer`: 1-4个图片chunks，根据参数控制文件来源

  * `table_as_answer`: 1-2个表格chunks，根据参数控制文件来源

  * `text_as_answer`: 1-4个文本chunks，根据参数控制文件来源

  * `image_plus_text_as_answer`: 2-5个chunks（必须包含文本和图片），根据参数控制文件来源

* 当chunks不足时跳过该组合

**测试用例3**：验证每种answer\_type的组合规则（文件来源、数量限制）

### 步骤4：实现测试模式mode01

* 创建 `test_mode01()` 函数

* 显示四种answer\_type chunks组的主要情况

* 将每种answer\_type的chunks组输出为独立JSON文件（包含所有chunk信息）：

  * `test_image_as_answer.json`

  * `test_table_as_answer.json`

  * `test_text_as_answer.json`

  * `test_image_plus_text_as_answer.json`

**测试用例4**：验证测试模式输出四个测试文件，且包含完整chunk信息

### 步骤5：实现QA类型过滤

* 修改主循环，根据 `--QA` 参数过滤answer\_type

* 如果未指定 `--QA`，则生成所有四种类型的QA

**测试用例5**：验证 `--QA` 参数正确过滤answer\_type

### 步骤6：移除不需要的函数

* 移除 `extract_chunks` 和 `extract_relevant_chunks` 的导入和调用

* 移除 `load_chunks` 函数中调用这两个函数的代码

* 更新 `load_chunks` 函数使用新的JSON加载逻辑

**测试用例6**：验证代码编译无错误，且功能正常

### 步骤7：创建自动化测试脚本

* 创建 `test_qa_synthesize.py` 脚本

* 包含所有测试用例的自动化执行

* 使用 `batch_retrieve_results_test.json` 进行测试

* 输出测试结果报告

**测试用例7**：运行自动化测试脚本，验证所有功能

## 测试策略

每个步骤完成后立即运行对应的测试用例，确保功能正确。

## 文件修改清单

1. `/home/project/UniDoc-Bench/src/qa_synthesize/2_qa_synthesize.py` - 主要修改
2. `/home/project/UniDoc-Bench/test_qa_synthesize.py` - 新建测试脚本

