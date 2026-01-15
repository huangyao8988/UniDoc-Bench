## 实现计划

创建一个 Python 程序 `create_langfuse_dataset.py`，实现以下功能：

1. **导入必要的库**：langfuse, json, argparse 等
2. **定义配置参数**：
   - `--dataset-name`: dataset 名称（必需）
   - `--input-file`: 输入文件路径（默认：/home/project/UniDoc-Bench/output.jsonl）
   - `--public-key`: Langfuse Public Key（必需或从环境变量读取）
   - `--secret-key`: Langfuse Secret Key（必需或从环境变量读取）
   - `--host`: Langfuse 主机地址（可选，默认为 https://cloud.langfuse.com）
3. **实现主要逻辑**：
   - 初始化 Langfuse 客户端
   - 创建或获取指定的 dataset
   - 读取并解析 output.jsonl 文件
   - 遍历每条记录，提取：
     - `question` 作为 `input`
     - `answer` 作为 `expected_output`
     - 其他字段作为 `metadata`
   - 使用 `dataset.create_item()` 添加数据项
   - 显示进度和统计信息
4. **添加错误处理和日志输出**

程序将支持命令行参数和环境变量两种配置方式，提供灵活的使用方式。