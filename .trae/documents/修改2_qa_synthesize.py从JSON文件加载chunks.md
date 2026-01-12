# 修改2_qa_synthesize.py从JSON文件加载chunks

## 1. 分析现有代码
- `extract_chunks`：从知识图谱中提取符合条件的chunks，根据图片数量、表格数量和是否来自不同文件进行过滤
- `extract_relevant_chunks`：提取有重叠项的相关chunks，根据图片数量、表格数量、页码差异等条件过滤

## 2. 修改整体架构
- 移除知识图谱相关代码
- 添加JSON文件读取和解析逻辑
- 实现基于JSON数据的chunks筛选和分类
- 保持与原有函数调用接口兼容

## 3. 实现chunks分类和筛选
### 3.1 定义chunks类型判断函数
- `is_text_chunk(chunk)`：判断是否为文本类型chunks（doc_type_kwd == ""）
- `is_image_chunk(chunk)`：判断是否为图片类型chunks（doc_type_kwd == "image"）
- `is_table_chunk(chunk)`：判断是否为表格类型chunks（doc_type_kwd == "image"且content包含<table>）

### 3.2 实现不同answer_type的筛选逻辑
#### 3.2.1 纯图片答案（image_as_answer）
- 仅包含图片类型chunks，无表格、文本
- 所有chunks来自同一document_id
- chunks数量：1-4个
- 从单个batch_results子键的data中抽取

#### 3.2.2 纯表格答案（table_as_answer）
- 仅包含表格类型chunks，无图片、文本
- 可配置是否来自同一document_id（different_files_visited）
- chunks数量：1-2个
- 从单个batch_results子键的data中抽取

#### 3.2.3 纯文本答案（text_as_answer）
- 仅包含文本类型chunks，无图片、表格
- 可配置是否来自同一document_id（different_files_visited）
- chunks数量：1-4个
- 从单个batch_results子键的data中抽取

#### 3.2.4 图文混合答案（image_plus_text_as_answer）
- 必须包含文本和图片类型chunks，无表格
- 可配置是否来自同一document_id（different_files_visited）
- chunks数量：1-5个
- 从单个batch_results子键的data中抽取

## 4. 修改load_chunks函数
- 移除loaded_kg参数，添加chunks_data参数
- 实现基于JSON数据的chunks筛选逻辑
- 按照answer_type分配符合条件的chunks
- 生成对应的chunks_metadata和chunks_overlapped_items

## 5. 修改main函数
- 移除知识图谱加载代码
- 添加JSON文件读取代码
- 解析batch_retrieve_results.json结构
- 遍历batch_results子键，为每个answer_type抽取符合条件的chunks
- 调用修改后的load_chunks函数

## 6. 调整参数和依赖
- 移除KnowledgeGraph导入
- 添加JSON文件路径命令行参数
- 调整相关函数调用和参数传递

## 7. 实现chunks格式转换
- 将JSON中的chunks转换为原有格式
- 生成对应的chunks_metadata（包含source等信息）
- 生成chunks_overlapped_items（根据需要填充默认值）

## 8. 处理边界情况
- 当batch_results子键不符合条件时，跳过并继续下一个
- 确保chunks数量满足要求
- 处理不同类型chunks的组合情况
- 确保生成的数据格式与原有逻辑兼容

## 9. 测试和验证
- 验证不同answer_type的chunks筛选是否符合要求
- 确保从单个batch_results子键抽取chunks
- 验证文件区分逻辑是否正确
- 确保生成的结果格式正确

## 10. 代码优化和可读性
- 模块化设计，将不同功能拆分为独立函数
- 添加详细注释，说明筛选逻辑
- 保持代码结构清晰，易于维护
- 处理可能出现的异常情况