修改 `/home/project/UniDoc-Bench/src/qa_synthesize/2_qa_synthesize.py` 文件中的 `is_image_chunk` 函数：

**修改内容**：
- 将 `is_image_chunk(chunk)` 函数的返回逻辑从 `chunk.get("doc_type_kwd", "") == "image"` 修改为 `chunk.get("doc_type_kwd", "") == "image" and "<table>" not in chunk.get("content", "")`

**修改位置**：
- 第27-29行的 `is_image_chunk` 函数

**修改目的**：
- 使 `is_image_chunk` 与 `is_table_chunk` 互斥
- `is_table_chunk`: `doc_type_kwd == "image"` 且 `content` **包含** `<table>`
- `is_image_chunk`: `doc_type_kwd == "image"` 且 `content` **不包含** `<table>`