## 添加第3种运行模式（--test mode02）的实现计划

### 目标
添加 `--test mode02` 参数，融合前两种模式的功能：
- 输出中间过程生成的JSON文件（如mode01）
- 输出最终QA的jsonl文件（如正常模式）

### 实现步骤

**步骤1：修改test模式判断逻辑（第606-609行）**
- 将条件从 `if args.test == "mode01":` 改为 `if args.test in ["mode01", "mode02"]:`
- 调用 `test_mode01(batch_chunks_list, chunk_groups_by_answer_type)` 函数
- 添加条件判断：如果是mode01，执行 `sys.exit(0)` 退出；如果是mode02，继续执行后续代码

**步骤2：可选优化test_mode01函数（第249-318行）**
- 为 `test_mode01()` 函数添加可选参数 `verbose=True`，控制是否输出统计信息
- 在mode02模式下调用时传入 `verbose=False`，减少冗余输出

### 修改后的行为
- `--test mode01`：输出中间JSON文件 + 统计信息 → 退出（不生成QA）
- `--test mode02`：输出中间JSON文件 + 统计信息 → 继续生成QA jsonl文件
- 不加 `--test`：不输出中间JSON文件 → 直接生成QA jsonl文件