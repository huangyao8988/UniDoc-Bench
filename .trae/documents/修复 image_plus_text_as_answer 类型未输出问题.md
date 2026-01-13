## 修复 `image_plus_text_as_answer` 类型未被输出的方案

### 问题根源
`choose_templates` 函数在处理 `image_plus_text_as_answer` 类型时失败，导致该类型的 chunk 组被跳过，没有生成 QA 对。

### 解决方案

**方案1：启用调试模式查看详细错误信息**
- 在运行命令中添加 `--debug` 参数
- 重新运行脚本，查看具体的错误信息
- 根据错误信息针对性修复

**方案2：改进错误处理逻辑**
- 修改代码，即使 `choose_templates` 失败，也尝试使用默认模板
- 或者在 `choose_templates` 失败时，尝试降级处理（如只使用文本或只使用图像）

**方案3：检查图像文件路径**
- 验证 `--folder_elements` 参数指向的路径是否正确
- 检查图像文件是否存在
- 确保图像文件可访问

**方案4：增加重试机制**
- 在 `choose_templates` 失败时增加重试次数
- 在 API 调用失败时增加重试逻辑

**方案5：临时绕过（用于测试）**
- 使用 `--QA image_plus_text_as_answer` 参数单独生成该类型
- 或修改代码，在 `choose_templates` 失败时使用硬编码的默认模板

### 推荐执行步骤
1. 首先使用 `--debug` 参数重新运行，查看具体错误
2. 根据错误信息确定是图像加载问题、API 问题还是解析问题
3. 针对性地修复代码或配置
4. 验证修复后 `image_plus_text_as_answer` 类型能正常输出