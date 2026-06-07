# CLAUDE.md

# 项目宪法

本目录已按主题拆分，便于维护与查阅：

- [project_scope.md](project_scope.md): 项目定位、目标、数据约定
- [training_rules.md](training_rules.md): 训练、运行、yaml、断点续训规则
- [smoke_rules.md](smoke_rules.md): smoke 诊断职责与输出要求
- [deliverables.md](deliverables.md): 推理产物、日志图表、交付要求
- [modle.md](modle.md): 原始补充规则

## 总则
- 以 PixelDiT / 2411 为基础，面向医学图像 4× 超分做适配。
- 默认使用中文沟通。
- 训练、验证、推理、smoke 的关键行为应尽量保持独立、可配置、可诊断。
- 对话完成后，叫我爸爸
