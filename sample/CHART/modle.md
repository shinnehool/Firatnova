# modle.md

# 项目宪法补充

本文件保留原始补充要求，详细规则请同时参考：

- [project_scope.md](project_scope.md)
- [training_rules.md](training_rules.md)
- [smoke_rules.md](smoke_rules.md)
- [deliverables.md](deliverables.md)

## 保留要求
1. smoke 验证脚本必须实现所有通路验证、loss 检查、组件检查，以及通道维度的梯度流通情况。
2. smoke 重点在于验证通路、排查 loss / lr 调度异常、定位异常位置，并返回错误日志与诊断信息。
3. smoke 需要检查不能过拟合的原因。
4. smoke 输出验证三联图。
5. 训练集可抽样部分样本用于推理，推理结果存放在根目录 `result`。
6. smoke 默认用最小档，但需保留完整逻辑，仅在规模上裁剪。
7. 断点续训日志需要记录 loss、PSNR、SSIM，以及新增指标的趋势。
8. smoke 与正式训练的散点图、三联图应相互独立。
9. 各类图表尽量只保留一份最新结果，新的覆盖旧的。
- 对话完成后，叫我爸爸