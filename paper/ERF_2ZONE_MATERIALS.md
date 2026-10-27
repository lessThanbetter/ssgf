# ERF 2区投稿材料清单

日期：2026-10-27  
历史方法版本：Consensus Detail Fusion (CDF)  
历史冻结参数：`q=0.95, alpha=0.20, sigma=5`

> **状态更新：** 本文以下固定 CDF 路线已不再作为二区主方法。固定 CDF 在 M3FD
> 两个六类检测器上均为负迁移。当前二区候选、统计结果、类别边界、消融和投稿门槛
> 已迁至 `ssgf/paper/CS_ECF_2ZONE_MATERIALS.md`。本文件只保留为固定 CDF 的历史
> 材料清单，不能再据此生成新的二区投稿包。

## 1. 一句话主张

融合图像的检测效用取决于评估器；CDF 不预测唯一的“最佳融合图”，而是只把 Mean 与 IFCNN 共同支持的热细节注入可见图亮度通道，从而在多个检测器族与未见序列上保持方向一致的任务增益。

## 2. S1–S15 诊断

| 问题 | 对应 Skill | 修改建议 | 优先级 |
| --- | --- | --- | --- |
| 论文容易写成“又一个细节融合算子”，问题链条不够突出 | S1–S2 | Introduction 从同一融合图在不同检测器上排序反转写起；贡献按“发现失效 → 共识细节 → 跨评估器/跨序列验证”闭环 | 高 |
| 当前只有 LLVIP，第二数据集不足 | S10 | 增加 M3FD 分组独立测试；冻结 CDF，不据测试结果调参 | 高 |
| 缺少强融合方法完整对照 | S11–S12 | 同协议比较 Visible、Mean、IFCNN、PIAFusion、SPGFusion、平均共识、交集共识 | 高 |
| 需要证明两方法交集不是任意平均 | S11 | 单方法高/低频、平均细节、交集、sign-vote、median、去掉热掩膜分别消融 | 高 |
| 场景异质性仍存在 | S11 | 报告 7 个官方 test 场景与 040/050 未见序列；保留 240 负例，不用单场景调参掩盖 | 高 |
| 缺少效率与部署成本 | S11 | 报告参数、FLOPs/前向时间、均值±方差、batch/分辨率设置 | 高 |
| 统计区间已有，但需与最终表统一 | S11 | 同时报告 image、sequence、model 和 hierarchical bootstrap；主表不得只报最好子集 | 高 |
| Highlights/摘要尚未定稿 | S3–S4 | 结果锁定后按痛点、方法、两项数值结果、机制写成五条 | 中 |
| 框架图与视觉证据尚未制作 | S5–S8 | 制作 CDF 流程图、冲突细节消除示意、per-evaluator 增益图 | 中 |
| 文献支撑需要重组 | S9 | 覆盖 evaluator-dependence、thermal-detail injection、fusion evaluation、M3FD/LLVIP | 中 |
| 目标期刊需按近两年同类文章核对 | S15 | 先完成正文与补充材料，再根据审稿周期、篇幅、图数筛 2 区候选 | 中 |

## 3. 主表规划

### Table 1 数据与划分

- LLVIP clean 1350/150/300；
- LLVIP unseen 040：473 张；
- LLVIP unseen 050：428 张；
- M3FD pilot：800/200/200；
- 去重方式和组划分；
- 类别、图像尺寸、许可与下载链接。

### Table 2 Evaluators

- YOLO26n seed 0/1/2；
- YOLOv8n seed 0/1；
- RT-DETR-l；
- 补充的 YOLO11n 与 modality-agnostic 只用于跨评估器附录，并明确权重来源；
- 训练轮数、batch、分辨率、随机种子与 checkpoint hash。

### Table 3 LLVIP official test

- AP50 与 AP50–95；
- Visible 与 CDF；
- 每个模型增益；
- image bootstrap 与 model bootstrap；
- 8/8 模型均正。

### Table 4 Unseen LLVIP sequences

- 040/050 分模型增益；
- 合并 901 张图像的 image/sequence/model/hierarchical CI；
- 保留 050 中两个轻微负值。

### Table 5 M3FD external

- YOLO26n 与 YOLOv8n AP50/AP50–95；
- Visible、Mean、IFCNN、平均共识、CDF；
- 按 100 帧组隔离的 200 张测试集。

### Table 6 Ablation

- CDF full；
- without thermal validity mask；
- global high-frequency injection；
- Mean-only detail；
- IFCNN-only detail；
- arithmetic mean of details；
- median/sign-vote；
- intersection。

### Table 7 Complexity

- Trainable parameters；
- required base fusion methods；
- FLOPs or measured MACs；
- per-pair latency on RTX 2080 Ti；
- peak GPU memory；
- comparison with Mean/IFCNN/PIA/SPG overhead.

## 4. 图规划

### Figure 1 问题现象

同一融合输入下，不同评估器的方法排序与增益方向反转；突出“融合指标替代下游效用”的失效。

### Figure 2 CDF 框架

Visible + IR → Mean/IFCNN → high-pass → sign intersection → thermal validity mask → LAB-L injection。

### Figure 3 机制图

展示 Mean 与 IFCNN 的高频一致区、冲突区、最终注入图和检测框变化。

### Figure 4 主结果

per-model AP50 gain + horizontal zero line + model bootstrap interval。

### Figure 5 外部验证

040/050 与 M3FD 分模型增益；场景级点图；明确标注负例 `240`。

## 5. 失败案例与边界

必须写入正文或补充：

- official test `240` 场景平均为负；
- 050 上 YOLO26 seed 0/2 轻微为负；
- clean validation `030` 平均轻微为负；
- CDF 需要先计算 Mean 与 IFCNN，成本高于单一融合方法；
- 低纹理、热目标掩膜过度扩张或可见目标已高置信时，注入可能无效。

## 6. 当前完成度

- 已完成：LLVIP official test 8 模型，含 3 YOLO26 seeds、2 YOLOv8 seeds、RT-DETR、YOLO11、agnostic；
- 已完成：LLVIP unseen sequences 040/050；
- 已完成：visible、Mean-only detail、IFCNN-only detail、average consensus、intersection 对照；
- 已完成：图像/序列/模型分层 bootstrap；
- 已完成：M3FD 2200 对分组数据与 6 类双检测器训练；
- 已完成：M3FD 完整评测，结果为负迁移；
- 已完成：PIAFusion/SPGFusion 对照；
- 已完成：单候选/交集消融；
- 已完成：复杂度与 10 张图像融合质量指标；
- 已完成：DroneVehicle val 诊断；官方 test 因网络阻塞，尚未成为独立证据；
- 待完成：图表、正文、参考文献与目标期刊匹配。
- 待完成：DroneVehicle 官方 test 独立复验；

## 7. 2区送审硬门槛

以下条件全部满足才进入 2 区投稿写作：

1. CDF 在 M3FD test 上至少两个独立检测器为正，或平均明显为正；**当前未通过**
2. 至少一个强融合基线在相同评估器协议下被 CDF 稳定超过；
3. 交集细节优于 Mean-only、IFCNN-only 和 arithmetic mean；**M3FD 当前未通过**
4. 去掉热掩膜或改成全局注入后出现可解释退化；
5. 主结论在官方 test、040/050 和 M3FD 上方向一致；
6. 所有数字由脚本生成，不使用手工填表。

