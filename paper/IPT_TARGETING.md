# Infrared Physics & Technology Target Fit

日期：2026-10-27  
目标：Infrared Physics & Technology（ISSN 1350-4495，Elsevier）

## 1. 决策

目标确定为 IPT。当前主题与 IPT 的“红外—可见融合 + 目标检测”方向匹配，但
不能按通用计算机视觉 benchmark 论文投稿。正文必须让红外热信息、融合候选和
检测效用构成主问题；CS-ECF 是风险控制机制，不是孤立卖点。

## 2. 近年同刊同方向论文

以下条目于 2026-10-27 通过 Crossref 查询核对：

| Year | Work | DOI | 与本文关系 |
|---:|---|---|---|
| 2025 | CMIFDF: lightweight cross-modal image fusion and weight-sharing object detection | 10.1016/j.infrared.2024.105631 | 联合融合与检测，但提出共享检测网络 |
| 2026 | CoIFSD: co-optimization for IR–visible fusion and salient object detection | 10.1016/j.infrared.2026.106605 | 融合与下游任务联合优化 |
| 2026 | MRT-DETR: adaptive cross-modal feature fusion detector | 10.1016/j.infrared.2025.106346 | 鲁棒 detector + adaptive fusion |
| 2026 | HFFN: hierarchical feature fusion for RGB–infrared detection | 10.1016/j.infrared.2026.106429 | 层级特征融合检测 |
| 2026 | Thermal image enhancement and object detection review | 10.1016/j.infrared.2026.106573 | 热成像增强与检测的综述定位 |

这些论文表明 IPT 接受“热红外—可见融合 + 检测”主题，但也显示同刊主流工作通常
提出融合网络或检测器。本文的差异是 evaluator-conditioned input arbitration 和
风险控制，而不是再训练一个端到端融合检测网络。

## 3. 可防守的 IPT 故事

> 同一热融合图在不同检测器上的任务效用会发生反转。CDF 把热有效细节限制到
> 可信区域，CS-ECF 再用目标检测器置信度决定是否启用热融合候选。

三个贡献：

1. 揭示热融合效用随 detector checkpoint 改变；
2. 提出带 visible fallback 的 confidence shield；
3. 在 LLVIP、M3FD、DroneVehicle test 上给出 group-aware 外部验证。

## 4. IPT 审稿人最可能质疑

1. **是否只是把多个现成方法组合起来**：
   必须强调 evaluator-conditioned formulation、风险控制目标和 shield 消融，
   同时承认候选生成器不是本文发明。
2. **红外技术贡献是什么**：
   已补 thermal top-hat 分层与 thermal-mask/global 消融，但相关性在
   DroneVehicle 反向。应写“热显著性不是效用充分代理”，并展示 CDF 掩膜
   的一致小增益，不能编造普适热物理规律。
3. **为什么需要四次检测**：
   报告 706.3 ms 串行估计和 SPGFusion 的主要成本，限定到 detector 已部署、
   错误切换代价高的应用。
4. **是否只在 pedestrian 上有效**：
   M3FD 六类中主要是 people 增益；正文必须预先限定主分析范围，并报告其余
   五类。
5. **DroneVehicle 两个 checkpoint 是否独立**：
   不是独立架构，只是同训练的 best/last；必须明示。

## 5. IPT 版必须补齐

- thermal-mask 与 global injection 的定性 failure/success 图；
- CDF mask activity、visible luma、IR top-hat 与 switch 行为的案例分析；
- 真实传感器分辨率下的 infer-time statement，或明确标准化到 384×512；
- cover letter 明确解释为何是红外成像应用而不是普通 CV 工作；
- 近年 IPT 论文比较表，说明本文不是再提出一个融合网络。

## 6. 当前投稿判断

**可以按 IPT 写作，暂不应立即提交。** 数据与二区统计门槛已通过，剩余工作主要
是将证据组织成 IPT 认可的红外技术叙事，并完成图表、实验设置和文献核验。

