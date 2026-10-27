# Evaluator-Robust Fusion: Feasibility Report

日期：2026-10-27  
当前结论：**固定 CDF 的二区门槛仍未通过；CS-ECF 已在第三个独立数据集上得到
确认，可以进入二区论文成形阶段，但仍不能写成“六类通用提升”。** CS-ECF 在
LLVIP、M3FD 和 DroneVehicle 官方 test 子集上共 12 个 checkpoint 中 9 个为正；
三数据集等权增益 `+0.0133`，model×group 分层 bootstrap 为
`[+0.0028,+0.0213]`。
DroneVehicle 的两个同训练 checkpoint 分别为 `+0.0251/+0.0293`，三个固定
100 帧组均为非负，组合 group CI 为 `[+0.0002,+0.0406]`。主要剩余限制是：
LLVIP 单独区间仍跨零，M3FD 的 mAP 增益几乎全部来自 `people` 类，
DroneVehicle 的两个 checkpoint 不是独立架构。

## 1. 方法原型

当前冻结的候选称为 **Consensus Detail Fusion (CDF)**：

1. 生成 Mean 与 IFCNN 两个全局融合候选；
2. 提取两者的高频细节 \(d_m,d_i\)；
3. 用 IR top-hat、可见亮度与连通域构造热有效性掩膜 \(M\)；
4. 只保留两种融合器符号一致的细节：

\[
d(x)=
\begin{cases}
\operatorname{sign}(d_m+d_i)\min(|d_m|,|d_i|), & d_m d_i>0,\\
0,&\text{otherwise};
\end{cases}
\]

5. 将细节注入可见图的 LAB-L 通道，保持色度不变：

\[
L'=L+\alpha M d.
\]

当前固定参数：`q=0.95, alpha=0.20, sigma=5`。  
推理时不需要任何目标检测器。

## 2. 为什么不是 VASF

- 图像级、检测置信度驱动的 VASF 门控未能跨评估器泛化。
- 跨评估器固定算子明显更稳定。
- 因此应删除“预测单图最优候选”的主张，改为“多个融合候选共同支持的热细节更稳健”。

## 3. Test-set evaluator robustness

同一 300 张 LLVIP test 图像、同一 CDF 输出：

| Evaluator | Visible AP50 | CDF AP50 | Gain | Gain AP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n | 0.8811 | 0.8918 | +0.0106 | +0.0125 |
| Modality-agnostic | 0.8550 | 0.8609 | +0.0058 | +0.0109 |
| YOLO26n seed 0 | 0.8122 | 0.8168 | +0.0048 | +0.0024 |
| RT-DETR-l | 0.8095 | 0.8138 | +0.0042 | +0.0107 |
| YOLO26n seed 1 | 0.7937 | 0.8035 | +0.0098 | +0.0171 |
| YOLO26n seed 2 | 0.8061 | 0.8183 | +0.0121 | +0.0095 |
| YOLOv8n held-out seed 0 | 0.7801 | 0.7864 | +0.0063 | +0.0084 |
| YOLOv8n held-out seed 1 | 0.8055 | 0.8082 | +0.0025 | +0.0068 |

汇总：

- 8/8 模型点估计为正；
- 平均增益 +0.0070；
- 模型 bootstrap 95% CI：`[+0.0049, +0.0091]`；
- 图像 bootstrap 95% CI：`[0.0000, +0.0134]`；
- 场景聚类 95% CI：`[-0.0029, +0.0177]`；
- 模型×场景分层 95% CI：`[-0.0042, +0.0188]`。

解释：**跨评估器方向稳定，跨场景显著性尚未建立。**

## 4. Scene-level blocker

七个 test 场景的平均增益：

| Scene | Mean gain |
|---|---:|
| 190 | +0.0047 |
| 200 | +0.0372 |
| 210 | -0.0046 |
| 220 | +0.0123 |
| 230 | -0.0041 |
| 240 | -0.0247 |
| 260 | +0.0100 |

只有 4/7 场景平均为正。`240` 是最严重的稳定负迁移，多个 YOLO26 种子上均明显为负。

## 5. Clean validation

- YOLO26n seed 0/1/2 的 CDF 增益：`-0.0023/-0.0028/-0.0005`。
- YOLOv8n held-out seed 0 validation：`-0.0068`。
- YOLOv8n held-out seed 1 validation：`-0.0026`。
- RT-DETR-l validation：`+0.0051`。

validation 与 test 的差异说明当前 split 的 `030` 场景与 test 七场景存在显著异质性，不能把 validation 的轻微负值忽略。

## 6. 已试但失败的鲁棒化

- Activation budget 4%/5%/6%：会抹掉 YOLO26 seed 1/2 的大部分测试收益。
- Mask-area cap 6%/7%/8%：效果同样过强，不能解决 `240` 而不伤及其他场景。
- Sign-vote 六个融合器：YOLOv8n/RT-DETR 稳定性不如 Mean×IFCNN 交集。
- Trimmed mean / median six：没有比两方法符号交集更稳。
- 图像级 utility selector：在 leave-one-evaluator-out 下仍不可泛化。
- `q=0.99`：改善部分 YOLO26 结果，但在 YOLOv8n seed 1 上降至 −0.0001，未优于 q95 的跨模型稳定性。
- Mask-area hard gate 9%/10%：明显抹掉 YOLO26 seed 1 的收益。

## 7. 当前判断

### 可以继续的依据

- 方法无检测器依赖；
- 8/8 评估器/种子测试点估计为正；
- 包含一个完全留出的 YOLOv8n 检测器族；
- 模型层 bootstrap 显著；
- 方法简单、可复现、推理成本可控。

### 不能直接投稿的原因

- 场景聚类区间跨零；
- `240` 场景稳定负迁移；
- clean validation 上多个检测器轻微为负；
- 尚无第二个具有检测标签的数据集；
- 当前参数仍需通过独立预注册协议锁定；
- 尚无与可学习融合网络和最新 SOTA 的完整对照。

## 8. 建议的下一步

1. 冻结 CDF 公式与参数，不再根据 test 调整；
2. 在 FLIR 或其它带人检测标签数据集上测试；
3. 若跨数据集仍为正，再实现正式模块、复杂度和消融实验；
4. 若第二个数据集失败，停止方法论文转向。

本地当前没有可直接使用的 FLIR/MSRS 人检测标签，因此跨数据集验证是下一步外部 blocker。

## 9. 结果文件

- `ssgf/results/selective_fusion/test_scores_consensus_intersection/scores.json`
- `ssgf/results/selective_fusion/test_scores_consensus_s1/scores.json`
- `ssgf/results/selective_fusion/test_scores_consensus_s2/scores.json`
- `ssgf/results/selective_fusion/test_scores_yolo8_holdout/scores.json`
- `ssgf/results/selective_fusion/consensus_q095_evaluator_bootstrap_v2.json`
- `ssgf/results/selective_fusion/consensus_q099_evaluator_bootstrap.json`
- `ssgf/results/selective_fusion/test_scores_yolo8_s1/scores.json`

## 10. 未见 LLVIP 序列 040/050

为补足单一 test 集合的问题，另抽取 clean split 完全未使用的官方 train 序列：

- `040`：473 张；
- `050`：428 张；
- 两者与 train/validation/test 的 overlap 均为 0；
- 全部使用 clean 检测器评估；
- 图片临时放在 C 盘，评测后已删除；manifest、评分 NPZ 和统计结果保留在工作区。

### 分模型、分序列 AP50 增益

| Model | 040 | 050 | 合并 |
|---|---:|---:|---:|
| YOLO26n seed 0 | +0.0137 | -0.0033 | +0.0054 |
| YOLO26n seed 1 | +0.0091 | +0.0025 | +0.0056 |
| YOLO26n seed 2 | +0.0050 | -0.0005 | +0.0028 |
| YOLOv8n seed 0 | +0.0089 | +0.0064 | +0.0084 |
| YOLOv8n seed 1 | +0.0041 | +0.0022 | +0.0035 |
| RT-DETR-l | +0.0155 | +0.0008 | +0.0094 |

合并 901 张图像：

- 6/6 模型为正；
- 平均增益 +0.0058；
- 图像 bootstrap 95% CI：`[+0.0033,+0.0082]`；
- 序列 bootstrap 95% CI：`[+0.0013,+0.0094]`；
- 模型 bootstrap 95% CI：`[+0.0039,+0.0077]`；
- 图像×序列×模型分层 95% CI：`[+0.0002,+0.0116]`。

这是目前最强结果：在完全未见的两个序列上，四个区间全部保持为正。`050` 仍有 YOLO26 seed 0/2 的轻微负值，说明单序列内部并不一致，但两个序列合并后所有 clean 模型为正。

### 分区判断

- **3 区：当前仍具备较强可行性。**
- **2 区：按已执行实验，门槛未通过。** M3FD 上 CDF 对两个 clean 六类检测器均为负迁移。
- **1 区：仍不够。**

在决定是否按 2 区推进时，不应再使用测试数据调方法。下一步应该是第二数据来源或正式方法模块，而不是继续优化 LLVIP 指标。

### 新增结果文件

- `ssgf/results/selective_fusion/external_llvip040_manifest.json`
- `ssgf/results/selective_fusion/external_llvip050_manifest.json`
- `ssgf/results/selective_fusion/external_llvip040_bootstrap.json`
- `ssgf/results/selective_fusion/external_llvip040_050_bootstrap.json`
- `ssgf/results/selective_fusion/external_llvip040_scores/scores.json`
- `ssgf/results/selective_fusion/external_llvip050_scores/scores.json`

## 11. M3FD 第二数据集：2区门槛未通过

### 数据协议

- M3FD 6 类：people、car、bus、motorcycle、lamp、truck；
- 按 100 帧连续组划分，避免相邻帧跨 split；
- train/validation/test = 1600/400/200；
- 测试组完全固定，未用于 CDF 调参；
- 数据、标签、检测器全部保存在
  `D:\Code\_ssgf_external\m3fd_erf_pilot`。

### 六类检测器

| Detector | Validation mAP50 | Validation mAP50-95 |
|---|---:|---:|
| YOLO26n | 0.533 | 0.358 |
| YOLOv8n | 0.504 | 0.323 |

### Test 全 6 类结果

| Detector | Visible AP50 | CDF q95 AP50 | Gain |
|---|---:|---:|---:|
| YOLO26n | 0.3717 | 0.3705 | -0.0012 |
| YOLOv8n | 0.3790 | 0.3731 | -0.0059 |

进一步在 M3FD validation 上选择最小一致幅值，τ=2 在
YOLO26n/YOLOv8n 上分别为 +0.0054/+0.0056；但同一 τ 在固定 test 组上为
-0.0014/-0.0056。τ=4 的 test 结果也为一正一负。验证集最优参数未迁移到测试组，
说明 M3FD 上再次发生 validation-to-test failure，不能据此选择 τ 并宣称外部有效。

以最终采用的类感知 mAP 脚本重算：

- q95 CDF 两模型平均 -0.0036；
- 0/2 模型为正；
- image bootstrap 95% CI：`[-0.0089,+0.0035]`；
- group-cluster 95% CI：`[-0.0100,-0.0002]`；
- model bootstrap 95% CI：`[-0.0059,-0.0012]`；
- hierarchical 95% CI：`[-0.0114,+0.0060]`。

旧文档中的 `-0.0044` 和 `[-0.0083,-0.0021]` 来自单类 pooled-AP 口径，
不能继续作为主表数字。

尝试 q99 或 pixel-mean × IFCNN 版本后仍为负。因此：

- M3FD 负迁移不是低质量 pilot 检测器造成；
- 不是仅由冻结自编码器 Mean 候选造成；
- CDF 在 LLVIP 夜间的收益不能外推到 M3FD。

### 强融合基线

同一 M3FD test、同一两个六类检测器：

| Input | YOLO26n AP50 | YOLOv8n AP50 |
|---|---:|---:|
| Visible | 0.3717 | 0.3790 |
| PIAFusion | 0.2078 | 0.2202 |
| SPGFusion | 0.2424 | 0.2591 |
| CDF q95 | 0.3705 | 0.3731 |

CDF 对全局融合灾难性退化有很强的抑制作用，但没有超过可见图。这个现象适合作为“稳健融合”的辅助论据，不能替代正收益。

### M3FD 消融

| Input | YOLO26n AP50 | YOLOv8n AP50 |
|---|---:|---:|
| Visible | 0.3717 | 0.3790 |
| Pixel mean detail only | 0.3679 | 0.3731 |
| IFCNN detail only | 0.3729 | 0.3716 |
| CDF intersection | 0.3705 | 0.3731 |

交集在两模型之间没有稳定优于单 IFCNN 或单 Mean detail，削弱了“共识模块是主要收益来源”的机制主张。

### 复杂度

在 1024×768、RTX 2080 Ti 上：

- IFCNN 冻结参数：75,073；
- CDF 可训练参数：0；
- pixel mean：3.1 ms；
- IFCNN：33.0 ms；
- CDF 后处理：61.0 ms；
- 端到端：101.1 ms/对。

### 投稿含义

如果把现有结果投 2 区，最可能被审稿人抓住的问题是“只在 LLVIP 成立”。因此 **2 区标准尚未达到**。下面两条路二者择一：

1. 继续做方法学研究，开发能在 M3FD 与 LLVIP 同时为正的阈值/一致性门控；
2. 以当前 LLVIP 跨评估器、跨序列证据投 3 区，并把 M3FD 负迁移作为 limitation 与 negative result。

### 新增结果文件

- `ssgf/results/selective_fusion/m3fd_6cls_multiclass/scores.json`
- `ssgf/results/selective_fusion/m3fd_6cls_bootstrap.json`
- `ssgf/results/selective_fusion/cdf_complexity.json`
- `ssgf/results/selective_fusion/m3fd_sota_scores/scores.json`
- `ssgf/results/selective_fusion/m3fd_ablation/scores.json`
- `ssgf/results/selective_fusion/m3fd_cdf_fidelity_10.json`
- `ssgf/results/selective_fusion/cdf_pixelmean_complexity.json`

## 12. DroneVehicle：官方 val 诊断与独立 test

已经从官方 val 抽取 300 对，按 3 个连续 100 帧组评估 YOLO11x-OBB。
以下 val 结果只作为历史诊断，不进入 confirmatory 主表。

### 结果

| Evaluator | Visible AP50 | CDF AP50 | Gain |
|---|---:|---:|---:|
| YOLO11x-OBB best | 0.6747 | 0.6945 | +0.0198 |
| YOLO11x-OBB last | 0.6792 | 0.7200 | +0.0408 |

best checkpoint：

- 类感知 image bootstrap 95% CI：`[+0.0059,+0.0390]`；
- 类感知 group bootstrap 95% CI：`[+0.0001,+0.0424]`；
- 3 个官方组均为正，增量分别为 `+0.0001/+0.0064/+0.0424`。

### 为什么官方 val 不能计入主表

- 该第三方 checkpoint 很可能使用官方 val 选择 best；
- 因此官方 val 对模型不是独立测试；
- 官方 test.zip 的远程分块读取连续被 CDN 拒绝；
- 因此当时不能把 DroneVehicle val 写成独立外部验证。

此外，同一 val 子集上 PIAFusion/SPGFusion 分别为：

- Visible：0.6747；
- PIAFusion：0.8619；
- SPGFusion：0.8751。

这进一步说明 CDF 的价值是“避免灾难性退化”，不是跨域绝对性能最优。

### 12.1 官方 test 独立复验

随后从完整官方 `test.zip` 中按预先固定的随机种子和连续帧组抽取 300 对：

- 官方 test 组：`[7, 32, 71]`；
- 源归档大小 `4,430,343,899` bytes，SHA-256
  `94BF47E493E7A57FD5D900439F614F339E7B6C7181E4D781BCFEBAEB963FFD8E`；
- 每组 100 个连续帧；
- Visible、IR、XML 和单类 vehicle 标签均本地保存；
- 同一模型训练的 best 与 last 两个 checkpoint 分别评估；
- 候选生成参数与验证协议完全一致。

| Checkpoint | Visible AP50 | CS-ECF AP50 | Shield gain | No-shield gain |
|---|---:|---:|---:|---:|
| YOLO11x-OBB best | 0.6563 | 0.6814 | +0.0251 | +0.1064 |
| YOLO11x-OBB last | 0.6474 | 0.6767 | +0.0293 | +0.1016 |
| 两 checkpoint 平均 | - | - | +0.0272 | +0.1040 |

两个 checkpoint 的组合 bootstrap：

- image CI95：`[-0.0028,+0.0682]`；
- group CI95：`[+0.0002,+0.0406]`，这是主分析区间；
- best 的三个组增益：`+0.0199/+0.0002/+0.0241`；
- last 的三个组增益：`+0.0189/+0.0002/+0.0388`。

解释必须克制：

- 所有 6 个 checkpoint×group 单元均非负，但组 32 的增益只有 `+0.0002`；
- 只有 3 个测试组，cluster bootstrap 的分辨率有限；
- best/last 来自同一训练运行，不是两个独立架构；
- no-shield 在 DroneVehicle 上收益更高但波动极大，说明当前 shield 是风险控制，
  不是最大化平均收益。

在当前证据下，DroneVehicle test 可以进入 confirmatory 主表，但仍需在论文中明确
“两个 checkpoint 同训练”的依赖关系。

### 新增结果文件

- `ssgf/results/selective_fusion/dronevehicle_scores/scores.json`
- `ssgf/results/selective_fusion/dronevehicle_bootstrap.json`
- `ssgf/results/selective_fusion/dronevehicle_scores_last/scores.json`
- `ssgf/results/selective_fusion/dronevehicle_sota_scores/scores.json`
- `D:\Code\_ssgf_external\dronevehicle_val300`
- `ssgf/results/selective_fusion/dronevehicle_test_summary.json`
- `ssgf/results/selective_fusion/dronevehicle_test_last_summary.json`
- `ssgf/results/selective_fusion/dronevehicle_test_combined.json`
- `D:\Code\_ssgf_external\dronevehicle_test300`

## 13. CS-ECF：新的二区候选

### 13.1 定义

对每个输入图像构造四个候选：

1. Visible；
2. CDF；
3. PIAFusion；
4. SPGFusion。

用当前目标检测器分别运行四个候选，并对每个候选取 Top-5 检测置信度均值：

\[
q_c=\frac{1}{5}\sum_{i=1}^{5}s_{c,i}.
\]

令 \(c^\*=\arg\max_c q_c\)。冻结策略为：

\[
c=
\begin{cases}
c^\*,&c^\*\ne\mathrm{vis}\ \text{且}\ q_{c^\*}\ge1.5q_{\mathrm{vis}},\\
\mathrm{vis},&\text{otherwise}.
\end{cases}
\]

该策略不使用测试标签，但运行时需要目标检测器本身和四遍候选前向。它不是一个
单一融合网络，而是“评估器感知的融合输入仲裁器”，论文必须明确这一点及部署成本。

### 13.2 Validation 选择

阈值只在 validation 上选择：

| Dataset | Evaluator | Gain AP50 |
|---|---|---:|
| LLVIP | YOLO26n | +0.0018 |
| LLVIP | RT-DETR-l | -0.0001 |
| M3FD | YOLO26n | +0.0015 |
| M3FD | YOLOv8n | +0.0001 |

冻结值：`Top-5 mean, visible_margin=1.5`。validation 证据很薄，不能把它写成
“预注册显著策略”；工作区没有时间戳预注册文件。

### 13.3 Confirmatory test 结果

| Dataset | Evaluator | Visible AP50 | CS-ECF AP50 | Gain |
|---|---|---:|---:|---:|
| LLVIP | YOLO11n | 0.8811 | 0.8845 | +0.0034 |
| LLVIP | Modality-agnostic | 0.8550 | 0.8611 | +0.0061 |
| LLVIP | YOLO26n seed 0 | 0.8122 | 0.8088 | -0.0034 |
| LLVIP | RT-DETR-l | 0.8095 | 0.8136 | +0.0040 |
| LLVIP | YOLO26n seed 1 | 0.7937 | 0.7911 | -0.0027 |
| LLVIP | YOLO26n seed 2 | 0.8061 | 0.8087 | +0.0026 |
| LLVIP | YOLOv8n seed 0 | 0.7801 | 0.7761 | -0.0040 |
| LLVIP | YOLOv8n seed 1 | 0.8055 | 0.8119 | +0.0064 |
| M3FD | YOLO26n | 0.3717 | 0.3757 | +0.0041 |
| M3FD | YOLOv8n | 0.3790 | 0.3972 | +0.0182 |
| DroneVehicle test | YOLO11x-OBB best | 0.6563 | 0.6814 | +0.0251 |
| DroneVehicle test | YOLO11x-OBB last | 0.6474 | 0.6767 | +0.0293 |

汇总：

| Scope | Positive | Mean | Worst | Model CI | Image CI | Group/Hierarchical CI |
|---|---:|---:|---:|---:|---:|---:|
| LLVIP | 5/8 | +0.0016 | -0.0040 | `[-0.0014,+0.0042]` | `[-0.0010,+0.0051]` | - |
| M3FD | 2/2 | +0.0111 | +0.0041 | `[+0.0041,+0.0182]` | `[+0.0036,+0.0187]` | - |
| DroneVehicle test | 2/2 | +0.0272 | +0.0251 | `[+0.0251,+0.0293]` | `[+0.0002,+0.0611]` | group `[+0.0002,+0.0406]` |
| 三数据集等权 | 9/12 | +0.0133 | -0.0040 | `[+0.0101,+0.0167]` | `[+0.0032,+0.0248]` | group `[+0.0028,+0.0213]` |

解释：

- 三数据集等权分层区间为正，但 LLVIP 的三种区间仍均跨零；
- M3FD 的两个检测器均正，但只有两个检测器，模型层不确定性很大；
- 12 个 checkpoint 简单平均为 `+0.0074`，数据集等权为 `+0.0133`；
- DroneVehicle image CI 跨零，group CI 仅略高于零；两个 checkpoint 同训练，
  不能当作两个独立检测器架构。

### 13.4 Shield 消融

去掉 1.5 倍安全阈值后，12 个 checkpoint：

- no-shield 仅 7 个为正，平均 `+0.0132`，最差 `-0.0584`；
- CS-ECF 为 9 个为正，平均 `+0.0074`，最差 `-0.0040`；
- DroneVehicle 上 no-shield 平均 `+0.1040`，shield 后只有 `+0.0272`；
- LLVIP 上 shield 把最差退化从 `-0.0584` 压到 `-0.0040`。

因此 shield 的价值是明显的尾部风险控制，但会牺牲 DroneVehicle 的高峰值收益。
摘要和讨论必须按这个风险—收益交换表述，不能写成平均性能全面提升。

### 13.5 固定输入对照

| Policy | Positive | Mean gain | Dataset-balanced mean | Worst |
|---|---:|---:|---:|---:|
| Always CDF | 10/12 | +0.0052 | +0.0031 | -0.0059 |
| Always PIAFusion | 3/12 | -0.1087 | -0.0710 | -0.2910 |
| Always SPGFusion | 2/12 | -0.1136 | -0.0602 | -0.2501 |
| Top-5 no shield | 7/12 | +0.0132 | +0.0332 | -0.0584 |
| CS-ECF | 9/12 | +0.0074 | +0.0133 | -0.0040 |

Always CDF 仍在 10/12 个 checkpoint 上为正；CS-ECF 的优势是三数据集等权收益更高、
最差退化更小，并非对 Always CDF 的全面支配。no-shield 等权平均更高，但最差退化
达到 `-0.0584`。这些是二区审稿中最需要正面讨论的 baseline。

### 13.6 类别诊断

LLVIP 只有 pedestrian 一类，CS-ECF 的类别结果与 AP50 相同。

M3FD 六类：

| Evaluator | People gain | People image CI | Other five classes |
|---|---:|---:|---|
| YOLO26n | +0.0214 | `[-0.0070,+0.0498]` | 基本不变；lamp +0.0030 |
| YOLOv8n | +0.1090 | `[+0.0440,+0.1913]` | 最大绝对变化 0.0003 |

因此 M3FD 的总体正增益是一个明显的 **people-class effect**。若论文声称六个类别
普遍优于 Visible，审稿人可以很容易地用类别表否定。当前只支持
“pedestrian/people 为主的目标场景中，gate 可以带来条件性收益”，不支持六类均匀
提升。

### 13.7 失败的新策略

在 DroneVehicle test 完成前，为避免仅凭测试结果调参，又用 validation 选了两个
候选并在当时可用的 10 个固定测试评估器上评估：

| Policy | Validation result | Test positive | Test mean | Test hierarchical CI | Decision |
|---|---|---:|---:|---:|---|
| CDF+PIA, Top-10, margin 1.3 | 4/4 positive, min +0.0022 | 5/10 | +0.0021 | `[-0.0029,+0.0127]` | reject |
| Visible+SPG, mean, margin 1.5 | 4/4 non-negative, mean +0.0036 | 6/10 | -0.0039 | `[-0.0085,+0.0013]` | reject |

这两个负例说明四个 validation 评估器不足以稳定选择 gate 超参数。它们必须保留在
补充材料或内部记录中，不能只报原始 CS-ECF 而隐藏。

### 13.8 结果文件

- `ssgf/results/selective_fusion/confidence_shield_summary.json`
- `ssgf/results/selective_fusion/confidence_shield_policy_comparison.json`
- `ssgf/results/selective_fusion/confidence_shield_class_diagnostics.json`
- `ssgf/results/selective_fusion/confidence_shield_class_bootstrap.json`
- `ssgf/results/selective_fusion/confidence_shield_validation_search.json`
- `ssgf/results/selective_fusion/confidence_shield_v2_summary.json`
- `ssgf/results/selective_fusion/confidence_shield_v3_summary.json`
- `ssgf/results/selective_fusion/confidence_shield_three_dataset_summary.json`
- `ssgf/results/selective_fusion/confidence_shield_group_hierarchical.json`
- `ssgf/results/selective_fusion/confirmatory_policy_comparison.json`
- `ssgf/results/selective_fusion/dronevehicle_test_combined.json`
- `ssgf/results/selective_fusion/cs_ecf_complexity.json`
- `ssgf/results/selective_fusion/ipt_thermal_stratification.json`
- `ssgf/results/selective_fusion/ipt_mask_ablation.json`

### 13.9 复杂度

统一到 384×512 候选输入、640 检测输入、batch=1、RTX 2080 Ti：

| Component | Latency |
|---|---:|
| CDF pipeline | 22.3 ms |
| PIAFusion | 37.7 ms |
| SPGFusion | 516.3 ms |
| Four detector forwards | 130.0 ms |
| Serial estimate | 706.3 ms |

SPGFusion 是主要成本；其单阶段额外峰值显存约 1453.5 MiB。该结果是串行组件
估计，不包含模型换入换出和数据加载，因此不能写成完整的部署 wall time。

### 13.10 IPT 所需的热特征诊断

首轮热分层分析显示：

- LLVIP：thermal top-hat q95 与切换比例的 Spearman `rho=0.1505, p=0.0090`；
- M3FD：`rho=0.3198, p<1e-5`；
- DroneVehicle test：`rho=-0.4027`，方向反转；
- high-thermal/low-visible 中位数分组没有稳定带来更高 AP 增益。

因此不能声称“热对比度越高，融合越有益”。对 IPT 更准确的定位是：
原始热显著性不是检测效用的充分代理，融合候选是否启用仍必须由目标检测器
条件化判断。该负面分层结果应保留，不应为了迎合热成像叙事而删除。

DroneVehicle 固定输入的热掩膜消融则给出一致但较小的正效应：

- best：masked CDF `+0.0060`，global injection `+0.0057`；
- last：masked CDF `+0.0050`，global injection `+0.0038`；
- 两 checkpoint 平均 mask 优势 `+0.0007 AP50`。

这能支持“热有效性掩膜有可测贡献”，但不足以支持强物理机制声明。

## 14. 二区判断与剩余工作

当前可以按下面的窄主张进入二区写作：

> 在多模态 pedestrian detection 中，检测器身份会改变融合输入的效用；采用带
> 显著置信度保护的 evaluator-conditioned arbitration，可以在保留 Visible 安全
> 底线的同时，对少数高增益图像启用融合候选，并在 LLVIP、M3FD、DroneVehicle
> 三个测试来源上获得正的等权平均收益。

科学验证门槛已经从“缺第三数据集”推进为“条件通过”。提交前仍必须完成：

1. 把 `people/pedestrian` 从结果解释升级为预先声明的主分析范围，避免声称六类
   均匀提升；
2. 补一个独立 OBB 检测器架构，或把 DroneVehicle 的外部效度明确限定为
   “同训练 best/last checkpoint + 独立 test groups”；
3. 完成四候选生成、四遍检测前向、端到端延迟与显存实测；
4. 绘制 evaluator reversal、CS-ECF 流程、risk-return tradeoff 和类别效应四组图；
5. 按二区正文标准完成 Introduction、Methods、Results、Discussion、Highlights、
   摘要、文献和投稿信；
6. 用近两年同类型二区论文复核篇幅、图表数量与统计报告规范。

不再在当前 LLVIP/M3FD/DroneVehicle test 上调 margin、Top-k 或候选集合。现有
12 个 confirmatory checkpoint 已经足以支持写作；继续试阈值只会把 confirmatory
结果变成探索性结果。

