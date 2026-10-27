# CS-ECF 二区投稿材料

日期：2026-10-27  
方法版本：Confidence-Shielded Evaluator-Conditioned Fusion（CS-ECF, v1）  
冻结策略：四个候选、Top-5 置信度均值、`visible_margin=1.5`

## 1. 当前判断

**已经进入二区论文成形阶段，但仍不能直接生成 submission zip。**

固定 CDF 的二区路线已经失败：它在 LLVIP 上稳定，却在 M3FD 六类测试上为
`-0.0012/-0.0059`。CS-ECF 目前已经证明了四点：

1. 去掉安全阈值后，Top-5 选择器会产生最高 `-0.0584` 的退化；
2. 1.5 倍 shield 将 12 个 checkpoint 的最差退化压到 `-0.0040`；
3. 三数据集等权平均增益为 `+0.0133`，model×group 分层区间
   `[+0.0028,+0.0213]`；
4. DroneVehicle 独立 test 的两个 checkpoint 均正，组合 group
   CI 为 `[+0.0002,+0.0406]`。

M3FD 的总体增益仍几乎全部来自 `people` 类，因此论文必须写成
pedestrian/people-centric 的条件性结论。

因此当前最稳妥的二区故事是“评估器条件化、风险受控的融合输入仲裁”，不是
“提出一种全面更强的新型图像融合网络”。

## 2. 一句话主张

> 多模态融合图的检测效用依赖具体评估器；与其假设一张融合图普遍最优，不如让
> 目标检测器在 Visible、CDF、PIAFusion 和 SPGFusion 之间做带安全阈值的输入
> 仲裁，并在多数据集、多评估器上控制最坏退化。

英文工作版：

> Fusion utility is evaluator-conditioned. A confidence-shielded arbitration
> policy preserves the visible-image safety baseline and enables fusion inputs
> only when they provide a clear, detector-specific advantage.

## 3. S1–S15 诊断

| 问题 | Skill | 修改建议 | 优先级 |
|---|---|---|---|
| 最容易写成“又一个融合方法”，但实际是系统级输入仲裁 | S1–S2 | 开篇从“同一融合图在 12 个 checkpoint 上出现正负反转”切入；贡献按“失效发现 → 风险受控仲裁 → 三数据集确认”组织 | 高 |
| CS-ECF 不是单融合网络，必须明确部署边界 | S1–S2 | 方法节定义候选集合、目标评估器、`q_c`、shield 和推理流程；标题与摘要不得声称单图融合 | 高 |
| LLVIP 的三种 95% 区间均跨零 | S11 | 主表同时报告 LLVIP、M3FD、DroneVehicle 与三数据集等权；不得只报合并正值 | 高 |
| M3FD 增益集中在 `people` | S10–S11 | 预先定义 people 为主分析类别；六类全表中同时报告其余五类，不写成普遍提升 | 高 |
| Always CDF 是最强反例 | S11–S12 | 主表加入 Always CDF；说明 CS-ECF 的收益来自 M3FD 与尾部控制，而不是全面支配 | 高 |
| no-shield 与 shield 呈现风险—收益交换 | S11 | 报告最差值、分布和高增益损失；不能只报均值 | 高 |
| 验证集只有四个评估器，已有两种超参候选在测试集失败 | S11 | 保留 v2/v3 负例；DroneVehicle 独立确认已完成，不再用当前 test 调参 | 高 |
| 四候选推理成本未完整测量 | S11 | 测候选生成、四遍检测前向、总延迟和 GPU 峰值显存；与 Always CDF/Visible 同设置 | 高 |
| DroneVehicle 的 best/last 同训练 | S10 | 把独立官方 test 放主表，同时明确两个 checkpoint 不是独立架构 | 高 |
| Highlights 还没有按条件结论收紧 | S3 | 用数值写“尾部风险下降”和“数据集等权增益”，避免“全面提升” | 中 |
| 框架图尚未按 CS-ECF 重画 | S5–S8 | 图 1 为评估器反转；图 2 为四候选与 shield；图 3 为选择分布、尾部风险和类别效应 | 中 |
| Introduction 文献需要覆盖 evaluator dependence、fusion evaluation 与 risk control | S9 | 近三年工作优先；每条文献核对真实性与上下文 | 中 |
| 已选 IPT，但红外技术贡献仍需补强 | S15 | 主轴改为红外热信息的检测效用；补热对比度、掩膜、昼夜与案例，不复用旧 CDF 稿件 | 高 |

## 4. 可直接使用的贡献结构

1. **现象**：在相同可见输入和融合候选上，检测效用随评估器改变，固定融合输入
   的全局排序不稳定。
2. **方法**：提出 CS-ECF，用目标评估器自身的 Top-5 置信度对候选做输入仲裁，
   并用 1.5 倍可见图安全阈值抑制低可靠切换。
3. **证据**：在 LLVIP 8 个评估器、M3FD 2 个六类检测器和 DroneVehicle 2 个
   OBB checkpoint 上报告模型、图像/组和分层 bootstrap；不隐藏 LLVIP 跨零和
   M3FD 的 people-class concentration。
4. **边界**：说明该方法是 evaluator-conditioned system，计算代价高于单输入，
   当前结论限于多模态目标检测，不推广到通用图像质量。

## 5. 结果主表

### Table 1 数据与独立层级

| Dataset | Confirmatory status | Images | Classes | Groups |
|---|---|---:|---:|---|
| LLVIP clean test | 是 | 300 | 1 | 7 scenes |
| M3FD grouped test | 是 | 200 | 6 | 100-frame groups |
| DroneVehicle official test | 是 | 300 | 1 OBB vehicle | 3 fixed groups [7,32,71] |

### Table 2 Evaluators

- LLVIP：YOLO11n、modality-agnostic、YOLO26n seeds 0/1/2、RT-DETR-l、
  YOLOv8n seeds 0/1；
- M3FD：YOLO26n、YOLOv8n；
- DroneVehicle：YOLO11x-OBB best/last（同训练）；
- 明确权重来源、训练轮数、种子、图像尺寸、置信度阈值和 checkpoint hash。

### Table 3 主结果

| Policy | Positive checkpoints | Mean gain | Dataset-balanced | Worst |
|---|---:|---:|---:|---:|
| Always CDF | 10/12 | +0.0052 | +0.0031 | -0.0059 |
| Top-5 no shield | 7/12 | +0.0132 | +0.0332 | -0.0584 |
| CS-ECF | 9/12 | +0.0074 | +0.0133 | -0.0040 |

CS-ECF 的三数据集等权 model×group 分层区间：`[+0.0028,+0.0213]`。  
model×image 分层区间为 `[+0.0028,+0.0256]`；主分析采用更保守的 group 版本。  
LLVIP 的 model/image 区间仍跨零，必须显示。  
M3FD 总体为正，但 M3FD 增益集中在 `people`。

DroneVehicle 独立 test：

| Checkpoint | Visible | CS-ECF | Shield gain | No-shield gain |
|---|---:|---:|---:|---:|
| best | 0.6563 | 0.6814 | +0.0251 | +0.1064 |
| last | 0.6474 | 0.6767 | +0.0293 | +0.1016 |

DroneVehicle image CI：`[-0.0028,+0.0682]`；group CI：`[+0.0002,+0.0406]`。
三个组对两个 checkpoint 均为非负，但其中一组仅 `+0.0002`。

### Table 4 Shield 风险分析

- 切换图像比例；
- 每个候选被选择的图像数；
- no-shield 与 shield 的逐模型差值；
- 最差模型、最差图像组、增益分布；
- shield 对 modality-agnostic 的 `-0.0282` 损失。

### Table 5 类别效应

M3FD 必须报告六类 AP50：

- people：YOLO26n `+0.0214`，YOLOv8n `+0.1090`；
- car、bus、motorcycle、lamp、truck：基本不变或仅有 `0.0003` 量级变化；
- people 的 YOLO26n image CI 跨零：`[-0.0070,+0.0498]`；
- people 的 YOLOv8n image CI 为正：`[+0.0440,+0.1913]`。

### Table 6 Complexity

| Component | Latency | Extra peak memory | Notes |
|---|---:|---:|---|
| CDF pipeline | 22.3 ms | 待测 | 384×512，含 pixel-mean + IFCNN + postprocess |
| PIAFusion | 37.7 ms | 869.1 MiB | FP32，batch=1 |
| SPGFusion | 516.3 ms | 1453.5 MiB | FP32，batch=1 |
| 单次 YOLO11x-OBB | 32.5 ms | 115.6 MiB | 640 输入 |
| 四次检测前向 | 130.0 ms | 非相加 | 串行计算，未使用 batch=4 |
| CS-ECF 串行估计 | 706.3 ms | 非相加 | 候选生成 + 四次检测 |

该 706.3 ms 不包含模型换入换出、数据加载和最终检测后处理；per-stage memory
不能直接相加。主文必须报告这些边界，不能只引用 CDF 在 1024×768 上的 101 ms。

## 6. 图规划

### Figure 1 问题

同一图像的四个候选，在不同评估器上的 AP50 gain 与选择方向反转。突出
evaluator-conditioned utility，而不是再展示一组融合视觉对比。

### Figure 2 方法

`VIS + IR → 4 candidates → target detector → Top-5 confidence → 1.5× shield →
selected input → final detection`。图上必须标“detector reused at inference”。

### Figure 3 风险与收益

左侧：no-shield 与 CS-ECF 的逐模型 gain 点线图；右侧：最差退化与平均增益的
交换。使用统一零点、统一 AP 单位，不截断坐标。

### Figure 4 类别效应

M3FD 六类 gain 热图或分组点图；people 与其他五类分开标注。该图决定论文能否
诚实支持 pedestrian-centric 主张。

## 7. Highlights 工作稿

1. 融合输入的最优选择随下游检测器而反转。
2. CS-ECF 用检测器置信度进行带安全阈值的输入仲裁。
3. 安全阈值将最差 AP50 退化从 5.84 降至 0.40 点。
4. 三数据集等权 AP50 增益为 1.33 点，12 个 checkpoint 中 9 个为正。
5. M3FD 增益集中于 people，而非均匀覆盖六个类别。

第 4、5 条不是传统“全是优势”的 Highlights，但它们与现有证据一致。若目标期刊
不接受 limitation-style highlight，可把第 5 条移到摘要和讨论，前四条不得改写成
“六类全面提升”。

## 8. Abstract 逻辑

1. 说明多模态融合通常被当作单一最佳输入，但下游检测器会改变其效用；
2. 提出四候选、目标检测器 Top-5 置信度、1.5 倍可见图 shield；
3. 报告 12 个测试 checkpoint、9 个为正、最差 `-0.0040`；
4. 报告三数据集等权 `+0.0133` 及 group 分层区间 `[+0.0028,+0.0213]`；
5. 明确 LLVIP 区间跨零、收益主要在 pedestrian/people；
6. 报告 DroneVehicle 独立 test 的 `+0.0251/+0.0293`；
7. 结论限定为 evaluator-conditioned detection input arbitration。

## 9. 二区送审硬门槛

以下条件没有全部满足，不应生成新的二区 submission zip：

1. **独立确认：已完成**。DroneVehicle 官方 test 使用预先固定的 3 个连续帧组，
   best/last 两个 checkpoint 均正；必须保留 manifest、源归档大小和选择种子；
2. **非单点依赖：条件通过**。当前 DroneVehicle 只有同一训练运行的 best/last，
   仍需补一个独立 OBB 架构或明确写“dataset-level confirmation”；
3. **方向稳定**：至少 2/3 评估器为正，分层区间下界不为负，或明确把论文限定为
   尾部风险控制；
4. **类别预先声明**：如果仍以 people 为主，必须在评估前固定主分析类别，并报告
   其余类别；
5. **强基线完整**：Always CDF、Always PIA/SPG、Top-5 no-shield、CS-ECF 同协议；
6. **成本：首轮完成**：已测 384×512 串行组件与 640 检测输入，估计
   `706.3 ms`；仍需决定主文是否补充模型常驻时的端到端 wall time；
7. **可复现**：所有数字由脚本生成，保留结果路径、随机种子和 split manifest。

## 10. 期刊定位

**目标期刊已确定为 Infrared Physics & Technology（IPT）。**

这一选择的主轴不是“测量效度”，而是：

> 红外热信息在低光目标检测中的任务效用随检测器改变；CDF 提取热有效细节，
> CS-ECF 则决定何时让热融合候选替代可见图。

IPT 适配风险：

- 当前方法层仍是 evaluator-conditioned input arbitration，容易被视为 CV
  application；
- 若正文只报告 AP 与 bootstrap，没有热对比度、热掩膜或昼夜条件分析，editor
  可能认为红外物理/技术贡献不足；
- SPGFusion 是主要计算瓶颈，系统成本必须放在正文，不可隐藏在附录。

IPT 版必须补：

1. **已完成首轮**：热对比度对切换行为的分层分析；结果是 LLVIP/M3FD 正相关，
   DroneVehicle 反向，不能写成“热对比度越高越该融合”；
2. **已完成首轮**：CDF 热有效性掩膜消融；DroneVehicle 固定输入下 mask 比
   global injection 平均高 `+0.0007 AP50`，效应一致但较小；
3. 昼夜/低可见度条件分层；
4. 红外—可见—CDF—PIA—SPG 的成功与失败案例；
5. 标准化输入与真实传感器分辨率下的成本说明。

IEEE Sensors Journal 保留为备选：它更接受 sensing-system arbitration，但
论文主轴需从热成像改成多传感器系统。

## 11. 结果文件

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
- `ssgf/results/selective_fusion/cs_ecf_sota_runtime.json`
- `ssgf/results/selective_fusion/cs_ecf_detector_runtime.json`
- `ssgf/results/selective_fusion/cs_ecf_cdf_runtime_512.json`
- `ssgf/results/selective_fusion/ipt_thermal_stratification.json`
- `ssgf/results/selective_fusion/ipt_mask_ablation.json`
- `D:\Code\_ssgf_external\dronevehicle_test300`

