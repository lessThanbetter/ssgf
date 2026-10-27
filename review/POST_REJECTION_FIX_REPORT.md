# 拒稿后返修闭环报告

## 当前交付状态

- 主文：23 页，11 图，13 表，摘要 246 词，参考文献 47 条。
- 编译：0 个 LaTeX 错误，0 个未定义引用，0 个超过 10 pt 的 overfull box，无
  `??`。
- 补充材料：37 页，29 张表，0 个编译错误，0 个未定义引用；新增全量 VIF、
  配准敏感性、双口径 multiplicity、clean 全 evaluator 表和 clean 分层 BH 结果。
- 数据包：`ssgf_server_core_v4.tar.gz` 从独立解压目录通过 1198 项 manifest 校验；
  deposit 的 470 项 manifest 全部通过；5100 张融合图的归档 hash 保持不变。
- 当前主文与补充材料：`paper/PAPER_MEASUREMENT_v4.pdf`、
  `paper/Supplementary_Material_MEASUREMENT.pdf`；门户上传文件在
  `paper/measurement_submission/`。

## 已解决的关键审稿问题

1. **R2-2 已闭环。** 第 4 节的主结果现在覆盖 clean YOLO11n、YOLO26n、
   RT-DETR-l 三套 visible-only evaluator，以及 clean modality-agnostic YOLO11n。
   1,350/150/300 train/validation/test 划分的 stem 两两交集均为 0。
2. **图 2 的层级改为“干净可见光—干净模态无关—乐观池化审计”。** 42/42
   只在第三块作为敏感性分析出现，不再冒充主结果。
3. **口径和数值统一。** 三套 clean visible-only evaluator 共 42/42 个输入对比
   均为负，范围 −0.046 至 −0.246 mAP50；YOLO26n 与 RT-DETR-l 各 14/14 在
   BH 后 q<0.002。Clean modality-agnostic 侧有 4 个稳定正收益，均值 0.022–0.029；
   pooled audit 保留 0.014–0.152 及 42/42，并明确降级为敏感性分析。
4. **M4/R2-3 的保真度—检测关系可直接核对。** 表 8 在完整 300 张 LLVIP 图像上
   给出全部 14 个输入的 reported VIF、reference VIF、pooled 对比和 clean-split
   对比。TNO、MSRS、RoadScene 的 8 方法全量结果在 `results/reference_vif/`。
5. **M11 的归一化不再只靠论证。** 正文给出 TNO 与 MSRS 在四个温度下的
   per-image mean absolute weight deviation、cross-image SD 和对应检测结论。
6. **作者自己的规则不占便宜。** 干净划分中 top-hat 与 probe 同样输给 visible
   source；正文同时承认 activity-only 在两个 pooled 检测器上更强。
7. **R2-1、R3 的结构与图问题。** 图 5 的 SSIM 描述已修正；图 2 已重排，图 11
   已重绘为矢量科研示意图，使用可见光、红外和融合实例缩略图连接评价器与指标两条
   判据；主文当前为 23 页，表格按信息层级保留。

8. **M1/M12/M15 现在可直接核对。** 主文 Table 6 给出每个输入在 YOLO11n、
   YOLO26n 和 RT-DETR-l 下的 single-channel/colour 两行结果，并把两套 42 检验的
   BH q 值直接印在单元格中。single-channel 只有 SPGFusion 的三项达到 q<0.05，
   colour 有 40/42 达到 q<0.05。

9. **M5 已有校正后的分层结果。** 每个可见训练架构分别对 14 个输入乘 6 个重叠
   分层做 84 检验 BH。高分位对比与小目标在三个架构均为 14/14；拥挤分层为
   14/14、14/14、5/14；最低对比三分位为 2/14、2/14、0/14，RT-DETR-l 中另有
   1 个 activity-only 对比在上方。

10. **M6/M16/M18 已补到主文。** 六个种子对比逐 seed 列出；成本表给出参数、
    延迟和额外峰值显存，统一在 RTX 2080 Ti、512×384、FP32、batch 1 上测量。

11. **Measurement 投稿包装已完成。** 标题与摘要改为测量有效性定位，新增五项
    Highlights、cover letter 和独立上传目录 `paper/measurement_submission/`。

## 本轮新增闭环

1. **R2-3 已完成。** 14 个 LLVIP 输入和 TNO/MSRS/RoadScene 8 个方法均在完整
   数据上重算。reported 指数下 SPGFusion 三套基准均为第一；reference VIF 下均为
   最后。冻结 JSON/CSV 和复现脚本已进入交付包。
2. **M19 已完成可执行的敏感性分析。** 从公开 LLVIP ZIP 精确抽取 300 对源图，
   ECC 平移残差中位数 2.53 px。最好四分位中 mean/top-hat/probe 在四个 clean-split
   run 中均为正；最差四分位负差值最大。配准质量与对比度、目标大小共变，因此不能
   把结果解释成纯粹的配准因果效应。
3. **原结论已修正。** 300 张 pool 上的负差值仍成立，但不能表述为跨所有对齐子集
   不变；摘要、结果、讨论、局限和结论均已加入这一限定。

## 投稿判断

这版消除了“表文不一致”“主结果完全依赖重叠验证集”和“全量 reference VIF 缺失”
三个硬问题。当前包装目标为 **Measurement**，贡献定位为融合规则评测的测量有效性，
而不是新的融合网络。

1. **R3 的“缺乏核心创新”仍未闭环。** 当前贡献是 measurement/benchmark 协议
   与证据链，不是新融合网络。继续投 IPT 需要新的方法贡献；更匹配的是 IEEE TIM、
   Measurement 等接受测量与评测方法贡献的期刊。

作者已于 2026-10-27 确认 Zenodo DOI 可公开访问，并确认作者信息、基金、利益冲突
和 AI 使用声明无误。除 R3 的核心创新性意见外，当前返修清单中的审稿问题均已有
对应的正文、补充材料或数据证据。
