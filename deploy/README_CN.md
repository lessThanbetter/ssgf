# SSGF 新服务器交付包

这个目录应解压为：

```text
/root/autodl-tmp/ssgf
```

它包含当前代码、模型权重、聚合结果、逐图检测统计、完整 reference VIF 结果、
配准敏感性分析、论文和部署脚本。当前投稿包装目标为 Measurement，成品位于
`paper/measurement_submission/`。公开数据集和 1.9 GB 的 LLVIP 融合图不在核心包中，
避免重复上传。

## 1. 首次检查

```bash
cd /root/autodl-tmp/ssgf
cp deploy/server.env.example .env
# 按新机器实际情况修改 .env
bash deploy/bootstrap.sh
```

`bootstrap.sh` 检查 Python、PyTorch、SciPy、OpenCV、CUDA 和关键权重，并运行
reference VIF 的自测。它不会访问旧服务器，也不会修改公开数据集。

如需建立独立虚拟环境：

```bash
bash deploy/bootstrap.sh --install
```

完整复跑 SPGFusion 时额外安装：

```bash
pip install -r requirements-experiments.txt
```

## 2. 数据目录

默认约定：

```text
/root/autodl-tmp/datasets/
  det/       # 300 个 LLVIP source pairs 和 fused outputs
  tno/       # 74 个 source pairs
  msrs/      # 361 个 MSRS-test pairs
  roadscene/ # 221 个 source pairs
```

实际目录不同不必移动数据，只需在作业 JSON 中填写绝对路径。数据集下载入口见
`../README.md` 和数据处理脚本。

## 3. 生成三套融合基准输入

本项目四种规则及 IFCNN、DenseFuse 可由一个命令生成，并统一限制到 512 px：

```bash
python -m sota_runners.generate_fusion_outputs \
  --root <MSRS-test> --vis-dir vi --ir-dir ir \
  --out-root outputs/msrs \
  --methods mean tophat probe ensemble ifcnn densefuse
```

PIAFusion 和 SPGFusion 使用各自官方仓库与权重。外部权重的固定 URL、大小和 SHA-256
见 `EXTERNAL_ASSETS.md`；三个下载脚本都支持断点续传：

```bash
python -m sota_runners.download_spgfusion_weights
python -m sota_runners.download_clip_weights
python -m sota_runners.download_dino_weights
```

## 4. 补算 reported / reference VIF

复制示例配置并填好每个数据集的 visible、infrared 和八个 fused 目录：

```bash
cp deploy/reference_jobs.example.json deploy/reference_jobs.json
python analysis/run_reference_vif_parallel.py \
  --config deploy/reference_jobs.json --workers 8
```

每个方法使用独立子进程，已完成方法会自动跳过；输出为每个数据集一个 `.json`
和一个 `.csv`。也可以先用顺序版本检查配置：

```bash
python deploy/run_reference_jobs.py \
  --config deploy/reference_jobs.json --outdir results/reference_vif_check
```

冻结结果已在 `results/reference_vif/`：三套 fidelity benchmark 和完整 300 张
LLVIP 表的 reported/reference 数值均可直接检查。

## 5. 配准敏感性

以下命令只从 LLVIP ZIP 中抽取 300 个分析 stem，随后估计 ECC 平移残差并构造
best/worst quartile：

```bash
python -m analysis.extract_llvip_sources \
  --stems _deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_definitions.csv \
  --out <llvip-source-subset>
python -m analysis.estimate_alignment \
  --sources <llvip-source-subset> --out results/alignment/alignment.json
python -m analysis.alignment_sensitivity \
  --alignment results/alignment/alignment.csv \
  --regime-csv _deposit/ssgf-llvip-fusion-detection-v1/data/aggregate/regime_definitions.csv \
  --per-image _deposit/ssgf-llvip-fusion-detection-v1/data/per_image \
  --out results/alignment/alignment_sensitivity.json \
  --evaluators yolo11n_vis agnostic --bootstrap 1000
```

冻结结果、四 seed 范围和解释见 `results/alignment/README.md`。结果显示对齐质量与
场景内容共同影响负差值，不能把 pooled 结论表述成对所有对齐子集都成立。

## 6. LLVIP 现有结果复验

无需重新训练即可检查 clean-split 主结果：

```bash
python scripts/verify_clean_r2.py --root .
```

该检查验证 1,350/150/300 划分、零重叠、744 个测试标注、四个 clean evaluator、
两个单 run 检测器的 16 个输入、14 项 BH 家族、三个 84 项分层家族以及训练清单和
checkpoint hash。冻结结果在 `results/clean_r2/`，完整划分 stem 在
`_deposit/ssgf-llvip-fusion-detection-v1/data/splits/clean_split_manifest.json`。

四 seed YOLO11n 表的独立复验仍可运行：

```bash
python _clean_split_seed_ranges.py
python _r5_seeds.py --family vis_split --out results/vis_split_bootstrap.json
python _r5_seeds.py --family agn_split --out results/agn_split_bootstrap.json
```

## 7. 不要提交的文件

旧服务器地址、密码、SSH 私钥、数据集压缩包和训练输出不进入交付包。新机器的访问
信息请通过本地 `upload_and_run.ps1` 参数或 `~/.ssh/config` 提供。
