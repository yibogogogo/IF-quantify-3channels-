# IF-quantify-3channels-

免疫荧光三通道定量分析工具 — 全场比值法 v2

用于甩片制片的免疫荧光图像定量分析。

## 原理

```
目的基因平均表达 = (全场目标总荧光 − 背景) / DAPI 总核数
```

所有细胞为 CD138 分选阳性，因此全部 DAPI 核均计入分母（根据研究目的请使用agent自行修改和定义）

## 核分割方法

| 方法 | 命令 | 精度 | 速度 | 依赖 |
|------|------|------|------|------|
| **multi-Otsu**（默认） | `python if_quantify.py` | 中等 | 最快 | numpy, scipy, skimage |
| **StarDist** | `--stardist` | 高 | ~30s/视野 CPU | tensorflow, stardist |
| **Cellpose** | `--cellpose` | 高 | 慢 (需GPU) | torch, cellpose |
| **Ensemble** | `--ensemble` | 最高 | 两者并行 | 以上两者 |

> Ensemble 模式同时运行 StarDist + Cellpose
> 注意虽然集成了cellpose，但是太慢了（本人没有GPU所以没试过加速），建议直接stardist

## 数据要求

### 目录结构

```
当前目录/
├── if_quantify.py           ← 脚本放在这里
├── 样本A/                   ← 样本文件夹
│   ├── New-01.jpg           ← 合并图
│   ├── New-01.jpg_files/    ← 单通道图
│   │   ├── *c0*.jpg         ← 通道0 (默认: 594, CD138)
│   │   ├── *c1*.jpg         ← 通道1 (默认: 488, 目的基因)
│   │   └── *c2*.jpg         ← 通道2 (默认: DAPI)
│   ├── New-02.jpg
│   └── ...
├── 样本B/
└── ...
```

### 通道配置

默认通道映射（蔡司 Zen 导出 JPG）：

| Zen 通道 | 荧光 | 标记 | RGB 提取 |
|---------|------|------|---------|
| c0 | 594 nm | CD138 | R 通道 |
| c1 | 488 nm | 目的基因 | G 通道 |
| c2 | DAPI | 细胞核 | B 通道 |

## 安装依赖

```bash
# 基础依赖
pip install numpy Pillow scipy scikit-image

# StarDist（DL 核分割）
pip install stardist tensorflow

# Cellpose（备选 DL 模型，推荐 GPU）
pip install cellpose

# GPU 加速（NVIDIA）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# GPU 加速（AMD/Intel 待 DirectML 生态成熟）
```

## 安装

```bash
# 方式1：pip 安装
pip install if-quantify@git+https://github.com/yibogogogo/IF-quantify-3channels-.git@v2.0.0

# 方式2：直接克隆
git clone https://github.com/yibogogogo/IF-quantify-3channels-.git
pip install -r requirements.txt
```

## 使用方法

```bash
# 默认 multi-Otsu（最快）
python if_quantify.py

# StarDist 深度学习
python if_quantify.py --stardist

# 调整严格度
python if_quantify.py --stardist --prob-thresh 0.85   # 更严格
python if_quantify.py --stardist --prob-thresh 0.70   # 更宽松

# 双模型集成
python if_quantify.py --ensemble

# 样本校准（每样本一个GT，自动二分搜索最优阈值，最合理准确）
python if_quantify.py --stardist --calibrate calibrations.txt

# 并行加速（3视野并行）
python if_quantify.py --stardist --parallel 3

# 测试模式
python if_quantify.py --test

# 指定工作目录 / 单样本
python if_quantify.py --dir D:\experiment\data
python if_quantify.py --sample HD-HCZ

# 预览扫描
python if_quantify.py --dry-run
```

## 校准流程（新数据首次使用）

每样本人工计数一个视野的细胞核数（注意文件夹名称同步），写入 `calibrations.txt`：

```ini
# calibrations.txt
<文件夹名称>=745
<文件夹名称>=249
<文件夹名称>=57
```

运行校准 + 全量分析：

```bash
python if_quantify.py --stardist --calibrate calibrations.txt --parallel 3
```

脚本自动二分搜索每个样本的最优 prob_thresh，然后全量分析。

## 输出文件

```
_analysis_results/
├── if_quant_detail.csv      ← 每个视野详细结果
└── if_quant_summary.csv     ← 样本汇总统计（加权平均）
```

### CSV 列说明

| 列名 | 含义 |
|------|------|
| `mean_per_cell` | **目的基因平均荧光/细胞** ← 主要指标 |
| `n_nuclei` | DAPI 核计数（分母） |
| `n_sd` | StarDist 单独核数（ensemble 模式） |
| `n_cp` | Cellpose 单独核数（ensemble 模式） |
| `bg_mean` | 背景荧光 **p25 百分位**（四边框采样，排除核区域） |
| `total_raw` | 全场总荧光（原始） |
| `total_corrected` | 全场总荧光（背景校正后） |
| `cd138_qc_pct` | CD138 染色质控 % |
| `cd138_mean` | CD138+ 区域内均值（参考） |

> 背景校正使用 p25 百分位——比中位数更不受荧光碎屑影响。

## 参数调优

编辑脚本顶部配置区：

```python
NUCLEUS_SIZE_MIN = 200      # 最小核面积（像素²）
NUCLEUS_SIZE_MAX = 80000    # 最大核面积
CIRCULARITY_MIN = 0.1       # 最低圆形度
BACKGROUND_BORDER = 30      # 背景采样边框宽度
PROB_THRESH = 0.78          # StarDist 默认概率阈值
```

## 科学严谨性说明

1. **核计数**：multi-Otsu 用 3 类 Otsu 分离暗背景/弱信号/亮核；StarDist/Cellpose 用预训练 CNN 纠正过度分割
2. **背景校正**：四边框采样（排除核区域）取 **p25 百分位**，比均值和中位数更抗碎屑干扰
3. **全场比值法**：不依赖单细胞分割质量，用总荧光/总核数，对密集核场景稳健，但是注意需要选择较好染色的图来定量否则误差大

## 依赖

- Python ≥ 3.8
- numpy, Pillow, scipy, scikit-image
- 可选：tensorflow + stardist, torch + cellpose
