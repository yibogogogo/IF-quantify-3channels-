# IF-quantify-3channels-

免疫荧光三通道定量分析工具 — 全场比值法

用于**CD138分选阳性细胞**甩片制片的免疫荧光图像定量分析。

## 原理

```
TTLL12 平均表达 = (全场 488 总荧光 − 背景) / DAPI 总核数
```

所有细胞为 CD138 分选阳性，因此全部 DAPI 核均计入分母。
CD138 通道仅用于染色质控，不用于细胞身份判定。

## 数据要求

### 目录结构

```
当前目录/
├── if_quantify.py           ← 脚本放在这里
├── 样本A/                   ← 样本文件夹
│   ├── New-01.jpg           ← 合并图
│   ├── New-01.jpg_files/    ← 单通道图
│   │   ├── *c0*.jpg         ← 通道0 (默认: 594/CD138)
│   │   ├── *c1*.jpg         ← 通道1 (默认: 488/TTLL12)
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
| c1 | 488 nm | TTLL12 | G 通道 |
| c2 | DAPI | 细胞核 | B 通道 |

如需更改，编辑脚本顶部 `CHANNEL_MAP` 字典即可。

## 安装依赖

```bash
pip install numpy Pillow scipy scikit-image
```

## 使用方法

```bash
# 全量分析（当前目录）
python if_quantify.py

# 测试模式（每样本只处理前2个视野，快速验证）
python if_quantify.py --test

# 分析指定目录
python if_quantify.py --dir D:\experiment\data

# 只分析某个样本
python if_quantify.py --sample Sample01

# 仅预览扫描结果
python if_quantify.py --dry-run
```

如果不想每次都输 `python`，可以双击运行（需要 `.py` 关联到 Python）。

## 输出文件

```
_analysis_results/
├── if_quant_detail.csv      ← 每个视野的详细结果
└── if_quant_summary.csv     ← 每样本的汇总统计（加权平均）
```

### CSV 列说明

| 列名 | 含义 |
|------|------|
| `mean_per_cell` | **TTLL12 平均荧光/细胞** ← 主要指标 |
| `n_nuclei` | DAPI 核计数（分母） |
| `bg_mean` | 背景荧光均值（四边框采样） |
| `total_raw` | 全场总荧光（原始） |
| `total_corrected` | 全场总荧光（背景校正后） |
| `cd138_qc_pct` | CD138 染色质控 % |
| `cd138_mean` | CD138+ 区域内均值（参考） |

## 参数调优

编辑脚本顶部配置区：

```python
NUCLEUS_SIZE_MIN = 200      # 最小核面积（像素²）
NUCLEUS_SIZE_MAX = 80000    # 最大核面积
CIRCULARITY_MIN = 0.1       # 最低圆形度
BACKGROUND_BORDER = 30      # 背景采样边框宽度
```

## 依赖

- Python ≥ 3.8
- numpy
- Pillow
- scipy
- scikit-image
