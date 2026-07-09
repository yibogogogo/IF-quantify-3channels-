#!/usr/bin/env python3
"""
免疫荧光定量分析工具
====================
CD138 分选阳性细胞的 TTLL12 表达定量（全场比值法）

使用方法：
  1. 将此脚本放在与数据文件夹同级的目录中
  2. 确保数据目录结构为：
     当前目录/
       └── 样本文件夹/
            ├── New-01.jpg
            ├── New-01.jpg_files/
            │   ├── *c0*.jpg   (通道0)
            │   ├── *c1*.jpg   (通道1)
            │   └── *c2*.jpg   (通道2)
            └── ...
  3. 运行:  python if_quantify.py

实验背景：
  所有细胞为 CD138 分选阳性后甩片制片。
  定量公式: TTLL12 平均表达 = 全场 488 总荧光 / DAPI 总核数

安全约束：只读取文件，绝不修改或删除任何文件。
"""

import os, sys, re, csv, logging
import numpy as np
from PIL import Image
from scipy import ndimage as scipy_ndimage
from skimage import filters, morphology, measure, segmentation, feature

# ============================================================================
#  ▸▸▸ 配置区 ◂◂◂  按需修改以下参数
# ============================================================================

# ---- 目录设置 ----
# 脚本自动检测所在目录作为工作目录。也可手动指定：
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出目录（相对于 WORK_DIR）
OUTPUT_SUBDIR = "_analysis_results"

# ---- 通道配置 ----
# 蔡司 Zen 导出 JPG 的通道编号 → 荧光染料/标记
# 格式: {"Zen通道号": ("标签", "RGB通道提取")}
CHANNEL_MAP = {
    "c0": ("CD138",  "R"),   # 通道0: 594nm, CD138,  从RGB的R通道提取
    "c1": ("TTLL12", "G"),   # 通道1: 488nm, TTLL12, 从RGB的G通道提取
    "c2": ("DAPI",   "B"),   # 通道2: DAPI,  细胞核, 从RGB的B通道提取
}
# 以后换染料/通道顺序，只需改上面这三行！

# ---- 核分割参数 ----
NUCLEUS_SIZE_MIN   = 200     # 最小核面积（像素^2），小淋巴细胞核约200-500
NUCLEUS_SIZE_MAX   = 80000   # 最大核面积，大浆细胞核可达数万
CIRCULARITY_MIN    = 0.1     # 最低圆形度 (0~1)，越低越允许不规则核

# ---- 背景采样 ----
BACKGROUND_BORDER  = 30      # 从图像四边取 N 像素宽的边框作为背景

# ---- 测试模式 ----
TEST_FIELDS        = 2       # --test 时每样本只处理前 N 个视野

# ---- 文件匹配 ----
FIELD_PATTERN      = r"New-(\d+)\.jpg$"

# ============================================================================
#  配置结束，以下代码通常无需修改
# ============================================================================

OUTPUT_DIR = os.path.join(WORK_DIR, OUTPUT_SUBDIR)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("IF_Quant")

# 从 CHANNEL_MAP 构建反向映射
CH_BY_NAME = {v[0]: (k, v[1]) for k, v in CHANNEL_MAP.items()}


def scan_samples(work_dir: str) -> list:
    """扫描所有样本文件夹（不以下划线开头、不以 _files 结尾的目录）"""
    return sorted([
        os.path.join(work_dir, e) for e in sorted(os.listdir(work_dir))
        if os.path.isdir(os.path.join(work_dir, e))
        and not e.startswith("_") and not e.endswith("_files")
    ])


def scan_fields(sample_dir: str) -> list:
    """扫描样本内所有视野，返回 [(视野号, dapi路径, ttll12路径, cd138路径)]"""
    dapi_key,   dapi_ch   = CH_BY_NAME["DAPI"]
    ttll12_key, ttll12_ch = CH_BY_NAME["TTLL12"]
    cd138_key,  cd138_ch  = CH_BY_NAME["CD138"]
    ch_keys = {dapi_key: None, ttll12_key: None, cd138_key: None}

    fields = []
    for entry in sorted(os.listdir(sample_dir)):
        m = re.match(FIELD_PATTERN, entry)
        if not m:
            continue
        fid = m.group(1)
        fdir = os.path.join(sample_dir, entry + "_files")
        if not os.path.isdir(fdir):
            continue
        found = dict(ch_keys)
        for f in sorted(os.listdir(fdir), key=len):
            if not f.endswith(".jpg") or "_metadata" in f:
                continue
            mch = re.search(r'(?:^|_)t?\d*c(\d)', f)
            if mch and f"c{mch.group(1)}" in found:
                if found[f"c{mch.group(1)}"] is None:
                    found[f"c{mch.group(1)}"] = os.path.join(fdir, f)
        if not all(found.values()):
            continue
        fields.append((fid, found[dapi_key], found[ttll12_key], found[cd138_key]))
    return sorted(fields, key=lambda x: int(x[0]))


def extract_channel(jpg_path: str, rgb_ch: str = "B") -> np.ndarray:
    """从 RGB JPG 提取单通道灰度"""
    try:
        arr = np.array(Image.open(jpg_path), dtype=np.uint8)
    except Exception as e:
        logger.error(f"  读取失败 {os.path.basename(jpg_path)}: {e}")
        return None
    if arr.ndim == 2:
        return arr.astype(np.float64)
    return arr[:, :, {"R": 0, "G": 1, "B": 2}[rgb_ch.upper()]].astype(np.float64)


def segment_nuclei(dapi_img: np.ndarray) -> np.ndarray:
    """DAPI 核分割 -> 返回 label_mask"""
    blr = filters.gaussian(dapi_img, sigma=2.0)
    otsu = filters.threshold_otsu(blr)
    bin_ = blr > otsu
    op = morphology.opening(bin_, morphology.disk(2))
    cl = morphology.remove_small_objects(op, max_size=50)
    cl = morphology.closing(cl, morphology.disk(3))

    dist = scipy_ndimage.distance_transform_edt(cl)
    peaks = feature.peak_local_max(dist, min_distance=10,
                                    exclude_border=3, labels=cl)
    if len(peaks) == 0:
        return np.zeros(dapi_img.shape, dtype=np.int32)
    markers = np.zeros(dapi_img.shape, dtype=np.int32)
    for i, (y, x) in enumerate(peaks):
        markers[y, x] = i + 1
    labs = segmentation.watershed(-dist, markers, mask=cl)

    out = np.zeros_like(labs)
    for p in measure.regionprops(labs):
        if p.area < NUCLEUS_SIZE_MIN or p.area > NUCLEUS_SIZE_MAX:
            continue
        circ = (4 * np.pi * p.area) / (p.perimeter**2) if p.perimeter > 0 else 0
        if circ < CIRCULARITY_MIN:
            continue
        out[labs == p.label] = p.label
    return out


def measure_background(img: np.ndarray, nuclei_mask: np.ndarray) -> float:
    """从图像四边采样背景"""
    h, w = img.shape
    b = BACKGROUND_BORDER
    border = np.zeros((h, w), dtype=bool)
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
    bg = border & (nuclei_mask == 0)
    return img[bg].mean() if bg.sum() > 100 else img[border].mean()


def process_field(dapi_path: str, ttll12_path: str, cd138_path: str) -> dict:
    """处理单个视野"""
    dapi   = extract_channel(dapi_path,   CH_BY_NAME["DAPI"][1])
    ttll12 = extract_channel(ttll12_path, CH_BY_NAME["TTLL12"][1])
    cd138  = extract_channel(cd138_path,  CH_BY_NAME["CD138"][1])
    if any(x is None for x in [dapi, ttll12, cd138]):
        return None

    nuclei_mask = segment_nuclei(dapi)
    n_nuc = len(np.unique(nuclei_mask)) - 1
    if n_nuc == 0:
        return None

    # ---- 全场定量（主指标） ----
    bg_mean = measure_background(ttll12, nuclei_mask)
    total_raw = ttll12.sum()
    total_corrected = max(total_raw - bg_mean * ttll12.size, 0)
    mean_all = total_corrected / n_nuc

    # ---- CD138 染色质控 ----
    flt = filters.median(cd138, morphology.disk(2))
    thr = filters.threshold_otsu(flt)
    cd138_mask = morphology.closing(flt > thr, morphology.disk(8))
    cd138_mask = scipy_ndimage.binary_fill_holes(cd138_mask)
    cd138_mask = morphology.remove_small_objects(cd138_mask, max_size=500)

    cd138_pct = 0.0
    n_in_cd138 = 0
    mean_cd138 = 0.0
    if cd138_mask.sum() > 100:
        props = measure.regionprops(nuclei_mask)
        in_cd138 = sum(1 for p in props
                       if cd138_mask[int(round(p.centroid[0])),
                                    int(round(p.centroid[1]))])
        cd138_pct = round(in_cd138 / n_nuc * 100, 1) if n_nuc > 0 else 0
        total_cd138 = ttll12[cd138_mask].sum()
        cd138_bg = bg_mean * cd138_mask.sum()
        corrected_cd138 = max(total_cd138 - cd138_bg, 0)
        mean_cd138 = round(corrected_cd138 / in_cd138, 2) if in_cd138 > 0 else 0
        n_in_cd138 = in_cd138

    logger.info(
        f"    核={n_nuc}, TTLL12/核={mean_all:.0f}, "
        f"背景={bg_mean:.1f}, CD138质控={cd138_pct}%")

    return {
        "n_nuclei": n_nuc,
        "mean_per_cell": round(mean_all, 2),
        "bg_mean": round(bg_mean, 2),
        "total_raw": round(total_raw, 2),
        "total_corrected": round(total_corrected, 2),
        "cd138_qc_pct": cd138_pct,
        "cd138_nuclei": n_in_cd138,
        "cd138_mean": mean_cd138,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="IF定量工具 - 全场比值法")
    ap.add_argument("--dir", default=None,
                    help="工作目录（默认=脚本所在目录）")
    ap.add_argument("--sample", type=str, default=None, help="只分析指定样本")
    ap.add_argument("--test", action="store_true",
                    help=f"测试模式：每样本只处理前{TEST_FIELDS}个视野")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    work_dir = os.path.abspath(args.dir or WORK_DIR)
    if not os.path.isdir(work_dir):
        logger.error(f"x 目录不存在: {work_dir}"); sys.exit(1)

    out_dir = os.path.join(work_dir, OUTPUT_SUBDIR)

    ch_info = ", ".join(f"{k}={v[0]}({v[1]})" for k, v in CHANNEL_MAP.items())
    logger.info(f"工作目录: {work_dir}")
    logger.info(f"通道映射: {ch_info}")

    samples = scan_samples(work_dir)
    if not samples:
        logger.warning("x 未找到样本文件夹"); sys.exit(0)
    if args.sample:
        samples = [s for s in samples if os.path.basename(s) == args.sample]
        if not samples:
            logger.error(f"x 未找到 '{args.sample}'"); sys.exit(1)

    nf_total = 0
    for s in samples:
        fs = scan_fields(s)
        nf = min(len(fs), TEST_FIELDS) if args.test else len(fs)
        nf_total += nf
        tag = " (测试)" if args.test and nf < len(fs) else ""
        logger.info(f"  [{os.path.basename(s)}] {nf}/{len(fs)} 视野{tag}")
    logger.info(f"共 {len(samples)} 样本, {nf_total} 视野")

    if args.dry_run:
        sys.exit(0)

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"\n开始分析...")

    all_rows = []
    for sd in samples:
        sn = os.path.basename(sd)
        fields = scan_fields(sd)
        if args.test:
            fields = fields[:TEST_FIELDS]
        for fid, dap, t12, cd in fields:
            try:
                r = process_field(dap, t12, cd)
                if r:
                    all_rows.append({"Sample": sn, "Field": f"New-{fid}", **r})
            except Exception as e:
                logger.error(f"  x [{sn} New-{fid}] 失败: {e}")

    if not all_rows:
        logger.warning("x 无结果"); return

    # CSV: 每个视野详情
    detail_csv = os.path.join(out_dir, "if_quant_detail.csv")
    fn = ["Sample","Field",
          "n_nuclei","mean_per_cell","bg_mean","total_raw","total_corrected",
          "cd138_qc_pct","cd138_nuclei","cd138_mean"]
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)

    # CSV: 样本汇总
    summary = []
    for sn in sorted(set(r["Sample"] for r in all_rows)):
        sr = [r for r in all_rows if r["Sample"] == sn]
        nf = len(sr)
        total_cells = sum(r["n_nuclei"] for r in sr)
        wt = sum(r["mean_per_cell"] * r["n_nuclei"] for r in sr)
        wt = wt / total_cells if total_cells else 0
        field_means = [r["mean_per_cell"] for r in sr]
        sd = float(np.std(field_means, ddof=1)) if nf > 1 else 0
        qc_vals = [r["cd138_qc_pct"] for r in sr]
        summary.append({
            "Sample": sn, "N_Fields": nf,
            "Total_Cells": total_cells,
            "TTLL12_Mean": round(wt, 2),
            "SD": round(sd, 2),
            "SEM": round(sd / np.sqrt(nf), 2) if nf > 1 else 0,
            "CD138_QC_pct": round(np.mean(qc_vals), 1),
        })

    sum_csv = os.path.join(out_dir, "if_quant_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # 终端表格
    logger.info(f"\n{'='*68}")
    logger.info(f"分析完成 — {os.path.basename(work_dir)}")
    logger.info(f"{'='*68}")
    logger.info(f"{'样本':<12} {'视野':>4} {'总细胞':>8} "
                f"{'TTLL12均值':>14} {'SD':>10} {'SEM':>10} {'CD138质控':>9}")
    logger.info("-"*68)
    for r in summary:
        logger.info(f"{r['Sample']:<12} {r['N_Fields']:>4} {r['Total_Cells']:>8} "
                   f"{r['TTLL12_Mean']:>14.1f} {r['SD']:>10.1f} {r['SEM']:>10.1f} "
                   f"{r['CD138_QC_pct']:>8.1f}%")
    logger.info(f"\n详细: {detail_csv}")
    logger.info(f"汇总: {sum_csv}")
    logger.info(f"\n通道映射: {ch_info}")


if __name__ == "__main__":
    main()
