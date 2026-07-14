#!/usr/bin/env python3
"""
免疫荧光定量分析工具
====================
CD138 分选阳性细胞的目的基因表达定量（全场比值法）

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
  定量公式: 目的基因平均表达 = 全场 488 总荧光 / DAPI 总核数

安全约束：只读取文件，绝不修改或删除任何文件。
"""

import os, sys, re, csv, logging, gc
import numpy as np
from PIL import Image
from scipy import ndimage as scipy_ndimage
from skimage import filters, morphology, measure, segmentation, feature

# ============================================================================
#  GPU 自动检测
# ============================================================================

def _detect_gpu():
    """检测各框架 GPU 可用性，返回状态摘要

    支持 NVIDIA CUDA / AMD DirectML / Apple MPS。
    """
    result = {"pytorch": False, "tensorflow": False,
              "directml": False, "mps": False, "summary": "CPU"}

    # PyTorch CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            result["pytorch"] = True
            result["summary"] = f"GPU (NVIDIA): {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass

    # DirectML (AMD / Intel GPU via torch-directml)
    if not result["pytorch"]:
        try:
            import torch_directml
            torch_directml.device()
            result["directml"] = True
            result["summary"] = "GPU (AMD/Intel): DirectML"
        except ImportError:
            pass

    # Apple MPS
    if not result["pytorch"] and not result["directml"]:
        try:
            import torch
            if (hasattr(torch.backends, 'mps') and
                    torch.backends.mps.is_available()):
                result["mps"] = True
                result["summary"] = "GPU (Apple): MPS"
        except ImportError:
            pass

    # TensorFlow (StarDist)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            result["tensorflow"] = True
    except ImportError:
        pass

    return result


# GPU 检测缓存（避免重复检测）
_gpu_info = None


def _gpu_cached():
    """GPU 检测（带缓存）"""
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = _detect_gpu()
    return _gpu_info


# StarDist 延迟加载（仅 --stardist 时导入）
_stardist_model = None

def _get_stardist():
    """延迟加载 StarDist 预训练模型（2D_versatile_fluo，适用 DAPI 核）

    自动处理 Windows 符号链接权限问题。
    """
    global _stardist_model
    if _stardist_model is None:
        try:
            from stardist.models import StarDist2D
            try:
                _stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
            except OSError as e:
                # Windows 上创建符号链接需要管理员权限
                # Fallback: 手动复制解压后的模型文件夹
                if hasattr(e, 'winerror') and e.winerror == 1314:
                    _fix_stardist_symlink()
                    _stardist_model = StarDist2D.from_pretrained('2D_versatile_fluo')
                else:
                    raise
            logging.getLogger("IF_Quant").info(
                "StarDist 预训练模型加载完成 (2D_versatile_fluo)")
        except ImportError:
            logging.getLogger("IF_Quant").error(
                "StarDist 未安装。请运行: pip install stardist tensorflow")
            sys.exit(1)
        except Exception as e:
            logging.getLogger("IF_Quant").error(f"StarDist 模型加载失败: {e}")
            sys.exit(1)
    return _stardist_model


def _fix_stardist_symlink():
    """修复 Windows 上 StarDist 模型符号链接权限问题"""
    import shutil
    from pathlib import Path
    keras_home = os.environ.get('KERAS_HOME',
                                os.path.join(os.path.expanduser('~'), '.keras'))
    model_dir = Path(keras_home) / 'models' / 'StarDist2D' / '2D_versatile_fluo'
    extracted = model_dir / '2D_versatile_fluo_extracted'
    target = model_dir / '2D_versatile_fluo'
    if extracted.is_dir() and not target.exists():
        shutil.copytree(str(extracted), str(target))
        logging.getLogger("IF_Quant").info("StarDist 模型目录已手动复制（绕过 symlink）")

# Cellpose 延迟加载（仅 --cellpose 时导入）
_cellpose_model = None

def _get_cellpose():
    global _cellpose_model
    if _cellpose_model is None:
        try:
            from cellpose import models as cp_models
            gpu_info = _gpu_cached()
            _cellpose_model = cp_models.CellposeModel(
                gpu=gpu_info["pytorch"])
            logging.getLogger("IF_Quant").info(
                f"Cellpose 预训练模型加载完成 (cpsam_v2, {gpu_info['summary']})")
        except ImportError:
            logging.getLogger("IF_Quant").error(
                "Cellpose 未安装。请运行: pip install cellpose")
            sys.exit(1)
    return _cellpose_model

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
    "c1": ("目的基因", "G"),   # 通道1: 488nm, 目的基因, 从RGB的G通道提取
    "c2": ("DAPI",   "B"),   # 通道2: DAPI,  细胞核, 从RGB的B通道提取
}
# 以后换染料/通道顺序，只需改上面这三行！

# ---- 核分割参数 ----
NUCLEUS_SIZE_MIN   = 200     # 最小核面积（像素^2），小淋巴细胞核约200-500
NUCLEUS_SIZE_MAX   = 80000   # 最大核面积，大浆细胞核可达数万
CIRCULARITY_MIN    = 0.1     # 最低圆形度 (0~1)，越低越允许不规则核
PROB_THRESH        = 0.78    # StarDist 概率阈值（5样本交叉验证，平均误差~21%）

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
    ttll12_key, ttll12_ch = CH_BY_NAME["目的基因"]
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
    """DAPI 核分割 -> 返回 label_mask（v2: multi-Otsu 3类 + 弱 closing）"""
    blr = filters.gaussian(dapi_img, sigma=1.5)
    # multi-Otsu 3类：暗背景 / 弱信号 / 亮核
    # 取最亮类（>第2个阈值）作为核候选，大幅减少背景误检
    thresholds = filters.threshold_multiotsu(blr, classes=3)
    bin_ = blr > thresholds[1]
    op = morphology.opening(bin_, morphology.disk(2))
    cl = morphology.remove_small_objects(op, max_size=30)
    # 只做 disk(1) closing，避免把相邻核粘连成大片
    cl = morphology.closing(cl, morphology.disk(1))

    dist = scipy_ndimage.distance_transform_edt(cl)
    peaks = feature.peak_local_max(dist, min_distance=8,
                                    exclude_border=3, labels=cl)
    if len(peaks) == 0:
        return np.zeros(dapi_img.shape, dtype=np.int32)
    markers = np.zeros(dapi_img.shape, dtype=np.int32)
    for i, (y, x) in enumerate(peaks):
        markers[y, x] = i + 1
    labs = segmentation.watershed(-dist, markers, mask=cl)

    out = np.zeros(labs.shape, dtype=np.int16)
    for p in measure.regionprops(labs):
        if p.area < NUCLEUS_SIZE_MIN or p.area > NUCLEUS_SIZE_MAX:
            continue
        circ = (4 * np.pi * p.area) / (p.perimeter**2) if p.perimeter > 0 else 0
        if circ < CIRCULARITY_MIN:
            continue
        out[labs == p.label] = p.label
    return out


def segment_nuclei_stardist(dapi_img: np.ndarray,
                            prob_thresh: float = 0.78) -> np.ndarray:
    """StarDist 预训练模型 DAPI 核分割 -> 返回 label_mask

    CNN 内部已完成去噪/阈值/分割/形状识别。
    仅后置面积过滤去除碎片。
    """
    model = _get_stardist()
    labels, _ = model.predict_instances(
        dapi_img,
        prob_thresh=prob_thresh,
        nms_thresh=0.3,
    )
    out = np.zeros(labels.shape, dtype=np.int16)
    for p in measure.regionprops(labels):
        if p.area < NUCLEUS_SIZE_MIN or p.area > NUCLEUS_SIZE_MAX:
            continue
        out[labels == p.label] = p.label
    return out


def segment_nuclei_cellpose(dapi_img: np.ndarray,
                            flow_threshold: float = 0.4,
                            cellprob_threshold: float = 0.0) -> np.ndarray:
    """Cellpose 预训练模型 DAPI 核分割 -> 返回 label_mask

    SAM-based 模型内部已完成分割/形状识别。
    仅后置面积过滤去除碎片。
    """
    model = _get_cellpose()
    masks, _, _, _ = model.eval(
        dapi_img,
        channels=[0, 0],
        diameter=None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    out = np.zeros(masks.shape, dtype=np.int16)
    for p in measure.regionprops(masks):
        if p.area < NUCLEUS_SIZE_MIN or p.area > NUCLEUS_SIZE_MAX:
            continue
        out[masks == p.label] = p.label
    return out


def measure_background(img: np.ndarray, nuclei_mask: np.ndarray) -> float:
    """从图像四边采样背景（低百分位，排除核区域）

    用 p25 估计背景：比中位数更不受碎屑影响，比均值更稳健。
    荧光图像背景分布右偏，p25 代表"典型暗背景"。
    """
    h, w = img.shape
    b = BACKGROUND_BORDER
    border = np.zeros((h, w), dtype=bool)
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
    bg = border & (nuclei_mask == 0)
    pixels = img[bg]
    if pixels.size < 100:
        pixels = img[border]
    return float(np.percentile(pixels, 25))


def process_field(dapi_path: str, ttll12_path: str, cd138_path: str,
                  use_stardist: bool = False,
                  use_cellpose: bool = False,
                  use_ensemble: bool = False,
                  prob_thresh: float = 0.78) -> dict:
    """处理单个视野

    use_ensemble=True 时：同时跑 StarDist + Cellpose，核数取两者平均。
    """
    dapi   = extract_channel(dapi_path,   CH_BY_NAME["DAPI"][1])
    ttll12 = extract_channel(ttll12_path, CH_BY_NAME["目的基因"][1])
    cd138  = extract_channel(cd138_path,  CH_BY_NAME["CD138"][1])
    if any(x is None for x in [dapi, ttll12, cd138]):
        return None

    if use_ensemble:
        # 并行推理 StarDist + Cellpose
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_sd = pool.submit(segment_nuclei_stardist, dapi, prob_thresh=prob_thresh)
            fut_cp = pool.submit(segment_nuclei_cellpose, dapi)
            # StarDist ~30s CPU, Cellpose 无超时（CPU 上可能很久）
            mask_sd = fut_sd.result(timeout=120)
            mask_cp = fut_cp.result(timeout=600)  # Cellpose SAM 模型 CPU 慢
        n_sd = len(np.unique(mask_sd)) - 1
        n_cp = len(np.unique(mask_cp)) - 1
        if n_sd == 0 or n_cp == 0:
            return None
        n_nuc = round((n_sd + n_cp) / 2)
        logger.info(f"    [Ensemble] StarDist={n_sd}, Cellpose={n_cp} → 平均={n_nuc}")
    elif use_cellpose:
        nuclei_mask = segment_nuclei_cellpose(dapi)
        n_nuc = len(np.unique(nuclei_mask)) - 1
    elif use_stardist:
        nuclei_mask = segment_nuclei_stardist(dapi, prob_thresh=prob_thresh)
        n_nuc = len(np.unique(nuclei_mask)) - 1
    else:
        nuclei_mask = segment_nuclei(dapi)
        n_nuc = len(np.unique(nuclei_mask)) - 1

    if use_ensemble:
        # ensemble 背景测量用并集 mask，更全面排除核区域
        nuclei_mask = np.where((mask_sd > 0) | (mask_cp > 0), mask_sd, 0).astype(np.int32)
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
        f"    核={n_nuc}, 目的基因/核={mean_all:.0f}, "
        f"背景={bg_mean:.1f}, CD138质控={cd138_pct}%")

    result = {
        "n_nuclei": n_nuc,
        "mean_per_cell": round(mean_all, 2),
        "bg_mean": round(bg_mean, 2),
        "total_raw": round(total_raw, 2),
        "total_corrected": round(total_corrected, 2),
        "cd138_qc_pct": cd138_pct,
        "cd138_nuclei": n_in_cd138,
        "cd138_mean": mean_cd138,
    }
    if use_ensemble:
        result["n_sd"] = n_sd
        result["n_cp"] = n_cp
    return result


def _safe_write_csv(path, rows, fieldnames):
    """安全写入 CSV，自动重试处理文件锁"""
    import time
    for attempt in range(5):
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames,
                                   extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(0.5)
            else:
                raise


def main():
    import argparse
    ap = argparse.ArgumentParser(description="IF定量工具 - 全场比值法")
    ap.add_argument("--dir", default=None,
                    help="工作目录（默认=脚本所在目录）")
    ap.add_argument("--sample", type=str, default=None, help="只分析指定样本")
    ap.add_argument("--test", action="store_true",
                    help=f"测试模式：每样本只处理前{TEST_FIELDS}个视野")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stardist", action="store_true",
                    help="使用 StarDist 预训练 DL 模型进行 DAPI 核分割（需先 pip install stardist tensorflow）")
    ap.add_argument("--cellpose", action="store_true",
                    help="使用 Cellpose 预训练模型进行 DAPI 核分割（需先 pip install cellpose）")
    ap.add_argument("--ensemble", action="store_true",
                    help="双模型集成：同时使用 StarDist + Cellpose，核数取平均提高准确率")
    ap.add_argument("--prob-thresh", type=float, default=0.78,
                    help="StarDist 概率阈值 (0~1)，越高越严格，默认 0.78 (5样本交叉验证)")
    ap.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="并行处理 N 个视野（默认 1=串行，建议 3-5）")
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
    if args.ensemble:
        seg_method = f"Ensemble SD+CP (prob={args.prob_thresh})"
    elif args.cellpose:
        seg_method = "Cellpose (cpsam_v2)"
    elif args.stardist:
        seg_method = f"StarDist (prob={args.prob_thresh})"
    else:
        seg_method = "multi-Otsu"
    logger.info(f"\n开始分析... (核分割: {seg_method})"
                + (f", 并行={args.parallel}视野" if args.parallel > 1 else ""))

    # 收集所有待处理视野
    all_tasks = []
    for sd in samples:
        sn = os.path.basename(sd)
        fields = scan_fields(sd)
        if args.test:
            fields = fields[:TEST_FIELDS]
        for fid, dap, t12, cd in fields:
            all_tasks.append((sn, fid, dap, t12, cd))

    def _process_one(task):
        sn, fid, dap, t12, cd = task
        try:
            r = process_field(dap, t12, cd,
                              use_stardist=args.stardist or args.ensemble,
                              use_cellpose=args.cellpose or args.ensemble,
                              use_ensemble=args.ensemble,
                              prob_thresh=args.prob_thresh)
            if r:
                return {"Sample": sn, "Field": f"New-{fid}", **r}
        except Exception as e:
            logger.error(f"  x [{sn} New-{fid}] 失败: {e}")
        finally:
            gc.collect()  # 释放该视野的大数组内存
        return None

    all_rows = []
    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_process_one, t): t for t in all_tasks}
            for n_done, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                if r:
                    all_rows.append(r)
                if n_done % 5 == 0 or n_done == len(futures):
                    logger.info(f"  进度: {n_done}/{len(futures)} 视野")
    else:
        for t in all_tasks:
            r = _process_one(t)
            if r:
                all_rows.append(r)

    if not all_rows:
        logger.warning("x 无结果"); return

    # CSV: 每个视野详情
    detail_csv = os.path.join(out_dir, "if_quant_detail.csv")
    fn = ["Sample","Field",
          "n_nuclei","n_sd","n_cp",
          "mean_per_cell","bg_mean","total_raw","total_corrected",
          "cd138_qc_pct","cd138_nuclei","cd138_mean"]
    _safe_write_csv(detail_csv, all_rows, fn)

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
            "Gene_Mean": round(wt, 2),
            "SD": round(sd, 2),
            "SEM": round(sd / np.sqrt(nf), 2) if nf > 1 else 0,
            "CD138_QC_pct": round(np.mean(qc_vals), 1),
        })

    sum_csv = os.path.join(out_dir, "if_quant_summary.csv")
    _safe_write_csv(sum_csv, summary, list(summary[0].keys()))

    # 终端表格
    logger.info(f"\n{'='*68}")
    logger.info(f"分析完成 — {os.path.basename(work_dir)}")
    logger.info(f"{'='*68}")
    logger.info(f"{'样本':<12} {'视野':>4} {'总细胞':>8} "
                f"{'目的基因均值':>14} {'SD':>10} {'SEM':>10} {'CD138质控':>9}")
    logger.info("-"*68)
    for r in summary:
        logger.info(f"{r['Sample']:<12} {r['N_Fields']:>4} {r['Total_Cells']:>8} "
                   f"{r['Gene_Mean']:>14.1f} {r['SD']:>10.1f} {r['SEM']:>10.1f} "
                   f"{r['CD138_QC_pct']:>8.1f}%")
    logger.info(f"\n详细: {detail_csv}")
    logger.info(f"汇总: {sum_csv}")
    logger.info(f"\n通道映射: {ch_info}")


if __name__ == "__main__":
    main()
