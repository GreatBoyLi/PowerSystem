import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from joblib import Parallel, delayed  # 导入 joblib
from utils.config import load_config
import multiprocessing


def parse_filename_to_time(filename):
    """从文件名解析时间"""
    try:
        parts = filename.split('_')
        date_part = parts[2]
        time_part = parts[3]
        dt_str = f"{date_part}{time_part}"
        return datetime.strptime(dt_str, "%Y%m%d%H%M")
    except Exception:
        return None


def process_alignment_for_day(current_date, input_root, output_root):
    """处理单日的时间对齐 (作为子进程任务)"""
    yyyy = current_date.strftime("%Y")
    mm = current_date.strftime("%m")
    dd = current_date.strftime("%d")
    yyyymm = f"{yyyy}{mm}"

    day_input_dir = os.path.join(input_root, yyyymm, dd)
    day_output_dir = os.path.join(output_root, yyyymm, dd)

    if not os.path.exists(day_input_dir):
        return f"⚠️ {yyyy}-{mm}-{dd} 目录不存在"

    files = [f for f in os.listdir(day_input_dir) if f.endswith(".npy")]
    file_map = {parse_filename_to_time(f): os.path.join(day_input_dir, f) for f in files if parse_filename_to_time(f)}

    if not file_map:
        return f"⚠️ {yyyy}-{mm}-{dd} 无有效文件"

    if not os.path.exists(day_output_dir):
        os.makedirs(day_output_dir, exist_ok=True)

    target_times = pd.date_range(start=current_date, periods=24 * 4, freq='15min')
    success_count = 0

    for target_t in target_times:
        save_name = f"sat_15min_{target_t.strftime('%Y%m%d_%H%M')}.npy"
        save_path = os.path.join(day_output_dir, save_name)

        # 逻辑 A: 直接匹配 (00, 30)
        if target_t in file_map:
            img = np.load(file_map[target_t])
            np.save(save_path, img)
            success_count += 1
        # 逻辑 B: 线性插值 (15, 45)
        else:
            minute = target_t.minute
            remain = minute % 10
            prev_t = target_t - timedelta(minutes=remain)
            next_t = prev_t + timedelta(minutes=10)

            if prev_t in file_map and next_t in file_map:
                img_prev = np.load(file_map[prev_t]).astype(np.float32)
                img_next = np.load(file_map[next_t]).astype(np.float32)
                img_interp = (img_prev + img_next) / 2.0
                np.save(save_path, img_interp)
                success_count += 1

    return f"✅ {yyyy}-{mm}-{dd}: {success_count}/96"


if __name__ == "__main__":
    # 1. 加载配置
    config = load_config("../config/config.yaml")
    CROP_DIR = config["file_paths"]["crop_statellite_path"]
    ALIGNED_DIR = config["file_paths"]["aligned_satellite_path"]
    dates = pd.date_range(start=config["dates"]["start_date"],
                          end=config["dates"]["end_date"], freq='D')

    print(f"🚀 开始并行时间对齐 (10min -> 15min)")

    # 获取CPU核心数，留一个核心给系统，避免死机
    num_cores = multiprocessing.cpu_count() - 10

    # 2. 执行并行任务
    # n_jobs=-1 表示使用所有 CPU 核心
    # 使用 tqdm 包裹 Parallel 来显示总进度条
    results = Parallel(n_jobs=-1)(
        delayed(process_alignment_for_day)(d, CROP_DIR, ALIGNED_DIR)
        for d in tqdm(dates, desc="提交进度")
    )

    # 3. 打印结果摘要
    for r in results:
        if "⚠️" in r: print(r)

    print("\n🎉 所有日期对齐完成！")
