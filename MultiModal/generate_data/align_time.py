import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from MultiModal.utils.config import load_config


def parse_filename_to_time(filename):
    """
    从文件名解析时间
    假设格式类似: NC_H08_20200127_0000_R21_FLDK...._crop.npy
    """
    try:
        # 根据你实际保存的文件名分割
        parts = filename.split('_')
        # 通常日期在第2个(索引2)，时间在第3个(索引3)
        # 比如: parts[0]=NC, parts[1]=H08, parts[2]=20200127, parts[3]=0000
        date_part = parts[2]
        time_part = parts[3]
        dt_str = f"{date_part}{time_part}"
        return datetime.strptime(dt_str, "%Y%m%d%H%M")
    except Exception:
        return None


def process_alignment_for_day(current_date, input_root, output_root):
    """
    处理单日的时间对齐
    """
    yyyy = current_date.strftime("%Y")
    mm = current_date.strftime("%m")
    dd = current_date.strftime("%d")
    yyyymm = f"{yyyy}{mm}"

    # 构造当天的输入/输出目录
    day_input_dir = os.path.join(input_root, yyyymm, dd)
    day_output_dir = os.path.join(output_root, yyyymm, dd)

    if not os.path.exists(day_input_dir):
        print(f"⚠️ 跳过日期 {yyyy}-{mm}-{dd} (输入目录不存在)")
        return

    # 1. 扫描当天所有 .npy 文件并建立索引
    files = [f for f in os.listdir(day_input_dir) if f.endswith(".npy")]
    file_map = {}  # Key: datetime, Value: full_path

    for f in files:
        dt = parse_filename_to_time(f)
        if dt:
            file_map[dt] = os.path.join(day_input_dir, f)

    if not file_map:
        print(f"⚠️ 日期 {yyyy}-{mm}-{dd} 下无有效 .npy 文件")
        return

    if not os.path.exists(day_output_dir):
        os.makedirs(day_output_dir)

    # 2. 生成当天的 15分钟 目标时间点 (00:00 到 23:45)
    # start_time = datetime(int(yyyy), int(mm), int(dd), 0, 0)
    # end_time = datetime(int(yyyy), int(mm), int(dd), 23, 45)
    target_times = pd.date_range(start=current_date, periods=24 * 4, freq='15min')

    success_count = 0

    for target_t in target_times:
        save_name = f"sat_15min_{target_t.strftime('%Y%m%d_%H%M')}.npy"
        save_path = os.path.join(day_output_dir, save_name)

        # 逻辑 A: 刚好有对应时刻 (00, 30) -> 直接拷贝
        if target_t in file_map:
            img = np.load(file_map[target_t])
            np.save(save_path, img)
            success_count += 1

        # 逻辑 B: 需要插值 (15, 45) -> 找前后 10分钟 的邻居
        else:
            minute = target_t.minute
            remain = minute % 10  # 应该是 5

            # 推算前后的 10分钟 时间点
            # 例 00:15 -> prev=00:10, next=00:20
            prev_t = target_t - timedelta(minutes=remain)
            next_t = prev_t + timedelta(minutes=10)

            if prev_t in file_map and next_t in file_map:
                # 加载两个邻居 (float32 用于计算)
                img_prev = np.load(file_map[prev_t]).astype(np.float32)
                img_next = np.load(file_map[next_t]).astype(np.float32)

                # 线性插值 (取平均)
                img_interp = (img_prev + img_next) / 2.0

                # 保存 (保持 float32)
                np.save(save_path, img_interp)
                success_count += 1
            else:
                # 邻居缺失，无法生成该时刻数据 (正常现象，比如原始数据这几分钟没拍)
                pass

    print(f"✅ {yyyy}-{mm}-{dd} 处理完成: 生成 {success_count}/96 帧")


if __name__ == "__main__":
    # 加载配置
    config_file = "../config/config.yaml"
    config = load_config(config_file)

    # 路径配置
    # 上一步裁剪好的 .npy 根目录
    CROP_DIR = config["file_paths"]["crop_statellite_path"]

    ALIGNED_DIR = config["file_paths"]["aligned_satellite_path"]

    start_date = config["dates"]["start_date"]
    end_date = config["dates"]["end_date"]

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    print(f"🚀 开始时间对齐 (10min -> 15min interpolation)")
    print(f"📂 输入目录: {CROP_DIR}")
    print(f"📂 输出目录: {ALIGNED_DIR}")

    for current_date in tqdm(dates):
        process_alignment_for_day(current_date, CROP_DIR, ALIGNED_DIR)

    print("\n🎉 所有日期对齐完成！")