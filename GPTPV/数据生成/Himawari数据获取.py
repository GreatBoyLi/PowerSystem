import ftplib
import os
import pandas as pd
import logging
from tqdm import tqdm

# =================配置区域=================
FTP_HOST = "ftp.ptree.jaxa.jp"
# ⚠️ 请替换为您的真实账号密码
FTP_USER = "leewenpeng12_gmail.com"
FTP_PASS = "SP+wari8"

LOCAL_SAVE_DIR = "../data/himawari/"
LOG_FILE = "download_status.log"

START_DATE = "2020-01-01"
END_DATE = "2020-12-31"  # 可以设置为全年

BASE_REMOTE_DIR = "/pub/himawari/L2/PAR/021"


# =========================================

def setup_logger():
    """配置日志"""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8'
    )


def connect_ftp():
    """连接FTP"""
    ftp = ftplib.FTP()
    try:
        ftp.connect(FTP_HOST, timeout=30)
        ftp.login(FTP_USER, FTP_PASS)
        return ftp
    except Exception as e:
        msg = f"FTP连接失败: {e}"
        print(f"❌ {msg}")
        logging.error(msg)
        return None


def download_file_smart(ftp, remote_filename, local_path):
    """
    智能下载 (终极版)：
    1. 检查是否存在且完整 (跳过)
    2. 下载到 .tmp 临时文件
    3. 下载完成后重命名为正式文件
    这样保证本地永远不会有 "损坏的 .nc 文件"
    """
    try:
        # 1. 获取远程大小
        try:
            remote_size = ftp.size(remote_filename)
        except:
            remote_size = None

        # 2. 检查本地正式文件是否已存在且完整
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)

            # 如果远程能获取到大小，且两者相等 -> 完美匹配，跳过
            if remote_size is not None and local_size == remote_size:
                # 在进度条位置打印一行跳过信息 (可选，为了清爽可以注释掉)
                # print(f"   [跳过] 文件已存在且完整: {remote_filename}")
                logging.info(f"跳过已存在: {remote_filename}")
                return True  # 直接返回成功

            # 如果本地有文件但大小不对 -> 认为是坏文件，准备重下
            elif remote_size is not None and local_size != remote_size:
                logging.warning(f"文件不完整 (本地:{local_size} vs 远程:{remote_size})，重新下载: {remote_filename}")

            # 如果远程获取不到大小 (remote_size is None)，但本地有文件 -> 保守起见，跳过
            elif remote_size is None and local_size > 0:
                logging.info(f"跳过已存在 (无法获取远程大小): {remote_filename}")
                return True

        # ========================================================
        # 核心修改：使用 .tmp 临时文件名
        # ========================================================
        temp_path = local_path + ".tmp"  # 例如 data.nc.tmp

        logging.info(f"开始下载: {remote_filename}")

        with open(temp_path, "wb") as f:
            with tqdm(total=remote_size, unit='B', unit_scale=True, unit_divisor=1024,
                      desc=remote_filename, leave=False, miniters=1) as pbar:
                def callback(data):
                    f.write(data)
                    pbar.update(len(data))

                ftp.retrbinary(f"RETR {remote_filename}", callback, blocksize=32768)

        # ========================================================
        # 核心修改：只有下载这步完全没报错，才把 .tmp 改名为 .nc
        # ========================================================
        # 如果旧的损坏文件还在，先删掉它，给新文件腾位置
        if os.path.exists(local_path):
            os.remove(local_path)

        os.rename(temp_path, local_path)  # 改名操作是原子性的（瞬间完成）

        logging.info(f"下载成功: {remote_filename}")
        return True

    except Exception as e:
        # 捕捉网络错误
        logging.error(f"下载失败: {remote_filename} - {e}")
        print(f"\n   ⚠️ 下载出错: {e}")
        # 删掉那个半成品的 .tmp 文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    except KeyboardInterrupt:
        # 捕捉 Ctrl+C 手动停止
        print(f"\n   🛑 用户手动停止下载！清理临时文件...")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise  # 继续抛出异常，让主程序停止


def main():
    setup_logger()
    print(f"📝 日志已开启，查看 {LOG_FILE} 了解详细状态")

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    print(f"🚀 开始任务：{START_DATE} 至 {END_DATE}")

    ftp = connect_ftp()
    if not ftp: return

    for current_date in dates:
        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")
        dd = current_date.strftime("%d")
        yyyymm = f"{yyyy}{mm}"

        print(f"\n📅 处理日期: {yyyy}-{mm}-{dd}")

        # 只下载白天的数据 北京时间 04:00-22:00
        # 北京时间和UTC时间相差8个小时，即北京时间 - 8 等于 UTC时间
        # 这样可以节省大量时间和空间！
        # 如果需要全天，改回 range(24)
        for hour in range(4, 21):
            hour1 = (hour - 8) if (hour - 8) >= 0 else (hour - 8 + 24)
            hh = f"{hour1:02d}"

            remote_dir = f"{BASE_REMOTE_DIR}/{yyyymm}/{dd}/{hh}/"
            local_day_dir = os.path.join(LOCAL_SAVE_DIR, yyyymm, dd, hh)
            os.makedirs(local_day_dir, exist_ok=True)

            try:
                ftp.cwd(remote_dir)
                file_list = ftp.nlst()

                # 【筛选规则】只下载 5km 分辨率 (.02401_02401)
                nc_files = [f for f in file_list if f.endswith(".nc") and "02401_02401" in f]

                if not nc_files:
                    continue

                # 打印一下该小时有多少个文件
                print(f"   🕒 [UTC {hh}点] 发现 {len(nc_files)} 个目标文件")

                for filename in nc_files:
                    local_file_path = os.path.join(local_day_dir, filename)

                    # 调用新的智能下载函数
                    success = download_file_smart(ftp, filename, local_file_path)

                    if not success:
                        print("      🔄 连接重置，尝试重连...")
                        try:
                            ftp.quit()
                        except:
                            pass

                        ftp = connect_ftp()
                        if ftp:
                            ftp.cwd(remote_dir)
                            download_file_smart(ftp, filename, local_file_path)

            except ftplib.error_perm:
                pass  # 目录不存在，跳过
            except Exception as e:
                print(f"   ❌ 异常: {e}")
                try:
                    ftp = connect_ftp()
                except:
                    pass

    ftp.quit()
    print("\n✅ 所有任务结束！")


if __name__ == "__main__":
    main()
