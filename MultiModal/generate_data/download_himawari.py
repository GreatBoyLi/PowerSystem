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

# 这是一个存放科学数据的目录
LOCAL_SAVE_DIR = "../data/himawari_nc/"
LOG_FILE = "download_himawari_status.log"

# 论文使用的数据时间段 (示例)
START_DATE = "2020-01-27"
END_DATE = "2020-01-27"

# 根据之前的探测，目标数据位于此路径
BASE_REMOTE_DIR = "/jma/netcdf"


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
    # 提前定义 temp_path，防止在 ftp.size 报错时 except 块因找不到变量而崩溃
    temp_path = local_path + ".tmp"

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
                # 为了保持控制台清爽，这里不打印跳过信息，只写日志
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
    print(f"🚀 任务开始")
    print(f"📡 目标FTP目录: {BASE_REMOTE_DIR}")
    print(f"📂 本地保存目录: {LOCAL_SAVE_DIR}")

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    ftp = connect_ftp()
    if not ftp: return

    for current_date in dates:
        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")
        dd = current_date.strftime("%d")
        yyyymm = f"{yyyy}{mm}"

        print(f"\n📅 处理日期: {yyyy}-{mm}-{dd}")

        # JMA NetCDF 目录结构: /jma/netcdf/YYYYMM/DD/
        remote_dir = f"{BASE_REMOTE_DIR}/{yyyymm}/{dd}"
        local_day_dir = os.path.join(LOCAL_SAVE_DIR, yyyymm, dd)
        os.makedirs(local_day_dir, exist_ok=True)

        try:
            ftp.cwd(remote_dir)
            files = ftp.nlst()

            # 【筛选逻辑 - 针对论文复现】
            # 1. 必须是 .nc 结尾
            # 2. 找 "02401_02401" (代表 0.05度分辨率 ≈ 5km，符合论文)
            target_files = [f for f in files if f.endswith(".nc") and "02401_02401" in f]

            # 如果没找到特定分辨率的，尝试下载所有 nc 文件作为备选
            if not target_files:
                target_files = [f for f in files if f.endswith(".nc")]
                if target_files:
                    print(f"   ⚠️ 未找到明确标记 02401 的文件，将下载该目录所有 .nc 文件 ({len(target_files)}个)")

            if not target_files:
                print(f"   🚫 该日期目录下没有找到 .nc 文件")
                continue

            print(f"   🔍 发现 {len(target_files)} 个目标文件")

            for filename in target_files:
                local_path = os.path.join(local_day_dir, filename)

                # 调用终极版智能下载
                success = download_file_smart(ftp, filename, local_path)

                # 如果失败尝试重连一次
                if not success:
                    print("      🔄 连接重置，尝试重连...")
                    try:
                        ftp.quit()
                    except:
                        pass
                    ftp = connect_ftp()
                    if ftp:
                        try:
                            ftp.cwd(remote_dir)
                            download_file_smart(ftp, filename, local_path)
                        except Exception as e:
                            print(f"      ❌ 重连后依然失败: {e}")

        except ftplib.error_perm:
            print(f"   ⚠️ 远程目录不存在: {remote_dir}")
        except Exception as e:
            print(f"   ❌ 目录遍历异常: {e}")
            try:
                ftp = connect_ftp()  # 尝试恢复连接以处理下一个日期
            except:
                pass

    try:
        ftp.quit()
    except:
        pass
    print("\n✅ 所有任务结束！")


if __name__ == "__main__":
    main()