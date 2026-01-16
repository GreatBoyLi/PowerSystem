import ftplib

FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = "leewenpeng12_gmail.com"  # 你的账号
FTP_PASS = "SP+wari8"  # 你的密码


def explore_ftp():
    ftp = ftplib.FTP()
    try:
        print(f"🔌 正在连接 {FTP_HOST} ...")
        ftp.connect(FTP_HOST, timeout=30)
        ftp.login(FTP_USER, FTP_PASS)
        print("✅ 登录成功！\n")

        # 1. 列出根目录
        print("📂 根目录下的文件夹:")
        root_files = ftp.nlst()
        print(root_files)

        # 2. 尝试寻找常见的卫星数据目录
        # 常见的可能路径有: /jma, /pub, /nc, /gridded 等
        potential_dirs = ['/jma', '/pub', '/pub/himawari', '/jma/netcdf']

        print("\n🔍 深度探测常见路径:")
        for d in potential_dirs:
            try:
                files = ftp.nlst(d)
                print(f"  ✅ 发现路径: {d}")
                # 打印该路径下的前3个内容，看看是不是我们要的
                print(f"     内容示例: {files[:3]}")
            except ftplib.error_perm:
                print(f"  ❌ 路径不存在或无权限: {d}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        ftp.quit()


if __name__ == "__main__":
    explore_ftp()