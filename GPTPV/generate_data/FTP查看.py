from ftplib import FTP


def check_latest_version(username, password):
    ftp = FTP("ftp.ptree.jaxa.jp")
    ftp.login(username, password)

    # 1. 尝试进入辐射产品目录
    # 注意：JAXA 目录名可能会变，常见的是 'SRP' 或 'PAR'
    try:
        ftp.cwd("/pub/himawari/L2/SRP")
    except:
        print("没找到 SRP 目录，尝试 PAR...")
        ftp.cwd("/pub/himawari/L2/PAR/021/202001/01/02")

    # 2. 列出该目录下的所有文件夹 (即版本号)
    versions = ftp.nlst()
    print("服务器上现有的版本文件夹:", versions)

    # 3. 找出数字最大的那个
    # latest = max(versions)
    # print(f"👉 最新版本应该是: {latest}")

    ftp.quit()

MY_USER = "leewenpeng12_gmail.com"
MY_PASS = "SP+wari8"

check_latest_version(MY_USER, MY_PASS)