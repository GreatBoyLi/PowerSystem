import os


def getreadfilepath(file, name=None):
    # 1. 获取当前文件的绝对路径
    script_path = os.path.abspath(file)
    print(f"文件绝对路径：{script_path}")

    # 2. 获取脚本所在的目录
    script_dir = os.path.dirname(script_path)
    print(f"文件所在目录：{script_dir}")
    if name is None:
        return script_dir
    else:
        file_dir = os.path.join(script_dir, name)
        file_dir = os.path.normpath(file_dir)  # 标准化路径（处理多余的/或..）
        print(f"{name}的绝对路径：{file_dir}")
        return file_dir


def getwritefilepath(file, subpath=None, name=None):
    # 1. 获取当前文件的绝对路径
    script_path = os.path.abspath(file)
    print(f"文件绝对路径：{script_path}")

    # 2. 获取脚本所在的目录
    script_dir = os.path.dirname(script_path)
    print(f"文件所在目录：{script_dir}")
    if subpath is not None:
        path = os.path.join(script_dir, subpath)
        path = os.path.normpath(path)  # 标准化路径（处理多余的/或..）
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        path = script_dir
    if name is None:
        return path
    else:
        file_dir = os.path.join(path, name)
        file_dir = os.path.normpath(file_dir)  # 标准化路径（处理多余的/或..）
        print(f"{name}的绝对路径：{file_dir}")
        return file_dir
