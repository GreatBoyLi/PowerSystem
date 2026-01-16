from GPTPV.utils.config import load_config
from GPTPV.generate_data.process_himawari import main
from GPTPV.generate_data.process_era5 import extract_and_broadcast_era5
from GPTPV.generate_data.merge_data import merge_datasets
from GPTPV.generate_data.generate_power import run_simulation



if __name__ == "__main__":
    config_file = "../config/config.yaml"
    config = load_config(config_file)

    # 1. 处理Himawari文件
    main(config)

    # 2. 处理Era5文件
    extract_and_broadcast_era5(config)

    # 3. 合并文件
    merge_datasets(config)

    # 4. 生成功率文件
    run_simulation(config)