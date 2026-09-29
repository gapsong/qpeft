"""common.yml + one arm's yml -> a full axolotl config (the arm's keys win)."""
import sys

import yaml

common_path, arm_path, data_path, out_dir, config_path = sys.argv[1:]
config = yaml.safe_load(open(common_path)) | yaml.safe_load(open(arm_path))
config["datasets"][0]["path"] = data_path
config["dataset_prepared_path"] = f"{out_dir}/prepared"
config["output_dir"] = out_dir
yaml.safe_dump(config, open(config_path, "w"), sort_keys=False)
