import sys
import hydra, yaml
from omegaconf import DictConfig
import os, json, sys
from PIL import Image
sys.path.append('../')
sys.path.append("./")

from transformer_ocr.models.model import TransformerOCRCTC

import os
# from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path='../conf', config_name="config")
def main(cfg: DictConfig):
    """if not os.path.exists("/kaggle/working/ckpts"):
        os.mkdir("/kaggle/working/ckpts")
    os.environ['KAGGLE_CONFIG_DIR'] = "/kaggle/input/credential"
    os.system("kaggle datasets init -p /kaggle/working/ckpts")
    # %%writefile /kaggle/working/ckpts/dataset-metadata.json
    dataset_metadata = {
      "title": "conformer-ctc-ckpt-full-data",
      "id": "generalhawking/conformer-ctc-ckpt-full-data",
      "licenses": [
        {
          "name": "CC0-1.0"
        }
      ]
    }
    
    with open("/kaggle/working/ckpts/dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f)"""

    conf_ctc = TransformerOCRCTC(config=cfg)
    # print(conf_ctc.model)
    for key in conf_ctc.record:
        print(f"{key}: {conf_ctc.record[key]}")
        print("---------------------------------------------------")
        
    # print(yaml.dump(conf_ctc.record, default_flow_style=False))

    if not cfg.pl_params.predict:
        conf_ctc.train()
    conf_ctc.export_submission(save_dir = "/kaggle/working")

    # os.system(f"kaggle datasets version -p /kaggle/working/ckpts -m cer_{conf_ctc.best_ckpt[1]}")

if __name__ == "__main__":
    main()
