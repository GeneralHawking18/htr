import sys
import hydra
from omegaconf import DictConfig
import os, json, sys
from PIL import Image
sys.path.append('../')
sys.path.append("./")

from transformer_ocr.models.model import TransformerOCRCTC

import os


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
    conf_ctc.train()
    conf_ctc.export_submission()
    # os.system(f"kaggle datasets version -p /kaggle/working/ckpts -m cer_{conf_ctc.best_ckpt[1]}")

if __name__ == "__main__":
    main()
