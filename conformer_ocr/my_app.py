import hydra
from omegaconf import DictConfig, OmegaConf

# import sys
# sys.path.append("/content/drive/MyDrive/HUST/ai_hackathon/ocr/htr/conformer_ocr")
# print(sys.path)

from transformer_ocr.models import model
from transformer_ocr.utils import dataset



config_path = "/content/drive/MyDrive/HUST/ai_hackathon/ocr/htr/conformer_ocr/conf"
config_name = "config"
@hydra.main(version_base=None, config_path=config_path, config_name= config_name)
def my_app(cfg : DictConfig) -> None:
    # dataset.test()
    conf_ocr = model.TransformerOCRCTC(cfg)
    conf_ocr.train()
    conf_ocr.export_submission()
    # print(cfg)
    # print(cfg.model)
    # a = OmegaConf.to_yaml(cfg)
    # print(a["dataset"])
    
if __name__ == "__main__":
    my_app()