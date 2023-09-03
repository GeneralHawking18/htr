import sys
import hydra
from omegaconf import DictConfig

sys.path.append('../')

from transformer_ocr.models.model import TransformerOCRCTC


@hydra.main(config_path='../conf', config_name="config")
def main(cfg: DictConfig):
    print("321312313123123123")
    model = TransformerOCRCTC(config=cfg)
    print(model.model)
    print(1313123)
    # model.train()


if __name__ == "__main__":
    main()
