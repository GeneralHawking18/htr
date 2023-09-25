import os
import io, zipfile
import time
import numpy as np
import torch
import math
import logging
from pathlib import Path

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from omegaconf import DictConfig, ListConfig
from ctcdecode import CTCBeamDecoder
from torchsummary import summary
from tqdm import tqdm
import hydra

from transformer_ocr.core.optimizers import NaiveScheduler
from transformer_ocr.utils.vocab import VocabBuilder
from transformer_ocr.utils.dataset import (
    OCRDataset, Test_OCRDataset,
    ClusterRandomSampler, Collator
)
from transformer_ocr.utils.augment import ImgAugTransform
from transformer_ocr.utils.metrics import metrics
from transformer_ocr.utils.image_processing import resize_img
from transformer_ocr.models.cnn_extraction.feature_extraction import FeatureExtraction
from transformer_ocr.models.transformers.conformer import ConformerEncoder
from transformer_ocr.models.transformers.tr_encoder import TransformerEncoder

import warnings
from torch.optim.lr_scheduler import _LRScheduler

class NoamAnnealing(_LRScheduler):
    def __init__(
        self, optimizer, *, d_model, warmup_steps=None, warmup_ratio=None, max_steps=None, min_lr=0.0, last_epoch=-1
    ):
        self._normalize = d_model ** (-0.5)
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )

        new_lrs = [self._noam_annealing(initial_lr=initial_lr, step=step) for initial_lr in self.base_lrs]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        if self.warmup_steps > 0:
            mult = self._normalize * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        else:
            mult = self._normalize * step ** (-0.5)

        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr


class TransformerOCR(nn.Module):
    def __init__(self, vocab_size,
                 cnn_model,
                 cnn_args,
                 transformer_type,
                 transformer_args):

        super(TransformerOCR, self).__init__()
        self.feature_extraction = FeatureExtraction(cnn_model, **cnn_args)

        if transformer_type == 'transformer':
            self.transformer = TransformerEncoder(vocab_size, **transformer_args)
        elif transformer_type == 'conformer':
            self.transformer = ConformerEncoder(vocab_size, **transformer_args)
        else:
            raise ('Not Support model_type {}'.format(transformer_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = self.feature_extraction(x)
        # print("after fe shape: ", src.shape)
        outputs = self.transformer(src)

        return outputs


class TransformerOCRCTC:
    """TODO

    Args:

    """

    def __init__(self, config: DictConfig):
        super(TransformerOCRCTC, self).__init__()

        super(TransformerOCRCTC, self).__init__()

        self.config = config
        self.vocab = VocabBuilder(config.model.vocab)
        self.model = TransformerOCR(vocab_size=len(self.vocab),
                                    cnn_model=config.model.cnn_model,
                                    cnn_args=config.model.cnn_args,
                                    transformer_type=config.model.transformer_type,
                                    transformer_args=config.model.transformer_args)
        device = self.get_devices(self.config.pl_params.pl_trainer.gpus)
        print(device)
        
        # summary_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        summary(self.model, (
            3, self.config.dataset.dataset.unchanged.img_height,
            self.config.dataset.dataset.unchanged.img_width_max,
        ), device = 'cpu')
        
        if isinstance(device, int):
            logging.info("It's running on GPU {}".format(device))
            self.device = 'cuda:{}'.format(device)
            self.model = self.model.to(self.device)
        elif isinstance(device, list):
            logging.info("It's running on multi-GPUs {}".format(device))
            self.device = 'cuda:{}'.format(device[0])
            self.model = self.model.to('cuda:{}'.format(device[0]))
            self.model = nn.DataParallel(self.model, device_ids=device)
        else:
            self.device = device
            self.model = self.model.to(self.device)

        
        self.batch_size = config.model.batch_size
        self.epoch = 0
        
        if not self.config.pl_params.predict:
            logging.info("Start training ...")
            if not os.path.exists(config.dataset.dataset.root_save_path):
                os.mkdir(config.dataset.dataset.root_save_path)
            self.train_data = self.train_dataloader()# self.train_dataloader()
            self.valid_data = self.test_dataloader()
            self.test_data = self.test_dataloader()#self.test_dataloader()
            self.criterion = nn.CTCLoss(**self.config.pl_params.loss_func)
        
        else:
            logging.info("Start predicting ...")
            file_name = Path(self.config.pl_params.pretrained).stem
            score = float(file_name.split("_")[-1])
            self.best_ckpt = (file_name, score)
            self.test_data = self.test_dataloader()

        self.ctc_decoder = CTCBeamDecoder(
            self.vocab.get_vocab_tokens(),
            **self.config.lm_models
        )
        self.best_ckpt = ["", float("inf")]
        self.optimizer = AdamW(self.model.parameters(),
                             lr=config.optimizer.optimizer.lr,
                             betas=tuple(config.optimizer.optimizer.betas),
                             eps=config.optimizer.optimizer.eps)
        """self.lr_scheduler = NaiveScheduler(self.optimizer,
                                        config.optimizer.optimizer.lr_mul,
                                        config.model.transformer_args.d_model,
                                        config.optimizer.optimizer.n_warm_steps,
                                        eval(config.optimizer.optimizer.n_steps))"""

        """self.optimizer.param_groups[0]['initial_lr'] = 6.59e-05
        self.optimizer.param_groups[0]['max_lr'] = 1e-3
        self.optimizer.param_groups[0]['min_lr'] = 1e-7
        self.optimizer.param_groups[0]['base_momentum '] = 0.85
        self.optimizer.param_groups[0]['max_momentum '] = 0.95"""


        """self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode = 'min',
            min_lr = 1e-6,
            patience = 3,
            verbose = True,
        )"""
        n_devices = 1 if isinstance(device, int) else len(device)
        """self.lr_scheduler = OneCycleLR(
            self.optimizer,
            epochs = self.config.pl_params.pl_trainer.max_epochs,
            steps_per_epoch = len(self.train_data),
            pct_start = config.optimizer.lr_scheduler.pct_start,
            max_lr = config.optimizer.lr_scheduler.max_lr,
            div_factor = config.optimizer.lr_scheduler.div_factor,
            final_div_factor = config.optimizer.lr_scheduler.final_div_factor,
            three_phase = config.optimizer.lr_scheduler.three_phase,
            # last_epoch = 32 * int(87550/self.batch_size), 
        )"""
        
        """self.lr_scheduler = hydra.utils.instantiate(
            config.optimizer.OneCycleLR,
            optimizer = self.optimizer,
            epochs = self.config.pl_params.pl_trainer.max_epochs,
            steps_per_epoch = len(self.train_data),
        )"""
        """self.lr_scheduler = hydra.utils.instantiate(
            config.optimizer.CyclicLR,
            _paritial_ = True,
            optimizer = self.optimizer,
            base_lr = config.optimizer.CyclicLR.base_lr,
            max_lr= config.optimizer.CyclicLR.max_lr,
            base_momentum = config.optimizer.CyclicLR.base_momentum,
            max_momentum  = config.optimizer.CyclicLR.max_momentum,
            step_size_up = config.optimizer.CyclicLR.epoch_size_up * len(self.train_data),
            step_size_down = config.optimizer.CyclicLR.epoch_size_down * len(self.train_data),
            scale_mode = config.optimizer.CyclicLR.scale_mode,
            mode = config.optimizer.CyclicLR.mode,
            gamma = config.optimizer.CyclicLR.gamma,
            cycle_momentum = config.optimizer.CyclicLR.cycle_momentum,
        )"""
        step_per_epoch = len(self.train_data)
        self.lr_scheduler = NoamAnnealing(
            self.optimizer,
            d_model = 768,
            warmup_steps = step_per_epoch * 2,
        )
        """self.lr_scheduler = hydra.utils.instantiate(
            config.optimizer.CyclicLR,
            optimizer = self.optimizer,
            step_size_up = config.optimizer.CyclicLR.step_size_up * len(self.train_data),
            step_size_down = config.optimizer.CyclicLR.step_size_down * len(self.train_data),
            gamma= 1 - 4e-4,
        )"""
        
        

        """self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, 
            base_lr = 6e-8,
            max_lr= 9e-05,
            base_momentum = 0.85,
            max_momentum  = 0.95,
            step_size_up = 683 * 1,
            step_size_down = 683 * 40,
            scale_mode = "iterations",
            mode = "exp_range",
            gamma = 1 - 1e-4,
            cycle_momentum = False,
        )"""
        
        if self.config.pl_params.pretrained:
            if not os.path.exists(self.config.pl_params.pretrained):
                logging.error('{} not exists. Please verify this!'.format(self.config.pl_params.pretrained))
                exit(0)

            logging.info("Start loading pre-trained ckpt from {}".format(self.config.pl_params.pretrained))
            
            self.load_checkpoint(self.config.pl_params.pretrained)
        
        self.best_ckpt = ["", float("inf")]
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, step):
        img = batch['img'].cuda(non_blocking=True, device=self.device)
        tgt_output = batch['tgt_output'].cuda(non_blocking=True, device=self.device)
        # print("len gt and gt: ", len(tgt_output), tgt_output.shape)
        
        #a = self.convert_to_string(tgt_output[0], tgt_output.shape[0])
        # print("string and gt num", a, tgt_output[0])
        outputs = self.model(img)
        # print("outputs prob shape: ", outputs.shape)
        
        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1).requires_grad_()
        
        # length = torch.tensor([tgt_output.size(1)] * outputs.size(1), device=outputs.device).long()
        length = batch['target_lens']
        # print(length.shape)
        preds_size = torch.tensor([outputs.size(0)] * outputs.size(1), device=outputs.device).long()

        loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)

        # Label smoothing loss
        if self.config.pl_params.ctc_smoothing:
            loss = loss * (1 - self.config.pl_params.ctc_smoothing) + \
                   self.kldiv_lsm_ctc(outputs.transpose(0, 1), preds_size) * self.config.pl_params.ctc_smoothing

        # Accumulation gradiant training
        loss = loss / self.config.pl_params.pl_trainer.accumulate_grad_batches
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.pl_params.max_norm)
        if (step + 1) % self.config.pl_params.pl_trainer.accumulate_grad_batches == 0:
            # self.optimizer.step_and_update_lr()
            self.optimizer.step()
            self.lr_scheduler.step()
            # print(step)
            self.optimizer.zero_grad()
        
        return loss

    def train(self):
        total_loss: float = 0.0
        total_loader_time: float = 0.0
        total_gpu_time: float = 0.0
        best_acc: float = 0.0
        start_step: int = 0
        total_data_size = 0
        data_iter = iter(self.train_data)
        self.model.train()

        
        while self.epoch < self.config.pl_params.pl_trainer.max_epochs:
            self.epoch += 1
            for i, batch in enumerate(
                (train_tqdm:= tqdm(
                    self.train_data, 
                    desc = f"Epoch {self.epoch}-training: ",
                    ncols=100,
                    leave=True,
                    position=0,
                ))
            ):
                self.model.train()
                loss = self.training_step(batch=batch, step=(self.epoch-1) * len(self.train_data) + i)
                train_tqdm.set_postfix({
                    'loss': round(loss.item(), 4),
                    "lr":  self.lr_scheduler.get_last_lr(), #self.optimizer.param_groups[0]['lr'] 

                })
                total_loss += loss.item()
                total_data_size += batch['img'].shape[0]
                
            info = 'Epoch {}: train loss: {:.3f} - lr: {:.2e}'.format(
                self.epoch,
                total_loss / len(self.train_data),
                self.lr_scheduler.get_last_lr()[0], # self.optimizer.param_groups[0]['lr'],
            )

            logging.info(info)
            total_loss = 0
            # val_info = self.validation()
            saved_ckpt = f"{self.config.pl_params.model_callbacks.dirpath}/best_ckpt_{self.epoch}.pth"
                    # self.config.pl_params.model_callbacks.filename.format()
            self.export_submission()
            self.save_checkpoint(saved_ckpt)
            
            """if val_info['cer'] < self.best_ckpt[1]:
                saved_ckpt = f"{self.config.pl_params.model_callbacks.dirpath}/best_ckpt_0_{self.epoch}.pth"
                    # self.config.pl_params.model_callbacks.filename.format()
                self.save_checkpoint(saved_ckpt)
                # torch.save(self.optimizer.state_dict(), filename)
                self.best_ckpt[0] = saved_ckpt
                self.best_ckpt[1] = val_info['cer']"""


    def validation(self):
        self.model.eval()
        losses = np.array([])
        pred_sents = []
        actual_sents = []
        num_samples = 0
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_data):
                valid_dict = self.validation_step(batch=batch)

                losses = np.append(losses, valid_dict['loss'].cpu().detach().numpy())
                num_samples += len(batch)
                
                actual_sents.extend(self.vocab.batch_decode(valid_dict['tgt_output'].tolist()))

                if self.config.pl_params.use_beamsearch:
                    beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(
                        valid_dict['logits'].softmax(2))

                    for i in range(beam_results.size(0)):
                        pred_sent = self.convert_to_string(beam_results[i][0], out_lens[i][0])
                        pred_sents.append(pred_sent.replace('<pad>', ''))
                else:
                    logits = valid_dict['logits'].cpu().detach().numpy()
                    pred_sents.extend([self._greedy_decode(logits[i]) for i in range(logits.shape[0])])
                

        avg_sent_acc = metrics(actual_sents, pred_sents, type='accuracy')
        acc_per_char = metrics(actual_sents, pred_sents, type='char_acc')
        normalized_ed = metrics(actual_sents, pred_sents, type='normalized_ed')
        cer = metrics(actual_sents, pred_sents, type='cer')
        """for i in range(len(pred_sents)):
            if pred_sents[i] != actual_sents[i]:
                print('Actual_sent: {}, pred_sent: {}'.format(actual_sents[i], pred_sents[i]))"""
        
        val_info = {"loss": losses.mean(),
                    "word accuracy": avg_sent_acc * 100,
                    "character accuracy": acc_per_char * 100,
                    "normalized edit distance": normalized_ed * 100,
                    "cer": cer.item()}
        
        logging.info(val_info)

        return val_info

    def validation_step(self, batch):
        img = batch['img'].cuda(non_blocking=True, device=self.device)
        tgt_output = batch['tgt_output'].cuda(non_blocking=True, device=self.device)

        logits = self.model(img)
        logits = F.log_softmax(logits, dim=2)
        outputs = logits.transpose(0, 1)
        # length = torch.tensor([tgt_output.size(1)] * outputs.size(1), device=outputs.device).long()
        length = batch['target_lens']
        preds_size = torch.tensor([outputs.size(0)] * outputs.size(1), device=outputs.device).long()
        loss = self.criterion(outputs, tgt_output, preds_size, length) / outputs.size(1)

        return {
            'loss': loss,
            'logits': logits,
            'tgt_output': tgt_output
        }
    
    def infer(self, batch):
        imgs_name = [os.path.basename(img_dir) for img_dir in batch['img_path']]
        imgs = batch['img']
        imgs = imgs.cuda(non_blocking=True, device=self.device)
        

        logits = self.model(imgs)
        self.model.eval()
        logits = F.log_softmax(logits, dim=2)
        pred_sents = []
        confidences = []
        
        if self.config.pl_params.use_beamsearch:
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(
                logits.softmax(2))
            beam_scores = 1/torch.exp(beam_scores)
            
            for i in range(beam_results.size(0)):
                pred_sent = self.convert_to_string(beam_results[i][0], out_lens[i][0])
                pred_sents.append(pred_sent.replace('<pad>', ''))
                confidences.append(beam_scores[i][0].item())
            else:
                logits = logits.cpu().detach().numpy()
                pred_sents.extend([self._greedy_decode(logits[i]) for i in range(logits.shape[0])])
        
        return imgs_name, pred_sents, confidences
    
    def predict(self, img):
        resized_img = resize_img(img, self.config.dataset.dataset.unchanged.img_height,
                                 self.config.dataset.dataset.unchanged.img_width_min,
                                 self.config.dataset.dataset.unchanged.img_width_max)

        img = transforms.ToTensor()(resized_img).unsqueeze(0).to(self.device)
        img = img / 255

        logits = self.model(img)
        logits = F.log_softmax(logits, dim=2)

        if self.config.pl_params.use_beamsearch:
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(logits.softmax(2))
            pred_sent = self.convert_to_string(beam_results[0][0], out_lens[0][0])
        else:
            logits = logits.cpu().detach().numpy()
            pred_sent = self._greedy_decode(logits[0])

        return pred_sent
    
    def export_submission(self):
        logging.info('Start predicting ...')
        # Load the best weight to submit
        """if self.best_ckpt[0] != "":
            self.load_weights(self.best_ckpt[0])"""
        self.model.eval()
        model_name = self.model.__class__.__name__
        best_cer_val = round(self.best_ckpt[1], 4)
        
        # save_folder_path = os.getcwd() + "/outputs"
        """if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)"""
        # Nó tự động lưu trong outputs mà hydra cài sẵn, nên không cần phải makedir
        file_zip_path = f"{model_name}_{self.epoch}.zip"# f"{model_name}_{best_cer_val}.zip"
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
            data = io.BytesIO()
            for batch in tqdm(self.test_data, desc = "Testing: "):
                
                img_names, preds, confidences = self.infer(batch)

                for img_name, pred, confidence in zip(img_names, preds, confidences):
                    if pred == "":
                        pred = "a"
                    line = bytes(f"{img_name} {pred} {confidence} \n", "utf=8")
                    data.write(line)
    
            zip_file.writestr("prediction.txt", data.getvalue())

        # Store file
        with open(file_zip_path, 'wb') as f:
            f.write(zip_buffer.getvalue())
        
        
    @property
    def transform(self):
        if not self.config.dataset.aug.image_aug:
            return None

        return ImgAugTransform()

    def train_dataloader(self) -> DataLoader:
        
        _dataloader = self._prepare_data(
            saved_path = '{}/train_{}'.format(
                self.config.dataset.dataset.root_save_path,
                self.config.dataset.dataset.name
            ),
            gt_path = self.config.dataset.dataset.train_annotation,
            imgs_path = self.config.dataset.dataset.train_imgs_dir,
            use_transform = True,
            # drop_last = True,
        )
        
        return _dataloader

    def val_dataloader(self) -> DataLoader:
        _dataloader = self._prepare_data(
            saved_path = '{}/valid_{}'.format(
                self.config.dataset.dataset.root_save_path,
                self.config.dataset.dataset.name,
            ),
            gt_path = self.config.dataset.dataset.valid_annotation,
            imgs_path = self.config.dataset.dataset.valid_imgs_dir,
            use_transform = False,
        )
        return _dataloader
    
    def test_dataloader(self) -> DataLoader:
        _dataloader = self._prepare_data(
            saved_path = '{}/test_{}'.format(
                self.config.dataset.dataset.root_save_path,
                self.config.dataset.dataset.name,
            ),
            gt_path = self.config.dataset.dataset.test_annotation,
            imgs_path = self.config.dataset.dataset.test_imgs_dir,
            use_transform = False,
        )
        return _dataloader 
    
    def predict_dataloader(self) -> DataLoader:
        _dataset = Test_OCRDataset(
            test_imgs_dir = self.config.dataset.dataset.test_imgs_dir,
            transform = self.transform,            
        )
        _dataloader = DataLoader(
            _dataset,
            batch_size = self.batch_size,
            collate_fn = Test_OCRDataset.collate_fn,
            shuffle = False,
        )
        return _dataloader


    def _prepare_data(self, saved_path: str, gt_path: str, imgs_path, 
                      use_transform: bool = True,
                      drop_last = False, shuffle = False) -> DataLoader:
        if not use_transform:
            transform = None
        else:
            transform = self.transform

        _dataset = OCRDataset(saved_path=saved_path,
                             gt_path=gt_path,
                             root_dir = imgs_path,
                             vocab_builder=self.vocab,
                             transform=transform,
                             **self.config.dataset.dataset.unchanged)

        _dataloader = DataLoader(
            _dataset,
            batch_size=self.batch_size,
            sampler=ClusterRandomSampler(_dataset, self.batch_size),
            collate_fn=Collator(),
            shuffle = shuffle,
            drop_last = drop_last,
            **self.config.dataset.dataloader)

        return _dataloader

    def _greedy_decode(self, logits) -> str:
        """Decode argmax of logits and squash in CTC fashion."""
        label_dict = {n: c for n, c in enumerate(self.vocab.get_vocab_tokens())}
        prev_c = None
        out = []
        for n in logits.argmax(axis=-1):
            c = label_dict.get(n, "")  # if not in labels, then assume it's CTC <blank> token or <pad> token

            if c in [self.vocab.index_2_tok[0], self.vocab.index_2_tok[1]]:
                c = ""

            if c != prev_c:
                out.append(c)
            prev_c = c

        return "".join(out)

    def save_weights(self, filename):
        dir = os.path.dirname(filename)
        os.makedirs(dir, exist_ok=True)
        torch.save(self.model.state_dict(), filename)
    def convert_to_string(self, tokens, seq_len):
        return "".join([self.vocab.get_vocab_tokens()[x] for x in tokens[0:seq_len]])

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict, strict=True)
        
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        
        self.epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.best_ckpt[0] = filename
        self.best_ckpt[1] = checkpoint['cer']
        
        # self.iter = checkpoint['iter']

        # self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {
            # 'iter': self.iter, 
            "epoch": self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), 
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "cer": self.best_ckpt[1]
        }
        
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    @staticmethod
    def kldiv_lsm_ctc(logits: torch.Tensor, ylens: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for label smoothing of CTC and Transducer models.
        Args:
            logits (FloatTensor): `[B, T, vocab]`
            ylens (IntTensor): `[B]`
        Returns:
            loss_mean (FloatTensor): `[1]`
        """
        bs, _, vocab = logits.size()

        log_uniform = logits.new_zeros(logits.size()).fill_(math.log(1 / (vocab - 1)))
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = torch.mul(probs, log_probs - log_uniform)
        loss_mean = sum([loss[b, :ylens[b], :].sum() for b in range(bs)]) / ylens.sum()

        return loss_mean

    @staticmethod
    def get_devices(device):
        if isinstance(device, int):
            _device = device
        elif isinstance(device, ListConfig):
            _device = list(device)
        else:
            raise Exception("Please fill list of integers or single values. For example, gpus: [0, 1] or gpus: 0")

        if not torch.cuda.is_available():
            logging.info("It's running on CPU!")
            _device = 'cpu'

        return _device

