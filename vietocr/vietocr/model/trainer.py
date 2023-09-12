from vietocr.optim.optim import ScheduledOptim
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import Adam, SGD, AdamW
from torch import nn
from vietocr.tool.translate import build_model
from vietocr.tool.translate import translate, batch_translate_beam_search
from vietocr.tool.utils import download_weights
from vietocr.tool.logger import Logger
from vietocr.loader.aug import ImgAugTransform

import yaml
import torch
from vietocr.loader.dataloader_v1 import DataGen
from vietocr.loader.dataloader import OCRDataset, ClusterRandomSampler, Collator
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, OneCycleLR

import torchvision 

from torchsummary import summary

from vietocr.tool.utils import compute_accuracy
from PIL import Image
import numpy as np
import os, io, zipfile 
from tqdm import tqdm

import matplotlib.pyplot as plt
import time

class Trainer():
    def __init__(self, config, pretrained=False, augmentor=ImgAugTransform()):
        gpus = config['gpus']
        device = config["device"]
        logger = config['trainer']['log']
        if logger:
            self.logger = Logger(logger) 
            
        self.config = config
        self.model, self.vocab = build_model(config)
        # summary(self.model, (3, 32, 128), "cpu")
        if "cuda" in device:
        
            if isinstance(gpus, int):
                self.logger.log("It's running on GPU {}".format(gpus))
                self.device = 'cuda:{}'.format(gpus)
                self.model = self.model.to(self.device)
            elif isinstance(gpus, list):
                self.logger.log("It's running on multi-GPUs {}".format(gpus))
                self.device = 'cuda:{}'.format(gpus[0])
                self.model = self.model.to('cuda:{}'.format(gpus[0]))
                self.model = nn.DataParallel(self.model, device_ids=gpus)
        else:
            self.device = config['device']
            self.model = self.model.to(self.device)
        
        
        self.num_iters = config['trainer']['iters']
        self.beamsearch = config['predictor']['beamsearch']

        self.data_root = config['dataset']['data_root']
        self.test_data_root = config['dataset']['test_data_root']
        self.train_annotation = config['dataset']['train_annotation']
        self.valid_annotation = config['dataset']['valid_annotation']
        self.test_annotation = config['dataset']['test_annotation']

        self.dataset_name = config['dataset']['name']

        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']
        
        self.image_aug = config['aug']['image_aug']
        self.masked_language_model = config['aug']['masked_language_model']

        self.checkpoint = config['trainer']['checkpoint']
        self.export_weights = config['trainer']['export']
        self.metrics = config['trainer']['metrics']
        
    

        if pretrained:
            weight_file = download_weights(**config['pretrain'], quiet=config['quiet'])
            self.load_weights(weight_file)

        self.iter = 0
        
        self.optimizer = AdamW(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, total_steps=self.num_iters, **config['optimizer'])
#        self.optimizer = ScheduledOptim(
#            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
#            #config['transformer']['d_model'], 
#            512,
#            **config['optimizer'])

        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)
        
        transforms = None
        if self.image_aug:
            transforms =  augmentor
        
        self.train_gen = self.data_gen('train_{}'.format(self.dataset_name), 
                self.data_root, self.train_annotation, self.masked_language_model, transform=transforms)
        if self.valid_annotation:
            self.valid_gen = self.data_gen('valid_{}'.format(self.dataset_name), 
                    self.data_root, self.valid_annotation, masked_language_model=False)
        
        if self.test_annotation:
            self.test_gen = self.data_gen('test_{}'.format(self.dataset_name), 
                    self.test_data_root, self.test_annotation, masked_language_model=False)

        self.train_losses = []
        self.best_cer = 0 
        self.max_seq_length = config['transformer']["max_seq_length"]

    def train(self):
        total_loss = 0
        
        total_loader_time = 0
        total_gpu_time = 0

        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}'.format(self.iter, 
                        total_loss/self.print_every, self.optimizer.param_groups[0]['lr'], 
                        total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info) 
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss = self.validate()
                acc_full_seq, acc_per_char, cer = self.precision(self.metrics)

                info = 'iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(self.iter, val_loss, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)

                if cer < self.best_cer:
                    self.save_weights(self.export_weights)
                    self.best_cer = cer

            
    def validate(self):
        self.model.eval()

        total_loss = []
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                batch = self.batch_to_device(batch)
                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

                outputs = self.model(img, tgt_input, tgt_padding_mask)
#                loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
               
                outputs = outputs.flatten(0,1)
                tgt_output = tgt_output.flatten()
                loss = self.criterion(outputs, tgt_output)

                total_loss.append(loss.item())
                
                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        
        return total_loss
    
    def predict(self, mode = "valid", sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []

        if mode == "valid":
            data_gen = self.valid_gen
        else:
            data_gen = self.test_gen

        for batch in tqdm(
            data_gen, desc = f"{mode}ing: ",
            ncols = 100, position=0, leave=True
        ):
            batch = self.batch_to_device(batch)

            if self.beamsearch:
                translated_sentence = batch_translate_beam_search(
                    batch['img'], self.model, 
                    max_seq_length = self.max_seq_length)
                prob = None
            else:
                translated_sentence, prob = translate(
                    batch['img'], self.model, 
                    max_seq_length = self.max_seq_length,
                )

            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_output'].tolist())

            img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            
            if sample != None and len(pred_sents) > sample:
                break

        return pred_sents, actual_sents, img_files, prob

    def precision(self, sample=None):

        pred_sents, actual_sents, _, _ = self.predict(sample=sample)

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
        cer = compute_accuracy(actual_sents, pred_sents, mode='cer')
        
        return acc_full_seq, acc_per_char, cer
    
    def export_submission(self):
            self.logger.log('Start predicting ...')
            self.model.eval()

            # Load the best weight to submit
            if self.export_weights:
                self.load_weights(self.export_weights)

            model_name = self.model.__class__.__name__
            best_cer_val = round(self.best_cer)

            # save_folder_path = os.getcwd() + "/outputs"
            """if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)"""

            # Nó tự động lưu trong outputs mà hydra cài sẵn, nên không cần phải makedir
            file_zip_path = f"{model_name}_{best_cer_val}.zip"

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                data = io.BytesIO()

                _, preds, _, img_names = self.predict(mode = "test")
                print(img_names)
                for img_name, pred in zip(img_names, preds):
                    if pred == "":
                        pred = "a"
                    line = bytes(f"{img_name} {pred}\n", "utf=8")
                    data.write(line)

            zip_file.writestr("prediction.txt", data.getvalue())

            # Store file
            with open(file_zip_path, 'wb') as f:
                f.write(zip_buffer.getvalue())

    def visualize_prediction(self, sample=16, errorcase=False, fontname='serif', fontsize=16):
        
        pred_sents, actual_sents, img_files, probs = self.predict(sample)

        if errorcase:
            wrongs = []
            for i in range(len(img_files)):
                if pred_sents[i]!= actual_sents[i]:
                    wrongs.append(i)

            pred_sents = [pred_sents[i] for i in wrongs]
            actual_sents = [actual_sents[i] for i in wrongs]
            img_files = [img_files[i] for i in wrongs]
            probs = [probs[i] for i in wrongs]

        img_files = img_files[:sample]

        fontdict = {
                'family':fontname,
                'size':fontsize
                } 

        for vis_idx in range(0, len(img_files)):
            img_path = img_files[vis_idx]
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]
            prob = probs[vis_idx]

            img = Image.open(open(img_path, 'rb'))
            plt.figure()
            plt.imshow(img)
            plt.title('prob: {:.3f} - pred: {} - actual: {}'.format(prob, pred_sent, actual_sent), loc='left', fontdict=fontdict)
            plt.axis('off')

        plt.show()
    
    def visualize_dataset(self, sample=16, fontname='serif'):
        n = 0
        for batch in self.train_gen:
            for i in range(self.batch_size):
                img = batch['img'][i].numpy().transpose(1,2,0)
                sent = self.vocab.decode(batch['tgt_input'].T[i].tolist())
                
                plt.figure()
                plt.title('sent: {}'.format(sent), loc='center', fontname=fontname)
                plt.imshow(img)
                plt.axis('off')
                
                n += 1
                if n >= sample:
                    plt.show()
                    return


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        
        optim = ScheduledOptim(
	       Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            	self.config['transformer']['d_model'], **self.config['optimizer'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.iter = checkpoint['iter']

        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter':self.iter, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=torch.device(self.device))

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                print('{} not found'.format(name))
            elif state_dict[name].shape != param.shape:
                print('{} missmatching shape, required {} but found {}'.format(name, param.shape, state_dict[name].shape))
                del state_dict[name]

        self.model.load_state_dict(state_dict, strict=False)

    def save_weights(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
       
        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

        batch = {
                'img': img, 'tgt_input':tgt_input, 
                'tgt_output':tgt_output, 'tgt_padding_mask':tgt_padding_mask, 
                'filenames': batch['filenames']
                }

        return batch

    def data_gen(self, lmdb_path, data_root, annotation, masked_language_model=True, transform=None):
        dataset = OCRDataset(lmdb_path=lmdb_path, 
                root_dir=data_root, annotation_path=annotation, 
                vocab=self.vocab, transform=transform, 
                image_height=self.config['dataset']['image_height'], 
                image_min_width=self.config['dataset']['image_min_width'], 
                image_max_width=self.config['dataset']['image_max_width'])

        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        collate_fn = Collator(masked_language_model)

        gen = DataLoader(
                dataset,
                batch_size=self.batch_size, 
                sampler=sampler,
                collate_fn = collate_fn,
                shuffle=False,
                drop_last=False,
                **self.config['dataloader'])
       
        return gen

    def data_gen_v1(self, lmdb_path, data_root, annotation):
        data_gen = DataGen(data_root, annotation, self.vocab, 'cpu', 
                image_height = self.config['dataset']['image_height'],        
                image_min_width = self.config['dataset']['image_min_width'],
                image_max_width = self.config['dataset']['image_max_width'])

        return data_gen

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']    
        
        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
#        loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
        outputs = outputs.view(-1, outputs.size(2))#flatten(0, 1)
        tgt_output = tgt_output.view(-1)#flatten()
        
        loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) 

        self.optimizer.step()
        self.scheduler.step()

        loss_item = loss.item()

        return loss_item
