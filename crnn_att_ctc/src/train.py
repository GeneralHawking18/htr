import os

import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

# from src.dataset import Synth90kDataset, synth90k_collate_fn
from src.model import * # CRNN
from src.evaluate import evaluate
from src.config import train_config as config


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]
    
    logits = crnn(images)[2: ]
    # print(logits.shape)
    # logits = logits.permute(1,0,2).contiguous().requires_grad_(True)
    log_probs = logits # torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.IntTensor([logits.size(0) - 2] * batch_size)
    # target_lengths = torch.IntTensor([len(t) for t in targets])
    # target_lengths = torch.flatten(target_lengths)
    print("log_probs, targets, input_lengths, target_lengths: ", \
    log_probs.shape, targets.shape, input_lengths.shape, target_lengths.shape)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()
