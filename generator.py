#!/bin/env python3
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from data_parallel import get_data_parallel
from running_log import RunningLog

CLASSES = ['#', '+', '2', '3', '4', '6', '7', '8', '9', 'B', 'C', 'E', 'F',
           'G', 'H', 'J', 'K', 'M', 'P', 'Q', 'R', 'T', 'V', 'W', 'X', 'Y']
CLASSES_TO_ID = {k: i for i, k in enumerate(CLASSES)}


class CaptchaDecoder(nn.Module):
    def __init__(self, hidden_size=1024, n_classes=25, sequence_len=5):
        super(CaptchaDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.embedding = nn.Embedding(n_classes + 1, hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, n_classes + 1)

    def forward(self, state, device):  # n x hidden_size
        batch_size = state.size(0)
        inputs = torch.LongTensor([0] * batch_size).unsqueeze(0).to(device)  # 1 x n
        outputs = []
        all_symbols = []
        for _ in range(self.sequence_len):
            embedded = self.embedding(inputs)  # 1 x n x hidden_size
            output, state = self.rnn(embedded, state.unsqueeze(0))  # 1 x n x hidden_size
            predicted_softmax = F.log_softmax(self.linear(output.view(-1, self.hidden_size)), dim=1)  # n x classes
            state = state.squeeze(0)
            outputs.append(predicted_softmax)
            symbols = predicted_softmax.topk(1)[1]  # n x 1
            inputs = symbols.view(1, -1)
            all_symbols.append(symbols.view(-1))
        return torch.stack(outputs), torch.stack(all_symbols)  # seq x n x classes, seq x n


class CaptchaGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(  # 1 x 40 x 40
            nn.Conv2d(1, 32, kernel_size=5),  # 32 x 36 x 36
            nn.MaxPool2d(2),  # 32 x 18 x 18
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),  # 32 x 16 x 16
            nn.MaxPool2d(2),  # 32 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),  # 64 x 6 x 6
            nn.MaxPool2d(2),  # 64 x 3 x 3
            nn.ReLU(inplace=True),
        )
        for parameter in self.features.children():
            parameter.requires_grad = False
        self.mapper = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3 * 3 * 64 * 15, 1024),
            nn.ReLU(inplace=True),
        )
        self.decoder = CaptchaDecoder()

    def forward(self, x, device):  # n x 1 x 180 x 40
        chunks = []
        for i in range(15):
            chunks.append(x[:, :, :, i * 10:i * 10 + 40])
        characters = torch.stack(chunks).transpose(0, 1).reshape(-1, 1, 40, 40)
        features = torch.stack(self.features(characters).chunk(x.size(0))).view(x.size(0), -1)
        features = self.mapper(features)
        return self.decoder(features, device)


def load_epoch(save_path, epoch):
    tqdm.write('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(save_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')


class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.results = []
        for filename in sorted(os.listdir(root_dir)):
            name, ext = os.path.splitext(filename)
            name = name.split('.')
            if ext == '.jpeg' and len(name) == 2 and len(name[0]) == 32:
                self.results.append((filename, name[1]))

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        with open(os.path.join(self.root_dir, self.results[idx][0]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        code = [CLASSES_TO_ID[char] for char in self.results[idx][1]]
        code = code + (5 - len(code)) * [1]
        return img, torch.LongTensor(code)


def eval_model(model, valid_data_loader, device):
    criterion = nn.CrossEntropyLoss().to(device)
    total_count, correct_count = 0, 0
    losses = []
    for data in tqdm(valid_data_loader, desc='Eval'):
        data = [x.to(device) for x in data]
        total_count += data[1].size(0)
        output, predicate = model(data[0], device)
        loss = criterion(output.transpose(0, 1).reshape(-1, 26), data[1].view(-1))
        losses.append(loss.item())
        # noinspection PyUnresolvedReferences
        correct_count += (predicate.transpose(0, 1) == data[1]).all(dim=1).sum().item()
    return np.mean(losses), correct_count / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['train', 'valid', 'train-all'],
                        default='train', help='task to run')
    parser.add_argument('--dataset_path', help='path to the dataset folder',
                        default='dataset/whole')
    parser.add_argument('--save_path', help='path for saving models and codes',
                        default='save/generator')
    parser.add_argument('--classifier_save_path', help='path for saving models and codes',
                        default='save/classifier')
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids separated by `,'")
    parser.add_argument('--load', type=int, default=0,
                        help='load module training at give epoch')
    parser.add_argument('--load_classifier', type=int, default=200,
                        help='load module training at give epoch')
    parser.add_argument('--epoch', type=int, default=200, help='epoch to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--log_every_iter', type=int, default=100,
                        help='log loss every numbers of iteration')
    parser.add_argument('--valid_every_epoch', type=int, default=1,
                        help='run validation every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--save_every_epoch', type=int, default=1,
                        help='save model every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--comment', default='', help='comment for tensorboard')
    args = parser.parse_args()
    running_log = RunningLog(args.save_path)
    running_log.set('parameters', vars(args))
    os.makedirs(args.save_path, exist_ok=True)
    model = get_data_parallel(CaptchaGenerator(), args.gpu)
    device = torch.device("cuda:%d" % args.gpu[0] if args.gpu else "cpu")
    optimizer_state_dict = None
    if args.load > 0:
        model_state_dict, optimizer_state_dict = \
            load_epoch(args.save_path, args.load)
        model.load_state_dict(model_state_dict)
    elif args.load_classifier > 0:
        tqdm.write('loading from epoch.%04d.pth' % args.load_classifier)
        classifier_state_dict, _ = torch.load(os.path.join(
            args.classifier_save_path, 'epoch.%04d.pth' % args.load_classifier), map_location='cpu')
        model_stat_dict = model.state_dict()
        model_stat_dict.update({k: v for k, v in classifier_state_dict.items()
                                if not k.startswith('module.classifier')})
        model.load_state_dict(model_stat_dict)
    model.to(device)
    running_log.set('state', 'interrupted')
    if args.task == 'train' or args.task == 'train-all':
        model.train()
        # noinspection PyUnresolvedReferences
        train_dataset = CaptchaDataset(os.path.join(
            args.dataset_path, 'train' if args.task == 'train' else 'all'),
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]))
        train_data_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True)
        valid_data_loader = None
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        criterion = nn.NLLLoss().to(device)
        writer = SummaryWriter(comment=args.comment or os.path.basename(args.save_path))
        step = 0
        for epoch in tqdm(range(args.load + 1, args.epoch + 1), desc='Epoch'):
            losses = []
            for iter, data in enumerate(tqdm(train_data_loader, desc='Iter'), 1):
                data = [x.to(device) for x in data]
                output, predicate = model(data[0], device)  # seq x n x classes, seq x n
                loss = criterion(output.transpose(0, 1).reshape(-1, 26), data[1].view(-1))
                losses.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter % args.log_every_iter == 0:
                    # noinspection PyStringFormat
                    tqdm.write('epoch:[%d/%d] iter:[%d/%d] Loss=%.5f' %
                               (epoch, args.epoch, iter, len(train_data_loader),
                                np.mean(losses)))
                    losses = []
                step += 1
            if args.valid_every_epoch and epoch % args.valid_every_epoch == 0:
                if valid_data_loader is None:
                    # noinspection PyUnresolvedReferences
                    valid_dataset = CaptchaDataset(os.path.join(args.dataset_path, 'test'),
                                                   transform=transforms.Compose([
                                                       transforms.Grayscale(),
                                                       transforms.ToTensor(),
                                                   ]))
                    valid_data_loader = DataLoader(valid_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
                model.eval()
                loss, acc = eval_model(model, valid_data_loader, device)
                # noinspection PyStringFormat
                tqdm.write('Loss=%f Accuracy=%f' % (loss, acc))
                writer.add_scalar('eval/loss', loss, epoch)
                writer.add_scalar('eval/acc', acc, epoch)
                model.train()
            if args.save_every_epoch and epoch % args.save_every_epoch == 0:
                tqdm.write('saving to epoch.%04d.pth' % epoch)
                torch.save((model.state_dict(), optimizer.state_dict()),
                           os.path.join(args.save_path,
                                        'epoch.%04d.pth' % epoch))

    running_log.set('state', 'succeeded')


if __name__ == '__main__':
    main()
