import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import tqdm

from model import LanguageModel

import os
from adaptive_softmax import *
from utils import *
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Benchmark for Adaptive Softmax')
parser.add_argument('--with_adaptive', type=int, default=1,
        help=('Whether use Adaptive Softmax or not , '
            '0 for common full softmax'))
parser.add_argument('--train_file_path', type=str,
                    default='./data/text8.train.txt')
parser.add_argument('--test_file_path', type=str,
                    default='./data/text8.valid.txt')

parser.add_argument('--train_pkl_path', type=str,
                    default='train.pkl')
parser.add_argument('--test_pkl_path', type=str,
                    default='test.pkl')
parser.add_argument('--dict_pkl_path', type=str,
                    default='dict.pkl')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--bptt', type=int, default=20)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--clip_global_norm_rate', type=float, default=0.25)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--learning_rate_decay', type=float, default=1e-5)
parser.add_argument('--pass_num', type=int, default=5)

parser.add_argument('--cutoff', type=str, default="2000,10000")

args = parser.parse_args()

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)

    else:
        return tuple(repackage_hidden(v) for v in h)

def clip_global_norm(model, clip):
    norms = []
    total_norm = 0

    for p in model.parameters():
        norm = p.grad.data.norm()

        if norm > clip:
            p.grad.data.div_(max(norm, 1e-6) / clip)

cutoff_list = [int(item) for item in args.cutoff.split(',')]

with_adaptive = True
if args.with_adaptive == 0:
    with_adaptive = False

train_file_path = args.train_file_path
test_file_path = args.test_file_path

train_pkl_path = args.train_pkl_path
test_pkl_path = args.test_pkl_path
dict_pkl_path = args.dict_pkl_path

if os.path.exists(dict_pkl_path) == False:
    print('The dict.pkl is not exists.')
    word_dict = build_dict(train_file_path)
    with open(dict_pkl_path, 'wb') as f:
        pickle.dump(word_dict, f)

with open(dict_pkl_path, 'rb') as f:
    word_dict = pickle.load(f)

dict_size = len(word_dict)
print("the dict len is:",dict_size)

if os.path.exists(train_pkl_path) == False:
    train_input, train_label = get_batch(train_file_path,
                                         word_dict,
                                         batch_size=args.batch_size,
                                         bptt = args.bptt)
    with open(train_pkl_path, 'wb') as f:
        pickle.dump({'data':train_input, 'label':train_label}, f)

if os.path.exists(test_pkl_path) == False:
    test_input, test_label = get_batch(test_file_path,
                                       word_dict,
                                       batch_size=args.batch_size,
                                       bptt = args.bptt)
    with open(test_pkl_path, 'wb') as f:
        pickle.dump({'data':test_input, 'label':test_label}, f)
        
with open(train_pkl_path, 'rb') as f:
    train_data = pickle.load(f)
with open(test_pkl_path, 'rb') as f:
    test_data = pickle.load(f)


model = LanguageModel(dict_size,
                      args.hidden_size,
                      args.hidden_size,
                      n_layer=1,
                      drop_rate=args.drop_rate,
                      adaptive_softmax=with_adaptive,
                      cutoff=cutoff_list)
model#.cuda()
optimizer = optim.Adagrad(model.parameters(),
                          lr=args.learning_rate,
                          lr_decay=args.learning_rate_decay,
                          weight_decay=args.weight_decay)

if with_adaptive:
    print('Use adaptive softmax.')
    criterion = AdaptiveLoss(cutoff_list)
else:
    print('Use common softmax.')
    criterion = nn.CrossEntropyLoss()

def train(batch_size, clip_global_norm_rate):
    pbar = tqdm.tqdm(zip(train_data['data'], train_data['label']))
    hidden = model.init_hidden(batch_size)

    for X_batch, Y_batch in pbar:
        X_tensor = torch.from_numpy(X_batch)#.cuda()
        Y_tensor = torch.from_numpy(Y_batch.astype(np.int))#.cuda()

        X_var, Y_var = Variable(X_tensor), Variable(Y_tensor.view(-1))
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(X_var, hidden, Y_var)
        loss = criterion(output, Y_var)
        loss.backward()
        clip_global_norm(model, clip_global_norm_rate)
        optimizer.step()
        pbar.set_description('Loss: {:.3f}'.format(loss.data[0]))


def test(batch_size, bptt):
    pbar = tqdm.tqdm(zip(test_data['data'], test_data['label']))

    if with_adaptive:
        criterion = nn.NLLLoss(size_average=False)

    else:
        criterion = nn.CrossEntropyLoss(size_average=False)

    nllloss = 0
    hidden = model.init_hidden(batch_size)

    for X_batch, Y_batch in pbar:
        X_tensor = torch.from_numpy(X_batch)#.cuda()
        Y_tensor = torch.from_numpy(Y_batch.astype(np.int))#.cuda()
        X_var, Y_var = Variable(X_tensor), Variable(Y_tensor.view(-1))
        hidden = repackage_hidden(hidden)

        if with_adaptive:
            output, hidden = model.log_prob(X_var, hidden, Y_var)
            nllloss += criterion(Variable(output), Y_var).data[0]

        else:
            output, hidden = model(X_var, hidden, Y_var, training=False)
            nllloss += criterion(output, Y_var).data[0]


    loss = nllloss / (len(test_data['data']) * batch_size * bptt)

    print('Perplexity:', np.exp(loss))

    return loss

for epoch in range(args.pass_num):
    train(args.batch_size, args.clip_global_norm_rate)
    test(args.batch_size, args.bptt)

