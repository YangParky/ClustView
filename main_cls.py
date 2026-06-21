from __future__ import print_function
import os
import string
import random
import logging
import shutil
import hydra
import omegaconf
import torch

import numpy as np
import sklearn.metrics as metrics
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from data.data_util import ModelNet40, ScanObjectNN
from models.clustview import ClustView_cls
from util import cal_loss, encode_onehot, find_free_port, Wandb, rsmix_provider, profile_model


def _init_(args):

    # fix random seed
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(find_free_port())

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_printoptions(10)
        cudnn.benchmark = True
        cudnn.deterministic = True


def train(args):

    omegaconf.OmegaConf.set_struct(args, False)
    writer = SummaryWriter(log_dir=args.log_dir)
    logger = logging.getLogger(__name__)
    logger.info('Working path: %s' % str(os.getcwd()))
    logger.info('Random seed is set to %s...' % str(args.seed))

    # data loading
    logger.info('Load %s dataset...' % args.dataset)
    DATA_PATH = hydra.utils.to_absolute_path(args.dataset_dir)

    if args.dataset == 'ModelNet40':
        train_loader = DataLoader(ModelNet40(DATA_PATH, partition='train', num_points=args.num_points), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(ModelNet40(DATA_PATH, partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False, pin_memory=True)
    elif args.dataset == 'ScanObjectNN':
        train_loader = DataLoader(ScanObjectNN(DATA_PATH, split=args.split, partition='training', num_points=args.num_points), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(ScanObjectNN(DATA_PATH, split=args.split, partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False, pin_memory=True)
    else:
        raise NotImplementedError     

    # model loading

    logger.info('Load %s model...' % args.model_name)
    model = ClustView_cls(args).cuda()
    logger.info(model)

    if args.use_sgd:
        logger.info("Use SGD...")
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        logger.info("Use AdamW")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.7)

    if args.warm_up:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_up_epoch, after_scheduler=scheduler)

    criterion = cal_loss

    shutil.copy(hydra.utils.to_absolute_path('models/' + args.model_name + '.py'), '.')
    shutil.copy(hydra.utils.to_absolute_path('main_cls.py'), '.')
    shutil.copy(hydra.utils.to_absolute_path('config/data/' + args.dataset.lower() + '.yaml'), '.')
    shutil.copy(hydra.utils.to_absolute_path('config/train_cls.yaml'), '.')

    try:
        checkpoint = torch.load('model.pth')
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_test_acc = checkpoint['test_acc']
        logger.info('Use pretrain model...')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0
        best_epoch = 0
        best_test_acc = 0

    # start training
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epochs):
        logger.info('Epoch (%d/%s):' % (epoch + 1, args.epochs))
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []

        for data, label in tqdm(train_loader):
            '''
            RSMIX Augmentation, inhereted from
            https://github.com/dogyoonlee/RSMix
            '''
            rsmix = False            
            r = np.random.rand(1)
            if args.beta > 0 and r < args.rsmix_prob:
                rsmix = True
                data = data.cpu().numpy() 
                data, lam, label, label_b = rsmix_provider.rsmix(
                    data, label, beta=args.beta, n_sample=args.rsmix_nsample)
                data = torch.FloatTensor(data)
                lam = torch.FloatTensor(lam) 
                if args.dataset == 'ScanObjectNN':
                    label = torch.FloatTensor(label)
                    label_b = torch.FloatTensor(label_b)
                lam, label_b = lam.cuda(), label_b.cuda().squeeze()   
                        
            data, label = data.cuda(), label.cuda().squeeze()

            if rsmix:
                batch_size = data.size()[0]
                optimizer.zero_grad()

                logits = model(data)
                loss = 0
                for i in range(batch_size):
                    loss_tmp = criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1-lam[i]) \
                               + criterion(logits[i].unsqueeze(0), label_b[i].unsqueeze(0).long()) * lam[i]
                    loss += loss_tmp

                loss = loss / batch_size
            else:
                batch_size = data.size()[0]
                optimizer.zero_grad()

                logits = model(data)
                loss = criterion(logits, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if optimizer.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if optimizer.param_groups[0]['lr'] < 1e-5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss = train_loss*1.0/count
        train_acc = metrics.accuracy_score(train_true, train_pred)
        logger.info('Train loss: %.6f, train acc: %.6f, lr: %.6f' % (train_loss, train_acc, optimizer.param_groups[0]['lr']))
        
        ####################
        # Test
        ####################
        with torch.no_grad():
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            for data, label in test_loader:
                data, label = data.cuda(), label.cuda().squeeze()
                batch_size = data.size()[0]

                logits = model(data)
                loss = criterion(logits, label)

                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_loss = test_loss*1.0/count
            test_acc = metrics.accuracy_score(test_true, test_pred)
            logger.info('Test loss: %.6f, test acc: %.6f' % (test_loss, test_acc))

            writer.add_scalars("Loss", {'train':train_loss, 'test':test_loss}, epoch)
            writer.add_scalars("Acc", {'train':train_acc, 'test':test_acc}, epoch)

            if test_acc >= best_test_acc:
                logger.info('Save model...')
                best_test_acc = test_acc
                best_epoch = epoch + 1
                state = {
                        'epoch': best_epoch,
                        'test_acc': test_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                torch.save(state, 'model.pth')
                
            logger.info('best: %.3f' % best_test_acc)
            writer.add_scalar('best_test', best_test_acc, epoch)

    # end of training
    logger.info('End of training...')
    writer.add_scalar('test_oa', best_test_acc, best_epoch)

    writer.flush()
    writer.close()


def test(args):
    logger = logging.getLogger(__name__)

    # data loading
    DATA_PATH = hydra.utils.to_absolute_path(args.dataset_dir)
    if args.dataset == 'ModelNet40':
        test_loader = DataLoader(ModelNet40(DATA_PATH, partition='test', num_points=args.num_points), num_workers=16,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)                                      
    elif args.dataset == 'ScanObjectNN':
        test_loader = DataLoader(ScanObjectNN(DATA_PATH, partition='test', num_points=args.num_points), num_workers=16,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise NotImplementedError     

    # model loading

    model = ClustView_cls(args).cuda()
    state_dict = torch.load('model.pth')
    state_dict = state_dict['model_state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[0:]  # remove `module.`, 表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    model.load_state_dict(new_state_dict)

    logger.info('Start Testing... ')
    model = model.eval()
    test_true = []
    test_pred = []
    for data, label in tqdm(test_loader):
        data, label = data.cuda(), label.cuda().squeeze()
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    test_acc_avg = metrics.balanced_accuracy_score(test_true, test_pred)
    logger.info('test acc: %.6f'%(test_acc))
    logger.info('test avg acc: %.6f'%(test_acc_avg))

    # if args.flops_profiler:
    #     input = [torch.randn_like(data)]
    #     flops, macs, params = profile_model(model, input)
    #     logger.info(f'GFLOPs\tGMACs\tParams.(M)')
    #     logger.info(f'{flops/(float(batch_size)*1e9): .2f}\t{macs/(float(batch_size)*1e9): .2f}\t{params/1e6: .3f}')


@hydra.main(config_path='config', config_name='train_cls')
def main(args):

    if args.seed is None:
        args.seed = np.random.randint(1, 10000)

    _init_(args)

    if not args.eval:
        Wandb.launch(args, args.wandb.use_wandb)
        train(args)
    else:
        with torch.no_grad():
            test(args)


if __name__ == "__main__":
    main()
