import torch
import os
import numpy as np
from model import models
from config import CFG
from utils import mmd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pre_data.ck_data_load import Cityscapes_Dataset    # 改
from test import test, logger

# from tensorboardX import SummaryWriter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
    train_loss_clf = checkpoint['train_loss_clf']
    train_loss_transfer = checkpoint['train_loss_transfer']
    train_loss_total = checkpoint['train_loss_total']
    epoch = checkpoint['epoch']
    return model, optimizer, train_loss_clf, train_loss_transfer, train_loss_total, epoch
    # model = load_checkpoint('checkpoint.pkl')


exp_data = []
def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG, train_loss_clf,
          train_loss_transfer, train_loss_total, Epoch=0, DA_ON=True):
    if DA_ON == True:
        for epoch in range(Epoch, CFG['epoch']):
            for iteration, ((source_images, source_targets), (target_images, _)) in enumerate(
                    zip(source_loader, target_train_loader)):

                # 处理batchsize不同
                if source_images.size(0) != target_images.size(0):
                    cut_size = min(source_images.size(0), target_images.size(0))
                    if source_images.size(0) > target_images.size(0):
                        source_images = source_images[:cut_size][:][:][:]
                        source_targets = source_targets[:cut_size][:][:][:]
                    else:
                        target_images = target_images[:cut_size][:][:]

                model.train()
                data_source, label_source = source_images.to(DEVICE), source_targets.to(DEVICE)
                data_target = target_images.to(DEVICE)
                optimizer.zero_grad()
                label_source_pred, transfer_loss = model(data_source, data_target, DA_ON=DA_ON)
                clf_loss = torch.nn.functional.binary_cross_entropy(label_source_pred, label_source)
                loss = clf_loss + CFG['lambda'] * transfer_loss
                loss.backward()
                optimizer.step()
                train_loss_clf.update(clf_loss.item())
                train_loss_transfer.update(transfer_loss.item())
                train_loss_total.update(loss.item())

                # log data
                loss_c =clf_loss.item()
                loss_t = transfer_loss.item()
                loss_total = train_loss_total.avg
                data = [loss_c, loss_t, loss_total]
                exp_data.append(data)

                if iteration % 20 == 0:
                    logger.info(
                        'Train iteration: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                            iteration + 1, int(2975 / CFG['batch_size']),
                            int(100 * (iteration + 1) / int(2966 / CFG['batch_size'])), train_loss_clf.avg,
                            train_loss_transfer.avg, train_loss_total.avg))
            if epoch % CFG['log_interval'] == 0:
                logger.info(
                    '--------------------------------------------------------------------------------------------------')
                logger.info(
                    'Train Epoch: [{}/{} ({:.1f}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                        epoch, CFG['epoch'], 100. * epoch / CFG['epoch'], train_loss_clf.avg,
                        train_loss_transfer.avg, train_loss_total.avg))
                checkpoint = {'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'train_loss_clf': train_loss_clf,
                              'train_loss_transfer': train_loss_transfer,
                              'train_loss_total': train_loss_total,
                              'epoch': epoch}
                torch.save(checkpoint, './C-K_checkpoint/da_checkpoint_%d.pkl' % (epoch))
                logger.info('da_checkpoint_%d.pkl save!' % (epoch))
                np.save('exp_data/loss_data_ck.npy', exp_data)
                test(model, target_test_loader, threshold=0.5)
                logger.info(
                    '--------------------------------------------------------------------------------------------------')
    else:
        for epoch in range(CFG['epoch']):
            for iteration, ((source_images, source_targets)) in enumerate(source_loader):
                model.train()
                data_source, label_source = source_images.to(DEVICE), source_targets.to(DEVICE)
                optimizer.zero_grad()
                label_source_pred = model(data_source, DA_ON=DA_ON)
                clf_loss = torch.nn.functional.binary_cross_entropy(label_source_pred, label_source)
                loss = clf_loss
                loss.backward()
                optimizer.step()
                train_loss_clf.update(clf_loss.item())
                train_loss_transfer.update(0)
                train_loss_total.update(loss.item())
                # writer.add_scalar('data/train_loss_total', train_loss_total.avg, i + e * n_batch)
                # writer.add_scalar('data/train_loss_clf', train_loss_clf.avg, i + e * n_batch)
                # writer.add_scalar('data/train_loss_transfer', train_loss_transfer.avg, i + e * n_batch)

                if iteration % 20 == 0:
                    logger.info(
                        'Train iteration: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                            iteration + 1, int(2975 / CFG['batch_size']),
                            int(100 * (iteration + 1) / int(2966 / CFG['batch_size'])), train_loss_clf.avg,
                            train_loss_transfer.avg, train_loss_total.avg))
            if epoch % CFG['log_interval'] == 0:
                logger.info(
                    '--------------------------------------------------------------------------------------------------')
                logger.info(
                    'Train Epoch: [{}/{} ({:.1f}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                        epoch + 1, CFG['epoch'], 100. * (epoch + 1. / CFG['epoch']), train_loss_clf.avg,
                        train_loss_transfer.avg, train_loss_total.avg))
                checkpoint = {'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'epoch': epoch}
                torch.save(checkpoint, './checkpoint/source_checkpoint_%d.pkl' % (epoch))
                logger.info('checkpoint_%d.pkl save!' % (epoch))
                test(model, target_test_loader, threshold=0.5)
                logger.info(
                    '--------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    model = models.Transfer_Net(CFG['n_class'], transfer_loss='mmd', base_net='resnet50').to(DEVICE)
    print(model)
    """cal total parameters"""
    total = sum(p.numel() for p in model.parameters())
    logger.info('Model Parameter Number: %d' % (total))

    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    # 加载checkpoint
    epoch_load = 0
    train_loss_clf = mmd.AverageMeter()
    train_loss_transfer = mmd.AverageMeter()
    train_loss_total = mmd.AverageMeter()
    if os.path.exists(CFG['checkpoint']):
        logger.info('model:{} loading...'.format(CFG['checkpoint']))
        model, optimizer, train_loss_clf, train_loss_transfer, train_loss_total, epoch_load = \
            load_checkpoint(model,optimizer,CFG['checkpoint'])
        logger.info('loading end')

    logger.info('Src: %s, Tar: %s', CFG['source_data'], CFG['target_data'])

    # log_dir = './data_logs/logs_DAN1_' + source_name + '_' + target_name
    # writer = SummaryWriter(log_dir=log_dir)

    source_train_dataset = Cityscapes_Dataset(image_root=CFG['source_img_path'], list_file=CFG['source_data_path'], train=True,
                                              transform=[transforms.ToTensor()])
    target_train_dataset = Cityscapes_Dataset(image_root=CFG['target_img_path'], list_file=CFG['target_data_path'], train=True,
                                              transform=[transforms.ToTensor()])
    target_test_dataset = Cityscapes_Dataset(image_root=CFG['target_img_path'], list_file=CFG['target_data_path'], train=False,    #路径改了, train改了
                                             transform=[transforms.ToTensor()])
    source_train_loader = DataLoader(source_train_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=2)
    target_train_loader = DataLoader(target_train_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=2)
    target_test_loader = DataLoader(target_test_dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=2)

    train(source_train_loader, target_train_loader, target_test_loader, model, optimizer, CFG, train_loss_clf,      # source和target改了
          train_loss_transfer, train_loss_total, epoch_load, DA_ON=CFG['DA_ON'])
    # test(model, target_test_loader, threshold=0.5)

