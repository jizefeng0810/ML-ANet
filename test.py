import torch
import numpy as np
from config import CFG
from utils.logger import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = Logger('./CK_all_log.txt', level='info')  # 保存输出信息
logger = log.logger


def test(model, target_test_loader, threshold=0.5):
    model.eval()
    TP = np.zeros((1, CFG['n_class']))
    FP = np.zeros((1, CFG['n_class']))
    FN = np.zeros((1, CFG['n_class']))
    TN = np.zeros((1, CFG['n_class']))
    with torch.no_grad():
        for iteration, (data, target) in enumerate(target_test_loader):
            data, target = data.to(device), target.to(device)
            pred = model.predict(data)
            pred = pred > threshold
            pred = pred.cpu().numpy().astype(np.int)
            target = target.cpu().numpy().astype(np.int)
            tp = np.sum(np.logical_and(np.equal(target, 1), np.equal(pred, 1)), axis=0)
            fp = np.sum(np.logical_and(np.equal(target, 0), np.equal(pred, 1)), axis=0)
            fn = np.sum(np.logical_and(np.equal(target, 1), np.equal(pred, 0)), axis=0)
            tn = np.sum(np.logical_and(np.equal(target, 0), np.equal(pred, 0)), axis=0)
            TP = TP + tp
            FP = FP + fp
            FN = FN + fn
            TN = TN + tn
    Acc = (TP + TN) / (TP + FP + FN + TN)
    AP = TP / (TP + FP)
    RC = TP / (TP + FN)
    each_acc = np.mean(Acc, axis=0)
    each_ap = np.mean(AP, axis=0)
    each_rc = np.mean(RC, axis=0)
    logger.info('{} --> {}: '.format(CFG['source_data'], CFG['target_data']))
    logger.info(
        'each accuracy: person/rider:{:.3f} car/truck/bus:{:.3f} motorcycle/bicycle:{:.3f}'
        .format(each_acc[0], each_acc[1], each_acc[2]))
    logger.info(
        'each precision: person/rider:{:.3f} car/truck/bus:{:.3f} motorcycle/bicycle:{:.3f}'
        .format(each_ap[0], each_ap[1], each_ap[2]))
    logger.info(
        'each recall: person/rider:{:.3f} car/truck/bus:{:.3f} motorcycle/bicycle:{:.3f}'
        .format(each_rc[0], each_rc[1], each_rc[2]))
    logger.info('average accuracy{: .2f}%'.format(100. * np.mean(Acc)))
    logger.info('average precision{: .2f}%'.format(100. * np.mean(AP)))
    logger.info('average recall{: .2f}%'.format(100. * np.mean(RC)))
