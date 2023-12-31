# coding:utf-8
import numpy as np
from PIL import Image

# 0:unlabeled, 1:car, 2:person, 3:bike, 4:curve, 5:car_stop, 6:guardrail, 7:color_cone, 8:bump

def get_palette():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128]
    person = [64, 64, 0]
    bike = [0, 128, 192]
    curve = [0, 0, 192]
    car_stop = [128, 128, 0]
    guardrail = [64, 64, 128]
    color_cone = [192, 128, 128]
    bump = [192, 64, 0]
    palette = np.array([unlabelled, car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(predictions):
    palette = get_palette()
    # for (i, pred) in enumerate(predictions):
    img = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    for cid in range(0, len(palette)):  # fix the mistake from the MFNet code on Dec.27, 2019
        img[predictions == cid] = palette[cid]
    return Image.fromarray(np.uint8(img))
        # img.save('Pred_' + weight_name + '_' + image_name)


def compute_results(conf_total):
    n_class = conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1

    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    # 每一类
    for cid in range(start_index, n_class):  # cid: class id
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = 0
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(
                conf_total[cid, 0:].sum())  # recall = TP/TP+FN

        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = 0
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:,cid].sum() -
                                                                      conf_total[cid, cid]))  # IoU = TP/TP+FP+FN

    # 全部
    MAcc = recall_per_class.sum()/9.
    MIOU = iou_per_class.sum()/9.

    return recall_per_class, iou_per_class, MAcc,MIOU
