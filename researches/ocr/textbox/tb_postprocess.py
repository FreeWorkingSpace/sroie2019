import os, time, sys, math, random, glob, datetime
sys.path.append(os.path.expanduser("~/Documents/sroie2019"))
import cv2, torch
import numpy as np
import omni_torch.utils as util
import researches.ocr.textbox as init
import researches.ocr.textbox.tb_data as data
import researches.ocr.textbox.tb_preset as preset
import researches.ocr.textbox.tb_model as model
from researches.ocr.textbox.tb_loss import MultiBoxLoss
from researches.ocr.textbox.tb_utils import *
from researches.ocr.textbox.tb_preprocess import *
from researches.ocr.textbox.tb_augment import *
from researches.ocr.textbox.tb_vis import visualize_bbox, print_box
from omni_torch.networks.optimizer.adabound import AdaBound
import omni_torch.visualize.basic as vb

def combine_boxes(prediction, w, h, y_thres=2, combine_thres=0.8, overlap_thres=0.0):
    save_dir = os.path.expanduser("~/Pictures/")
    #print_box(red_boxes=prediction, shape=(h, w), step_by_step_r=True, save_dir=save_dir)
    output_box = []
    _scale = torch.Tensor([w, h, w, h])
    if prediction.is_cuda:
        _scale = _scale.cuda()
    scale = _scale.unsqueeze(0).repeat(prediction.size(0), 1)
    prediction = prediction * scale

    # Merge the boxes contained in other boxes
    merged_boxes = []
    unmerge_idx = torch.ones(prediction.size(0))
    inter = intersect(prediction, prediction)
    pred_size = get_box_size(prediction).unsqueeze(0).expand_as(inter)
    identity = torch.eye(inter.size(0)).cuda() if inter.is_cuda else torch.eye(inter.size(0))
    #indicator = 2 / (inter / pred_size + pred_size / inter) - identity
    indicator = (inter / pred_size - identity) > combine_thres
    for idctr in indicator:
        # eliminate the index of predicted boxes that need to be merged
        unmerge_idx[idctr] = 0
        idx = idctr.unsqueeze(0).expand_as(indicator)
        # once a box is merged, it does not need tp be merged or calculated again
        indicator[idx] = 0
        merged_boxes.append(
            torch.cat([torch.min(prediction[idctr][:, :2], dim=0)[0], torch.max(prediction[idctr][:, 2:], dim=0)[0]])
        )
    merged_boxes = torch.stack(merged_boxes, dim=0)
    unmerged_boxes = prediction[unmerge_idx]
    prediction = torch.cat([unmerged_boxes, merged_boxes], dim=0)

    # Find boxes with similar height
    vertical_height = (prediction[:, 3] - prediction[:, 1])
    vertical_height = vertical_height.unsqueeze(0).repeat(vertical_height.size(0), 1)
    dis_matrix = torch.abs(vertical_height - vertical_height.permute(1, 0))
    idx_h = dis_matrix < y_thres

    # Find boxes at almost same height
    vertical_height = (prediction[:, 3] + prediction[:, 1]) / 2
    vertical_height = vertical_height.unsqueeze(0).repeat(vertical_height.size(0), 1)
    dis_matrix = torch.abs(vertical_height - vertical_height.permute(1, 0))
    idx_v = dis_matrix < y_thres

    idx = idx_h * idx_v

    # Iterate idx in axis=0
    eliminated_box_id = set([])
    for i, box_id in enumerate(idx):
        #print(i)
        if i in eliminated_box_id:
            continue
        if int(torch.sum(box_id[i:])) == 1:
            output_box.append(prediction[i, :] / _scale)
            eliminated_box_id.add(i)
        else:
            # boxes that have the potential to be connected
            _box_id = np.where(box_id.cpu().numpy() == 1)[0]
            qualify_box = prediction[box_id]
            overlaps = jaccard(qualify_box, qualify_box)
            similar_boxes = overlaps > overlap_thres
            for j, similar_id in enumerate(similar_boxes):
                #print(similar_boxes)
                # Make the lower triangle part to be 0
                similar_id[:j] = 0
                if int(torch.sum(similar_id)) == 0:
                    continue
                elif int(torch.sum(similar_id)) == 1:
                    # this box has no intersecting boxes
                    eliminated_box_id.add(_box_id[(similar_id > 0).nonzero().squeeze()])
                    output_box.append(qualify_box[similar_id].squeeze() / _scale)
                else:
                    comb_boxes = qualify_box[similar_id]
                    # Combine comb_boxes
                    new_box = torch.cat([torch.min(comb_boxes[:, :2], dim=0)[0], torch.max(comb_boxes[:, 2:], dim=0)[0]])
                    output_box.append(new_box / _scale)
                    eliminated_box_id = eliminated_box_id.union(set(_box_id[(similar_id.cpu() > 0).nonzero().squeeze()]))
                # Eliminate the boxes that already been combined
                zero_id = similar_id.unsqueeze(0).repeat(similar_boxes.size(0), 1)
                similar_boxes[zero_id] = 0
    output = torch.stack(output_box, dim=0)
    return output



if __name__ == "__main__":
    box_num = 512
    square = 1024
    origin = torch.empty(box_num, 2).uniform_(0, 1)
    delta = torch.empty(box_num, 2).uniform_(0, 0.2)
    pred = torch.cat([origin, origin + delta], dim=1).clamp_(min=0, max=1).cuda()
    combinition = combine_boxes(pred, square, square)
    #print_box(combinition, shape=(square, square))
