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

TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
cfg = model.cfg
args = util.get_args(preset.PRESET)
if not torch.cuda.is_available():
    raise RuntimeError("Need cuda devices")
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")


def fit(args, cfg, net, dataset, optimizer, is_train):
    def avg(list):
        return sum(list) / len(list)
    if is_train:
        net.train()
    else:
        net.eval()
    Loss_L, Loss_C = [], []
    epoch_eval_result = {}
    for epoch in range(args.epoches_per_phase):
        visualize = False
        if args.curr_epoch != 0 and args.curr_epoch % 10 == 0 and epoch == 0:
            print("Visualizing prediction result...")
            visualize = True
        start_time = time.time()
        criterion = MultiBoxLoss(cfg, neg_pos=3)
        # Update variance and balance of loc_loss and conf_loss
        cfg['variance'] = [var * cfg['var_updater'] if var <= 0.95 else 1 for var in cfg['variance']]
        cfg['alpha'] *= cfg['alpha_updater']
        for batch_idx, (images, targets) in enumerate(dataset):
            #if not net.fix_size:
                #assert images.size(0) == 1, "batch size for dynamic input shape can only be 1 for 1 GPU RIGHT NOW!"
            images = images.cuda()
            ratios = images.size(3) / images.size(2)
            if ratios != 1.0:
                print(ratios)
            targets = [ann.cuda() for ann in targets]
            out = net(images, is_train)
            if args.curr_epoch == 0 and batch_idx == 0:
                visualize_bbox(args, cfg, images, targets, net.module.prior, batch_idx)
            if is_train:
                loss_l, loss_c = criterion(out, targets, ratios)
                loss = loss_l + loss_c
                Loss_L.append(float(loss_l.data))
                Loss_C.append(float(loss_c.data))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                if len(targets) == 0:
                    continue
                eval_result = evaluate(images, out.data, targets, batch_idx, visualize=visualize)
                for key in eval_result.keys():
                    if key in epoch_eval_result:
                        epoch_eval_result[key] += eval_result[key]
                    else:
                        epoch_eval_result.update({key: eval_result[key]})
        if is_train:
            args.curr_epoch += 1
            print(" --- loc loss: %.4f, conf loss: %.4f, at epoch %04d, cost %.2f seconds ---" %
                  (avg(Loss_L), avg(Loss_C), args.curr_epoch + 1, time.time() - start_time))
    if not is_train:
        for key in sorted(epoch_eval_result.keys()):
            eval = np.mean(np.asarray(epoch_eval_result[key]).reshape((-1, 4)), axis=0)
            print(" --- Conf=%s: accuracy=%.4f, precision=%.4f, recall=%.4f, f1-score=%.4f  ---" %
                  (key, eval[0], eval[1], eval[2], eval[3]))
        print("")
        # represent accuracy, precision, recall, f1_score
        return  eval[0], eval[1], eval[2], eval[3]
    else:
        return avg(Loss_L), avg(Loss_C)


def val(args, cfg, net, dataset, optimizer, prior):
    with torch.no_grad():
        fit(args, cfg, net, dataset, optimizer, prior, False)


def evaluate(img, detections, targets, batch_idx, visualize=False):
    conf_thresholds = [0.1, 0.2, 0.3, 0.4]
    eval_result = {}
    for threshold in conf_thresholds:
        idx = detections[0, 1, :, 0] >= threshold
        boxes = detections[0, 1, idx, 1:]
        gt_boxes = targets[0][:, :-1].data

        if gt_boxes.size(0) == 0:
            print("No ground truth box in this patch")
            break
        if boxes.size(0) == 0:
            print("No predicted box in this patch")
            break

        # Eliminate overlap area smaller than 0.5
        jac = jaccard(boxes, gt_boxes)
        overlap, idx = jac.max(1, keepdim=True)
        text_boxes = boxes[overlap.squeeze(1) > 0.5]
        text_boxes_eliminated = boxes[overlap.squeeze(1) <= 0.5]
        if text_boxes_eliminated.size(0) == 0:
            text_boxes_eliminated = tuple()

        accuracy, precision, recall = measure(text_boxes, gt_boxes)
        if (recall + precision) < 1e-3:
            f1_score = 0
        else:
            f1_score = 2 * (recall * precision) / (recall + precision)
        if visualize and threshold == 0.4:
            pred = [[float(coor) for coor in area] for area in text_boxes]
            gt = [[float(coor) for coor in area] for area in gt_boxes]
            print_box(text_boxes_eliminated, green_boxes=gt, blue_boxes=pred, idx=batch_idx,
                      img=vb.plot_tensor(args, img, margin=0), save_dir=args.val_log)
        eval_result.update({threshold: [accuracy, precision, recall, f1_score]})
    return eval_result


def measure(pred_boxes, gt_boxes):
    if pred_boxes.size(0) == 0:
        return 0.0, 0.0, 0.0
    inter = intersect(pred_boxes, gt_boxes)
    text_area = get_box_size(pred_boxes)
    gt_area = get_box_size(gt_boxes)
    num_sample = max(text_area.size(0),  gt_area.size(0))
    accuracy = torch.sum(jaccard(pred_boxes, gt_boxes).max(0)[0]) / num_sample
    precision = torch.sum(inter.max(1)[0] / text_area) / num_sample
    recall = torch.sum(inter.max(0)[0] / gt_area) / num_sample
    return float(accuracy), float(precision), float(recall)


def main():
    if args.fix_size:
        aug = aug_sroie()
        #aug_test = aug_sroie()
    else:
        aug = aug_sroie_dynamic_2()
        args.batch_size_per_gpu = 1
    datasets = data.fetch_detection_data(args, sources=args.train_sources, k_fold=1,
                                         batch_size=args.batch_size_per_gpu * torch.cuda.device_count(),
                                         batch_size_val=1, auxiliary_info=args.train_aux, split_val=0.1,
                                         pre_process=None, aug=aug)
    for idx, (train_set, val_set) in enumerate(datasets):
        loc_loss, conf_loss = [], []
        accuracy, precision, recall, f1_score = [], [], [], []
        print("\n =============== Cross Validation: %s/%s ================ " %
              (idx + 1, len(datasets)))
        net = model.SSD(cfg, connect_loc_to_conf=True, fix_size=args.fix_size,
                        incep_conf=True, incep_loc=True)
        net = torch.nn.DataParallel(net)
        # Input dimension of bbox is different in each step
        torch.backends.cudnn.benchmark = True
        net = net.cuda()
        if args.fix_size:
            net.module.prior = net.module.prior.cuda()
        if args.finetune:
            net = util.load_latest_model(args, net, prefix="dilation")
        # Using the latest optimizer, better than Adam and SGD
        optimizer = AdaBound(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,)

        for epoch in range(args.epoch_num):
            loc_avg, conf_avg = fit(args, cfg, net, train_set, optimizer, is_train=True)
            loc_loss.append(loc_avg)
            conf_loss.append(conf_avg)
            train_losses = [np.asarray(loc_loss), np.asarray(conf_loss)]
            if val_set is not None:
                accu, pre, rec, f1 = fit(args, cfg, net, val_set, optimizer, is_train=False)
                accuracy.append(accu)
                precision.append(pre)
                recall.append(rec)
                f1_score.append(f1)
                val_losses = [np.asarray(accuracy), np.asarray(precision),
                              np.asarray(recall), np.asarray(f1_score)]
            if epoch != 0 and epoch % 20 == 0:
                util.save_model(args, args.curr_epoch, net.state_dict(), prefix="cv_%s" % (idx + 1),
                                keep_latest=20)
            if epoch > 5:
                # Train losses
                vb.plot_loss_distribution(train_losses, ["location", "confidence"], args.loss_log, dt + "_loss", window=5)
                # Val metrics
                vb.plot_loss_distribution(val_losses, ["Accuracy", "Precision", "Recall", "F1-Score"], args.loss_log,
                                          dt + "_val", window=5, bound=[0.0, 1.0])
        # Clean the data for next cross validation
        del net, optimizer
        args.curr_epoch = 0


if __name__ == "__main__":
    #test_rotation()
    main()


