import os, time, sys, math, random, datetime
sys.path.append(os.path.expanduser("~/Documents/sroie2019"))
import numpy as np
import cv2, torch, distance
import torch.nn as nn
import omni_torch.visualize.basic as vb
import omni_torch.utils as util
import researches.ocr.attention_ocr.aocr_data as data
import researches.ocr.attention_ocr.aocr_presets as preset
import researches.ocr.attention_ocr.aocr_models as att_model
from omni_torch.networks.optimizer.adabound import AdaBound
from researches.ocr.attention_ocr.aocr_augment import *
from researches.ocr.attention_ocr.aocr_util import *
from researches.ocr.attention_ocr.aocr_args import *
import researches.ocr.attention_ocr as init

opt = parse_arguments()
edict = util.get_args(preset.PRESET)
args = util.cover_edict_with_argparse(opt, edict)

invert_dict = invert_dict(args.label_dict)
if not torch.cuda.is_available():
    raise RuntimeError("Need cuda devices")
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
"""
Useful Tips:
https://danijar.com/tips-for-training-recurrent-neural-networks/
"""


def fit(args, encoder, decoder, dataset, encode_optimizer, decode_optimizer, criterion, is_train=True):
    if is_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()
    Loss = []
    Lev_Dis, Str_Accu = [], []
    decoder.module.teacher_forcing_ratio *= args.teacher_forcing_ratio_decay
    for epoch in range(args.epoches_per_phase):
        visualize = False
        if args.curr_epoch % 5 == 0 and epoch == 0:
            print("Visualizing prediction result at %d th epoch %d th iteration" % (args.curr_epoch, epoch))
            visualize = True
        start_time = time.time()
        for batch_idx, data in enumerate(dataset):
            img_batch, label_batch = data[0][0].cuda(), data[0][1].cuda()
            encoder_outputs = encoder(img_batch)
            # Decoder input is default the index of SOS token
            #input = torch.zeros([encoder_outputs.size(0), 1]).long().cuda() + args.label_dict["SOS"]
            outputs, attentions = decoder(x=encoder_outputs, y=label_batch, is_train=is_train)
            loss = [criterion(outputs[:, :, i], label_batch[:, i]) for i in range(outputs.size(2))]
            loss = sum(loss) / len(loss)
            Loss.append(float(loss))
            if is_train:
                encode_optimizer.zero_grad()
                decode_optimizer.zero_grad()
                loss.backward()
                encode_optimizer.step()
                decode_optimizer.step()
            else:
                # Measure the string level accuracy
                index = [outputs[:, :, i].topk(1)[1] for i in range(outputs.size(2))]
                index = torch.cat(index, dim=1)
                pred_str = []
                for idx in index:
                    pred_str.append("".join([invert_dict[int(i)] for i in idx]))
                label_str = []
                for idx in label_batch:
                    label_str.append("".join([invert_dict[int(i)] for i in idx]))
                # Calculate Levelstein
                lev_dist = [distance.levenshtein(pred_str[i], label_str[i]) for i in range(len(label_str))]
                Lev_Dis.append(avg(lev_dist))
                # Calculate String Level Accuracy
                correct = [100 if label == pred_str[i] else 0 for i, label in enumerate(label_str)]
                Str_Accu.append(avg(correct))
                print_pred_and_label(pred_str, label_str, print_correct=True)
        if is_train:
            args.curr_epoch += 1
            print(" --- Pred loss: %.4f, at epoch %04d, cost %.2f seconds ---" %
                  (avg(Loss),  args.curr_epoch + 1, time.time() - start_time))
        else:
            print(" --- Levenstein Distance = %.2f,  String Level Accuracy = %.2f  ---" %
                  (avg(Lev_Dis), avg(Str_Accu)))
    if is_train:
        return avg(Loss)
    else:
        return avg(Lev_Dis), avg(Str_Accu)
        

def visualize_attention(img_batch, label_batch, attention):
    output_images = []
    # Iterate all batches
    for i in range(label_batch.size(0)):
        char_len = 0
        for j in range(label_batch.size(1)):
            if int(label_batch[i, j]) == 20:
                break
            char_len += 1


def main():
    aug = aug_aocr(args)
    datasets = data.fetch_data(args, args.datasets, batch_size=args.batch_size_per_gpu,
                              batch_size_val=args.batch_size_per_gpu_val, k_fold=1, split_val=0.1,
                               pre_process=None, aug=aug)

    for idx, (train_set, val_set) in enumerate(datasets):
        losses = []
        lev_dises, str_accus = [], []
        print("\n =============== Cross Validation: %s/%s ================ " %
                  (idx + 1, len(datasets)))
        # Prepare Network
        encoder = att_model.Attn_CNN(backbone_require_grad=True)
        decoder = att_model.AttnDecoder(args)
        criterion = nn.NLLLoss()
        if args.finetune:
            encoder, decoder = util.load_latest_model(args, [encoder, decoder],
                                                      prefix=["ori_encoder", "ori_decoder"])
        else:
            # Missing Initialization functions for RNN in CUDA
            # splitting initialization on CPU and GPU respectively
            decoder.apply(init.init_rnn).to(args.device).apply(init.init_others)
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        torch.backends.cudnn.benchmark = True
        
        # Prepare loss function and optimizer
        encoder_optimizer = AdaBound(encoder.parameters(),
                                     lr=args.learning_rate, weight_decay=args.weight_decay)
        decoder_optimizer = AdaBound(decoder.parameters(),
                                     lr=args.learning_rate, weight_decay=args.weight_decay)

        for epoch in range(args.epoch_num):
            loss = fit(args, encoder, decoder, train_set, encoder_optimizer,
                       decoder_optimizer, criterion, is_train=True)
            losses.append(loss)
            train_losses = [np.asarray(losses)]
            if val_set is not None:
                lev_dis, str_accu = fit(args, encoder, decoder, val_set, encoder_optimizer,
                                        decoder_optimizer, criterion, is_train=False)
                lev_dises.append(lev_dis)
                str_accus.append(str_accu)
                val_losses = [np.asarray(lev_dises), np.asarray(str_accus)]
            if epoch % 10 == 0:
                util.save_model(args, args.curr_epoch, encoder.state_dict(), prefix="encoder",
                                keep_latest=20)
                util.save_model(args, args.curr_epoch, decoder.state_dict(), prefix="decoder",
                                keep_latest=20)
            if epoch > 4:
                # Train losses
                vb.plot_loss_distribution(train_losses, ["NLL Loss"], args.loss_log, dt + "_loss", window=5)
                # Val metrics
                if val_set is not None:
                    vb.plot_loss_distribution(val_losses, ["Levenstein", "String-Level"], args.loss_log,
                                              dt + "_val", window=5, bound=[0.0, 100.0])

if __name__ == "__main__":
    main()