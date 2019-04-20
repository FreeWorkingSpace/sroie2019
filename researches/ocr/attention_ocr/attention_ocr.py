import os, time, sys, math, random
sys.path.append(os.path.expanduser("~/Documents/sroie2019"))
import numpy as np
import cv2, torch
import torch.nn as nn
import omni_torch.visualize.basic as vb
import omni_torch.utils as util
import researches.ocr.attention_ocr as init
import researches.ocr.attention_ocr.aocr_data as data
import researches.ocr.attention_ocr.aocr_presets as preset
import researches.ocr.attention_ocr.aocr_models as att_model
from researches.ocr.attention_ocr.aocr_augment import *


args = util.get_args(preset.PRESET)
"""
Useful Tips:
https://danijar.com/tips-for-training-recurrent-neural-networks/
"""


def fit(args, encoder, decoder, dataset, encode_optimizer, decode_optimizer, criterion, is_train=True):
    if is_train:
        encoder.train()
        decoder.train()
        prefix = ""
        iter = args.epoches_per_phase
    else:
        encoder.eval()
        decoder.eval()
        prefix = "VAL"
        iter = 1
    for epoch in range(iter):
        args.teacher_forcing_ratio *= args.teacher_forcing_ratio_decay
        start_time = time.time()
        for batch_idx, data in enumerate(dataset):
            # Get data ops
            img_batch, label_batch = data[0][0].cuda(), data[0][1].cuda()
            encoder_outputs = encoder(img_batch)

            decoder_input = torch.ones([img_batch.size(0), 1], dtype=torch.long, device=args.device) * 19
            decoder_hidden = decoder.module.initHidden(img_batch.size(0))
            loss = 0
            use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False
            attention = []
            if use_teacher_forcing:
                for di in range(label_batch.size(1)):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    attention.append(decoder_attention)
                    loss += criterion(decoder_output, label_batch[:, di])
                    decoder_input = label_batch[:, di].unsqueeze(-1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(label_batch.size(1)):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    attention.append(decoder_attention)
                    topv, topi = decoder_output.topk(1)
                    loss += criterion(decoder_output, label_batch[:, di])
                    decoder_input = topi.detach()  # detach from history as input
            loss /= label_batch.size(1)
            if is_train:
                encode_optimizer.zero_grad()
                decode_optimizer.zero_grad()
                loss.backward()
                encode_optimizer.step()
                decode_optimizer.step()
            # Visualize
        print(prefix, loss)

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
    aug = aug_aocr()
    datasets = data.fetch_data(args, args.training_sources, batch_size=args.batch_size,
                              batch_size_val=1, text_seperator=":", k_fold=1, split_val=0.1,
                              pre_process=None, aug=aug)

    for idx, (train_set, val_set) in enumerate(datasets):
        print("\n =============== Cross Validation: %s/%s ================ " %
              (idx + 1, len(datasets)))
        # Prepare Network
        encoder = att_model.Attn_CNN(args.img_channel, args.encoder_out_channel)
        decoder = att_model.AttnDecoder(args)
        if args.finetune:
            encoder, decoder = util.load_latest_model(args, [encoder, decoder],
                                                      prefix=["encoder", "decoder"])
        else:
            # Missing Initialization functions for RNN in CUDA, splitting initialization on CPU and GPU respectively
            encoder.apply(init.init_rnn).to(args.device).apply(init.init_others)
            decoder.apply(init.init_rnn).to(args.device).apply(init.init_others)
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        torch.backends.cudnn.benchmark = True
        
        # Prepare loss function and optimizer
        criterion = nn.NLLLoss().to(args.device)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate,
                                             weight_decay=args.weight_decay, eps=args.adam_epsilon)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate,
                                             weight_decay=args.weight_decay, eps=args.adam_epsilon)

        for epoch in range(args.epoch_num):
            fit(args, encoder, decoder, train_set, encoder_optimizer, decoder_optimizer, criterion, is_train=True)
            if val_set is not None:
                fit(args, encoder, decoder, val_set, encoder_optimizer, decoder_optimizer, criterion, is_train=False)
            
            if epoch != 0 and epoch % 20 == 0:
                util.save_model(args, args.curr_epoch, encoder.state_dict(), prefix="encoder",
                                keep_latest=20)

if __name__ == "__main__":
    main()