import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import omni_torch.networks.blocks as omth_blocks


class Attn_CNN(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_p=0.3):
        super().__init__()
        self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1 = omth_blocks.Conv_Block(in_channel, 32, 3, padding=1, stride=1,
                                             batch_norm=None)
        self.block2 = omth_blocks.Conv_Block(32, 64, 3, padding=1, stride=1,
                                             batch_norm=None)
        self.block3 = omth_blocks.Conv_Block(64, [96, 96], [3, 1], padding=[1, 0], stride=[1, 1],
                                             batch_norm=[nn.BatchNorm2d, None])
        self.block4 = omth_blocks.Conv_Block(96, [128, 128, out_channel], [3, 3, 1], padding=[1, 1, 0],
                                             stride=[1, 1, 1], batch_norm=[nn.BatchNorm2d, None, nn.BatchNorm2d],
                                             dropout=[0, 0, dropout_p])
    
    def forward(self, x):
        x = self.maxout(self.block1(x))
        x = self.maxout(self.block2(x))
        x = self.maxout(self.block3(x))
        x = self.block4(x)
        return x


class Decoder_Inner_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channel = args.encoder_out_channel
        bottelneck = args.decoder_bottleneck
        hidden_size = args.hidden_size
        self.cnn1 = omth_blocks.Conv_Block(in_channel, 256, 2, padding=0, stride=[(1, 2)],
                                           batch_norm=nn.BatchNorm2d)
        self.cnn2 = omth_blocks.Conv_Block(256, 256, 3, padding=0, stride=3,
                                           batch_norm=nn.BatchNorm2d)
        self.fc_layer = omth_blocks.FC_Layer(bottelneck, [1024, 512], batch_norm=nn.BatchNorm1d,
                                             activation=nn.ReLU())
        self.fc_cat = omth_blocks.FC_Layer(512 + hidden_size, [hidden_size],
                                           batch_norm=nn.BatchNorm1d, activation=nn.ReLU())
    
    def forward(self, x, emb):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.fc_layer(x)
        x = torch.cat([x, emb], dim=1)
        return self.fc_cat(x).unsqueeze(0)


class AttnDecoder(nn.Module):
    def __init__(self, args, dropout_p=0.1):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.attn_length = args.attn_length
        self.rnn_layers = args.decoder_rnn_layers
        self.dropout_p = dropout_p
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.attn_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.inner_cnn = Decoder_Inner_CNN(args)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.rnn_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        #self.hidden = self.initHidden(args.batch_size_per_gpu)
    
    def _forward(self, encoder_outputs, label_batch, verbose=False):
        # Init
        decoder_outputs, decoder_attns = [], []
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        input = torch.zeros([encoder_outputs.size(0), 1]).long().cuda()
        hidden =  torch.zeros(self.rnn_layers, encoder_outputs.size(0), self.hidden_size).cuda()
        if verbose:
            print("Create decoder input with shape: %s." % str(input.shape))
            print("Create decoder hidden with shape: %s." % str(hidden.shape))
        
        # Forward
        self.gru.flatten_parameters()
        for di in range(label_batch.size(1)):
            embedded = self.dropout(self.embedding(input)).permute(1, 0, 2)
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded.squeeze(), hidden.squeeze()), 1)), dim=1)
            encoder_outputs = torch.mul(encoder_outputs, attn_weights.unsqueeze(1).unsqueeze(1) \
                                        .repeat(1, encoder_outputs.size(1), encoder_outputs.size(2), 1))
            output = self.inner_cnn(encoder_outputs, embedded.squeeze())
            output, hidden = self.gru(output, hidden)
            output = F.log_softmax(self.out(output[0]), dim=1)
            if verbose:
                print("Step: %d, decoder output shape: %s" % (di, str(output.shape)))
                print("Step: %d, decoder attention shape: %s" % (di, str(attn_weights.shape)))
            decoder_outputs.append(output)
            decoder_attns.append(attn_weights)
            if use_teacher_forcing:
                input = label_batch[:, di].unsqueeze(-1)  # Teacher forcing
            else:
                topv, topi = output.topk(1)
                input = topi.detach()  # detach from history as input
        return decoder_outputs, decoder_attns
    
    def forward(self, input, hidden, encoder_outputs, verbose=False):
        #self.gru.flatten_parameters()
        embedded = self.dropout(self.embedding(input)).permute(1, 0, 2)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded.squeeze(), hidden.squeeze()), 1)), dim=1)
        encoder_outputs = torch.mul(encoder_outputs, attn_weights.unsqueeze(1).unsqueeze(1) \
                                    .repeat(1, encoder_outputs.size(1), encoder_outputs.size(2), 1))
        output = self.inner_cnn(encoder_outputs, embedded.squeeze())
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        if verbose:
            print("Decoder input with shape: %s." % str(input.shape))
            print("Decoder hidden with shape: %s." % str(hidden.shape))
            print("Decoder output shape: %s" % str(output.shape))
            print("Decoder attention shape: %s" % str(attn_weights.shape))
        return output, hidden, attn_weights
    
    def initHidden(self, batch_size):
        if batch_size % torch.cuda.device_count() != 0:
            raise RuntimeError("Batch size (%d) should be dividable by GPUs (%d) in use."
                               % (batch_size, torch.cuda.device_count()))
        #batch_size = int(batch_size / torch.cuda.device_count())
        # each dimension means layer_number, batch_size, hidden_size
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_size).cuda()
    

class Attn_Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Attn_CNN(args.img_channel, args.encoder_out_channel)
        self.decoder = AttnDecoder(args)
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        
    def forward(self, x, y, criterion):
        loss = 0
        #outs = []
        x = self.encoder(x)
        input = torch.zeros([x.size(0), 1]).long().cuda()
        hidden = torch.zeros(self.args.decoder_rnn_layers, x.size(0), self.args.hidden_size).cuda()
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        for i in range(y.size(1)):
            out, hidden, attn = self.decoder(input, hidden, x)
            #outs.append(out)
            loss += criterion(out, y[:, i])
            if use_teacher_forcing:
                input = y[:, i].unsqueeze(-1)
            else:
                topv, topi = out.topk(1)
                input = topi.detach()  # detach from history as input
            del out
        return loss


if __name__ == "__main__":
    import time, os
    import omni_torch.utils as util
    import researches.ocr.attention_ocr.aocr_presets as preset
    
    args = util.get_args(preset.PRESET)
    
    img_batch, label_batch = torch.randn(128, 3, 48, 720).cuda(), torch.ones(128, 50).long().cuda()
    """
    attn_seq = Attn_Seq2Seq(args)
    attn_seq = torch.nn.DataParallel(attn_seq)
    attn_seq = attn_seq.cuda()
    criterion = nn.NLLLoss()
    pred = attn_seq(img_batch, label_batch, criterion)
    print("Output: %s"%str(pred))
    """
    
    
    encoder = Attn_CNN(args.img_channel, args.encoder_out_channel)
    decoder = AttnDecoder(args)
    encoder = torch.nn.DataParallel(encoder, device_ids=[0, 1]).cuda()
    decoder = torch.nn.DataParallel(decoder, device_ids=[0, 1]).cuda()
    criterion = nn.NLLLoss()
    
    encoder_outputs = encoder(img_batch)
    print("Image was encoded as shape: %s" % (str(encoder_outputs.shape)))
    
    # Decoder input is default the index of SOS token
    input = torch.zeros([encoder_outputs.size(0), 1]).long().cuda() + args.label_dict["SOS"]
    hidden = decoder.module.initHidden(encoder_outputs.size(0))
    print("Create decoder input with shape: %s." % str(input.shape))
    print("Create decoder hidden with shape: %s." % str(hidden.shape))
    start = time.time()
    for i in range(label_batch.size(1)):
        out, decoder_attention = decoder(input, encoder_outputs)
        print("Decoder output shape: %s" % str(out.shape))
        print("Decoder attention shape: %s" %str(decoder_attention.shape))
        #os.system("nvidia-smi")
    print("cost %.2f seconds"%(time.time() - start))
    #print(criterion(out, label_batch[:, 0]))
    
    #decoder_outputs, decoder_attentions = decoder(encoder_outputs, label_batch, verbose=True)
    #print("Decoder output shape: %s" % str(decoder_outputs[0].shape))
    #print("Decoder attention shape: %s" % str(decoder_attentions[0].shape))
    #"""