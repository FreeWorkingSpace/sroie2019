import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
import omni_torch.networks.blocks as omth_blocks
import researches.ocr.attention_ocr as init


class Attn_CNN(nn.Module):
    def __init__(self, backbone_require_grad=False):
        super().__init__()
        # Use pre-trained model as backbone
        """
        backbone = vgg16_bn(pretrained=True)
        for param in backbone.parameters():
            param.requires_grad = backbone_require_grad
        net = list(backbone.children())[0][:24]
        maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        """
        #net = [ceil_maxout if type(n) is nn.MaxPool2d else n for n in net]
        # 3 x 48 x 960
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, (2, 1), 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2))  # 512x1x25
        """
        backbone = vgg11_bn(pretrained=True)
        net = list(backbone.children())[0][:21]
        for param in backbone.features.parameters():
            param.requires_grad = backbone_require_grad
            """
        #self.cnn = nn.Sequential(*net)
        self.final_conv = omth_blocks.InceptionBlock(256, filters=[[256, 128], [256, 128]],
                                                     kernel_sizes=[[[3, 1], 1], [3, 1]], stride=[[1, 1], [1, 1]],
                                                     padding=[[[0, 0], 0], [[0, 1], 0]])

    def forward(self, x):
        x = self.cnn(x)
        x = self.final_conv(x)
        return x


class BidirectionalLSTM(nn.Module):
    """
    Result will be BAD after BLSTM was added after CNN output
    """
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=False)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class AttnDecoder(nn.Module):
    def __init__(self, args, encoder_hidden=256, dropout_p=0.1,
                 hidden_from_input="zero"):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.attn_length = args.attn_length
        self.rnn_layers = args.decoder_rnn_layers
        self.dropout_p = dropout_p
        self.sos_token = args.label_dict["SOS"]
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.hidden_from_x = hidden_from_input

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.attn_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.rnn_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def _forward(self, input, hidden, x):
        """
        Unable to parallelize,
        """
        embedded = self.dropout(self.embedding(input))
        attn_weights = self.attn(torch.cat((embedded.permute(1, 0, 2).squeeze(0), hidden[0]), 1))
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        attn_applied = torch.matmul(attn_weights, x.permute(1, 0, 2))
        output = self.attn_combine(torch.cat((attn_applied, embedded), dim=2).squeeze(1))
        output = F.relu(output)
        output, hidden = self.gru(output.unsqueeze(0), hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output

    def forward(self, x, y, is_train=True, verbose=False):
        # Init
        decoder_outputs, decoder_attns = [], []
        if is_train:
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        else:
            use_teacher_forcing = False
        input = torch.zeros([x.size(0), 1]).long().cuda() + self.sos_token
        hidden = self.initHidden(input.size(0), x).cuda()
        #cell_state = torch.zeros(hidden.shape).cuda()
        if verbose:
            print("Create decoder input with shape: %s." % str(input.shape))
            print("Create decoder hidden with shape: %s." % str(hidden.shape))

        # Forward
        assert x.size(2) == 1
        self.gru.flatten_parameters()
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        for di in range(y.size(1)):
            embedded = self.dropout(self.embedding(input))
            attn_weights = self.attn(torch.cat((embedded.permute(1, 0, 2).squeeze(0), hidden[0]), 1))
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
            attn_applied = torch.matmul(attn_weights, x.permute(1, 0, 2))
            output = self.attn_combine(torch.cat((attn_applied, embedded), dim=2).squeeze(1))
            output = F.relu(output)
            output, hidden = self.gru(output.unsqueeze(0), hidden)
            output = F.log_softmax(self.out(output[0]), dim=1)
            if verbose:
                print("Step: %d, output=>%s, attn=>%s" %
                      (di, str(output.shape), str(attn_weights.shape)))
            decoder_outputs.append(output)
            decoder_attns.append(attn_weights)
            if use_teacher_forcing:
                input = y[:, di].unsqueeze(-1)  # Teacher forcing
            else:
                topv, topi = output.topk(1)
                input = topi.detach()  # detach from history as input
        decoder_outputs = torch.stack(decoder_outputs, -1)
        decoder_attns = torch.cat(decoder_attns, dim=1)
        return decoder_outputs, decoder_attns

    def initHidden(self, batch_size, x):
        if self.hidden_from_x == "sum":
            pass
        elif self.hidden_from_x == "weight_sum":
            pass
        elif self.hidden_from_x == "linear":
            pass
        elif self.hidden_from_x == "zero":
            return torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import time, os
    import omni_torch.utils as util
    import researches.ocr.attention_ocr.aocr_presets as preset

    args = util.get_args(preset.PRESET)

    img_batch, label_batch = torch.randn(128, 3, 48, 960).cuda(), torch.ones(128, 50).long().cuda()
    """
    attn_seq = Attn_Seq2Seq(args)
    attn_seq = torch.nn.DataParallel(attn_seq)
    attn_seq = attn_seq.cuda()
    criterion = nn.NLLLoss()
    pred = attn_seq(img_batch, label_batch, criterion)
    print("Output: %s"%str(pred))
    """

    encoder = Attn_CNN()
    decoder = AttnDecoder(args)
    encoder = torch.nn.DataParallel(encoder).cuda()
    decoder = torch.nn.DataParallel(decoder).cuda()
    # decoder.module.hidden = torch.cuda.comm.broadcast(
    #  decoder.module.initHidden(64), list(range(torch.cuda.device_count()))
    # )
    criterion = nn.NLLLoss()

    encoder_outputs = encoder(img_batch)
    print("Image was encoded as shape: %s" % (str(encoder_outputs.shape)))

    # Decoder input is default the index of SOS token
    #input = torch.zeros([encoder_outputs.size(0), 1]).long().cuda() + args.label_dict["SOS"]
    #hidden = decoder.module.initHidden(encoder_outputs.size(0))
    #print("Create decoder input with shape: %s." % str(input.shape))
    #print("Create decoder hidden with shape: %s." % str(hidden.shape))

    start = time.time()
    out, decoder_attention = decoder(encoder_outputs, label_batch)
    # for i in range(label_batch.size(1)):
    #  out, decoder_attention = decoder(input, encoder_outputs)
    print("Decoder output shape: %s" % str(out.shape))
    print("Decoder attention shape: %s" % str(decoder_attention.shape))
    # os.system("nvidia-smi")
    print("cost %.2f seconds" % (time.time() - start))
    loss = [criterion(out[:, :, i], label_batch[:, i]) for i in range(out.size(2))]
    print(sum(loss) / len(loss))
    # print(criterion(out, label_batch[:, 0]))

    # decoder_outputs, decoder_attentions = decoder(encoder_outputs, label_batch, verbose=True)
    # print("Decoder output shape: %s" % str(decoder_outputs[0].shape))
    # print("Decoder attention shape: %s" % str(decoder_attentions[0].shape))
    # """