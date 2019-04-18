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


class MNIST_conv(nn.Module):
    def __init__(self, in_channel, BN=nn.BatchNorm2d):
        super().__init__()
        self.maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.block1 = omth_blocks.InceptionBlock(in_channel, filters=[[16], [16]], kernel_sizes=[[5], [3]],
                                                 padding=[[2], [1]], stride=[[1], [1]],
                                                 name="incep_block1", batch_norm=BN)
        self.block1_1 = omth_blocks.conv_block(32, [32], [1], [1], padding=[0], groups=[1],
                                               name="attention_block_1", batch_norm=BN)
        self.block2 = omth_blocks.InceptionBlock(32, filters=[[32], [32]], kernel_sizes=[[5], [3]],
                                                 padding=[[2], [1]], stride=[[1], [1]],
                                                 name="incep_block2", batch_norm=BN)
        self.block2_1 = omth_blocks.conv_block(64, [64], [1], [1], padding=[0], groups=[1],
                                               name="attention_block_2", batch_norm=BN)
        self.block3 = omth_blocks.InceptionBlock(64, filters=[[64, 64], [32, 32], [64, 64], [32, 32]],
                                                 kernel_sizes=[[(5, 3), 1], [5, 1], [3, 1], [(3, 5), 1]],
                                                 padding=[[(2, 1), 0], [2, 0], [1, 0], [(1, 2), 0]],
                                                 stride=[[1, 1], [1, 1], [1, 1], [1, 1]], name="incep_block3")
        self.block4 = omth_blocks.conv_block(192, filters=[192], kernel_sizes=[1],
                                             stride=[1], padding=[0], groups=[1],
                                             name="concat_block", batch_norm=BN)

    def forward(self, x):
        x = self.block1_1(self.maxout(self.block1(x)))
        x = self.block2_1(self.maxout(self.block2(x)))
        x = self.block4(self.maxout(self.block3(x)))
        x = self.maxout(x)
        return x

class AttnDecoder(nn.Module):
    def __init__(self, args, dropout_p=0.1):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.attn_length = args.attn_length
        self.rnn_layers = args.decoder_rnn_layers
        self.device = args.device
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.attn_length)
        self.dropout = nn.Dropout(self.dropout_p)
        self.inner_cnn = Decoder_Inner_CNN(args)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.rnn_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input)).permute(1, 0, 2)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded.squeeze(), hidden.squeeze()), 1)), dim=1)
        encoder_outputs = torch.mul(encoder_outputs, attn_weights.unsqueeze(1).unsqueeze(1)\
            .repeat(1, encoder_outputs.size(1), encoder_outputs.size(2), 1))

        output = self.inner_cnn(encoder_outputs, embedded.squeeze())
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        # each dimension means layer_number, batch_size, hidden_size
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_size, device=self.device)