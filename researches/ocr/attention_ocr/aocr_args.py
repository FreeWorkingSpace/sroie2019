import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')

    ##############
    #           DATA           #
    ##############
    parser.add_argument(
        "-bpg",
        "--batch_size_per_gpu",
        type=int,
        help="batch size inside each GPU during training",
        default=32
    )
    parser.add_argument(
        "-bpgv",
        "--batch_size_per_gpu_val",
        type=int,
        help="batch size inside each GPU during validation",
        default=32
    )
    parser.add_argument(
        "-lt",
        "--loading_threads",
        type=int,
        help="loading_threads correspond to each GPU during both training and validation, "
             "e.g. You have 4 GPU and set -lt 2, so 8 threads will be used to load data",
        default=2
    )
    
    parser.add_argument(
        "-d",
        "--datasets",
        nargs='+',
        help="a list folder/folders to use as training set",
        default=["SROIE2019_OCR_0"]
    )
    parser.add_argument(
        "-rh",
        "--resize_height",
        type=int,
        help="All image will be resized to this height",
        default=48
    )
    parser.add_argument(
        "-mis",
        "--max_img_size",
        type=int,
        help="Image will be pad/crop if their lenght is shorter/longer than this value",
        default=720
    )
    parser.add_argument(
        "-mss",
        "--max_str_size",
        type=int,
        help="label string will be pad/crop with EOS if its length is shorter/longer than this value",
        default=50
    )
    
    ##############
    #          MODEL         #
    ##############
    parser.add_argument(
        "-eoc",
        "--encoder_out_channel",
        type=int,
        help="Depth of output tensor of encoder CNN (Attn_CNN)",
        default=128
    )
    parser.add_argument(
        "-attnl",
        "--attn_length",
        type=int,
        help="The length of attention, e.g. the encoder output tensor has shape: "
             "[b, c, h, w], where b, c, h, w represent for batch, channel, height, width."
             "If the attn_length is L, then during each step, the model will implicitly divide"
             "the model into L pieces of [b, c, h, w/L] tensor, and assign different weight"
             "to them at each time step",
        default=90
    )
    parser.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        help="the hidden state size inside an RNN",
        default=128
    )
    parser.add_argument(
        "-drl",
        "--decoder_rnn_layers",
        type=int,
        help="RNN layer (GRU) in decoder",
        default=1
    )
    parser.add_argument(
        "-db",
        "--decoder_bottleneck",
        type=int,
        help="Depth of output tensor of encoder CNN (Attn_CNN)",
        default=3840
    )

    ##############
    #        TRAINING        #
    ##############
    parser.add_argument(
        "-dt",
        "--deterministic_train",
        action="store_true",
        help="if this is turned on, then everything will be deterministic"
             "and the training process will be reproducible.",
    )
    parser.add_argument(
        "-en",
        "--epoch_num",
        type=int,
        help="Epoch number of the training",
        default=100
    )
    parser.add_argument(
        "-tfr",
        "--teacher_forcing_ratio",
        type=float,
        help="Initial value of teacher forcing rate (from 0.0 ~ 1.0)",
        default=0.8
    )
    parser.add_argument(
        "-tfrd",
        "--teacher_forcing_ratio_decay",
        type=float,
        help="decay rate of teacher forcing rate in each epoch"
             "e.g. teacher_forcing_ratio *= teacher_forcing_ratio_decay",
        default=0.95
    )

    return parser.parse_args()