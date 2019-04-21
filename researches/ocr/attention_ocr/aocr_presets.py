def GeneralPattern(args):
    args.path = "~/Pictures/dataset/ocr"
    args.code_name = "_attention"
    args.deterministic_train = False
    args.learning_rate = 1e-4
    args.batch_size_per_gpu = 32
    args.batch_size_per_gpu_val = 32
    args.random_order_load = False
    args.loading_threads = 2
    args.cover_exist = True
    args.epoch_num = 100
    args.finetune = False

    args.img_channel = 3
    args.normalize_img = True
    args.normalize_min = 0
    args.normalize_max = 300

    args.img_mean = (0.5, 0.5, 0.5)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args

def Unique_Patterns(args):
    args.datasets = ["SROIE2019_OCR_0"]
    args.text_seperator = ":"
    args.epoches_per_phase = 1

    args.decoder_rnn_layers = 1
    args.encoder_out_channel = 128
    args.decoder_bottleneck = 3840
    
    args.resize_height = 48
    args.max_img_size = 720
    args.max_str_size = 45
    args.attn_length = 45
    args.hidden_size = 128
    args.teacher_forcing_ratio = 0.8
    args.teacher_forcing_ratio_decay = 0.95
    args.label_dict = {'SOS': 0, ' ': 1, '!': 2, '"': 3, '#': 4, '$': 5, '%': 6, '&': 7,
                       "'": 8, '(': 9, ')': 10, '*': 11, '+': 12, ',': 13, '-': 14, '.': 15, '/':16,
                       '0': 17, '1': 18, '2': 19, '3': 20, '4': 21, '5': 22, '6': 23,  '7': 24,
                       '8': 25, '9': 26, ':': 27, ';': 28, '<': 29, '=': 30, '>': 31, '?': 32,
                       '@': 33, 'A': 34, 'B': 35, 'C': 36, 'D': 37, 'E': 38, 'F': 39, 'G': 40,
                       'H': 41, 'I': 42, 'J': 43, 'K': 44, 'L': 45, 'M': 46, 'N': 47, 'O': 48,
                       'P': 49, 'Q': 50, 'R': 51, 'S': 52, 'T': 53, 'U': 54, 'V': 55, 'W': 56,
                       'X': 57, 'Y': 58, 'Z': 59, '[': 60, '\\': 61, ']': 62, '^': 63, '_': 64,
                       '`': 65, 'a': 66, 'b': 67, 'c': 68, 'd': 69, 'e': 70, 'f': 71, 'g': 72,
                       'h': 73, 'i': 74, 'j': 75, 'k': 76, 'l': 77, 'm': 78, 'n': 79, 'o': 80,
                       'p': 81, 'q': 82, 'r': 83, 's': 84, 't': 85, 'u': 86, 'v': 87, 'w': 88,
                       'x': 89, 'y': 90, 'z': 91, '{': 92, '|': 93, '}': 94, '~': 95, '·': 96,
                       'EOS': 97}
    args.output_size = len(set([args.label_dict[key] for key in args.label_dict.keys()]))
    return args

def Runtime_Patterns(args):
    return args

PRESET={
    "general" : GeneralPattern,
    "unique": Unique_Patterns,
    "runtime": Runtime_Patterns,
}