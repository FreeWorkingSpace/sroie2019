def GeneralPattern(args):
    args.path = "~/Pictures/dataset/ocr"
    args.deterministic_train = True
    args.learning_rate = 1e-4
    args.adam_epsilon = 1e-7
    args.batch_size = 64
    args.random_order_load = False
    args.loading_threads = 4

    args.img_channel = 3
    args.normalize_img = True
    args.normalize_min = 0
    args.normalize_max = 300
    
    args.do_imgaug = False
    # Affine Tranformation
    args.do_affine = True
    args.translation_x = (-0.0, 0.0)
    args.translation_y = (-0.0, 0.0)
    args.scale_x = (1.0, 1.2)
    args.scale_y = (1.0, 1.2)
    # Random Crop
    args.do_crop_to_fix_size = False
    # Random Flip
    args.do_random_flip = False
    # Random Color
    args.do_random_brightness = False
    # Random Noise
    args.do_random_noise = False
    # Size
    args.to_final_size = False
    args.final_size = (32, 32)
    args.standardize_size = False
    args.resize_gcd = 8

    args.img_mean = (0.5, 0.5, 0.5)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args

def Unique_Patterns(args):
    args.training_sources = ["receipts"]
    args.val_sources = ["open_cor"]
    args.epoches_per_phase = 1
    args.load_samples = -1

    args.decoder_rnn_layers = 1
    args.encoder_out_channel = 128
    args.decoder_bottleneck = 3840
    
    args.resize_height = 32
    args.max_img_size = 720
    args.max_str_size = 50
    args.attn_length = 90
    args.hidden_size = 128
    args.teacher_forcing_ratio = 0.05
    args.teacher_forcing_ratio_decay = 0.95
    # Load Image
    args.stretch_img = True
    args.label_dict = {'.': 0, '9': 1, ',': 2, '¥': 3, '6': 4, '4': 5, '3': 6, '〒': 7, '7': 8, '1': 9, '2': 10, '0': 11,
                       '-': 12, '8': 13, '5': 14, '': 15, "/": 16, "(": 17, ")": 18, '|': 15, ' ':15, 'SOS': 19, "EOS": 20}
    args.output_size = len(set([args.label_dict[key] for key in args.label_dict.keys()]))
    return args

def Runtime_Patterns(args):
    args.cover_exist = True
    args.code_name = "_attention"
    args.epoch_num = 5
    args.gpu_id = "1"
    args.finetune = False
    return args

PRESET={
    "general" : GeneralPattern,
    "unique": Unique_Patterns,
    "runtime": Runtime_Patterns,
}