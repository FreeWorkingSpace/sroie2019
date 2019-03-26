import torch

def GeneralPattern(args):
    args.path = "~/Pictures/dataset/ocr"
    args.deterministic_train = False
    args.seed = 1
    args.learning_rate = 1e-4
    args.batch_size = 24
    args.accu_batch_size = 48
    args.random_order_load = False
    args.loading_threads = 6
    args.img_channel = 3

    # We are defining the augmentation separately
    args.do_imgaug = False

    # Size
    args.to_final_size = False
    args.final_size = [{"height": 512, "width": 512}]
    args.standardize_size = False
    args.resize_gcd = 8
    
    args.img_mean = (0.5, 0.5, 0.5)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args


def Unique_Patterns(args):
    #args.train_sources = ["SROIE2019", "tempholding"]
    args.train_sources = ["SROIE2019"]
    #args.train_sources = ["tempholding"]
    #args.train_aux = [{"txt": "txt", "img": "jpg"}, {"txt": "xml", "img": "png"}]
    args.train_aux = [{"txt": "txt", "img": "jpg"}]
    #args.train_aux = [{"txt": "xml", "img": "png"}]
    args.min_bbox_threshold = 0.01
    args.val_sources = None
    args.epoches_per_phase = 1
    return args


def Runtime_Patterns(args):
    args.cover_exist = True
    args.code_name = "_text_detection"
    args.epoch_num = 2000
    args.gpu_id = "2"
    args.finetune = False
    return args


PRESET = {
    "general": GeneralPattern,
    "unique": Unique_Patterns,
    "runtime": Runtime_Patterns,
}