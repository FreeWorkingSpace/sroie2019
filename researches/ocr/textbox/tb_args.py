import argparse

def initialize():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--batch_per_gpu", type=int)
    parser.add_argument("--loading_thread_per_gpu", type=int)


    #  --------------------------Model Architecture-------------------------------


    # -------------------------------Training------------------------------------
    parser.add_argument("--gpu_id", type=str, default="0",
                             help="which gpu you want to use, multi-gpu is not supported here")
    parser.add_argument("--epoch_num", type=int, default=2000,
                             help="Total training epoch")
    parser.add_argument("--deterministic_train", type=bool, default=False,
                             help="Make the training reproducable")
    parser.add_argument("--seed", type=int, default=88,
                             help="If Deterministic train is allowed, this will be the random seed")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_norm", type=bool, default=True)
    parser.add_argument("--finetune", type=bool, default=False)

    # Models does not need to be specified unless you are going to use them
    parser.add_argument("--model1", type=str, help="Model you want to choose")
    parser.add_argument("--model2", type=str, help="Model you want to choose")
    parser.add_argument("--model3", type=str, help="Model you want to choose")
    parser.add_argument("--model4", type=str, help="Model you want to choose")
    parser.add_argument("--model5", type=str, help="Model you want to choose")
    parser.add_argument("--model6", type=str, help="Model you want to choose")

    parser.add_argument("--general_options", type=str, help='general settings for a '
                                                        'model that specifies the settings in the package [options]. '
                                                        'It can be both number or path')
    parser.add_argument("--unique_options", type=str, help="unique settings for a "
                                                        "particular model that are useless to others. "
                                                        "It can be both number or path")

    # ------------------------------MISC-----------------------------------
    parser.add_argument("--loading_threads", type=int, default=2,
                             help="threads used to load data to cpu-memory")
    parser.add_argument("--random_order_load", type=bool, default=False,
                             help="ingore the correspondence of input and output data when load the dataset")
    parser.add_argument("--path", type=str, help="the path of dataset")
    parser.add_argument("--extensions", type=list,
                             default = ["jpeg", "JPG", "jpg", "png", "PNG", "gif", "tiff"])

    args = parser.parse_args()
    return args
