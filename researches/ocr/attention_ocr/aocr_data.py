import os, random
import torch, cv2
import numpy as np
from torch.utils.data import *
import omni_torch.utils as util
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset
import omni_torch.data.data_loader as omth_loader

def get_path_and_label(args, length, paths, foldername):
    with open(paths, "r", encoding="utf-8") as txtfile:
        output_path = []
        output_label = []
        for _, line in enumerate(txtfile):
            #if _ in [7]:
                #print("")
            splitted_line = line.strip().split(args.text_seperator)
            output, label = splitted_line[0], splitted_line[1]
            output_path.append(os.path.join(args.path, foldername, output))
            #output_path.append(os.path.join(args.path, foldername, line[: line.find(args.text_seperator)]))
            #label = line[line.find(args.text_seperator) + 1:-1]
            try:
                output_label.append([args.label_dict[_] for _ in label])
            except KeyError:
                print("Key Error Occured at line %s"%(_))
    return [list(zip(output_path, output_label))]

def read_img_and_label(args, items, seed, size, pre_process, rand_aug, bbox_loader):
    path, label = items[0], items[1]
    img_tensor = omth_loader.read_image(args, path, seed, size, pre_process=pre_process,
                                        rand_aug=rand_aug, bbox_loader=bbox_loader)
    if len(label) > args.max_str_size - 2:
        # Characters in label exceed the maxium predictable string length(args.max_str_size)
        print("label in %s exceed max_str_size %s by %s"
              %(path, args.max_str_size - 2, len(label) - args.args.max_str_size + 2))
        label = [19] + label[: args.max_str_size - 2] + [20]
    else:
        # Pad the label using EOS token
        label = [19] + label + [20] * (args.max_str_size - 2 - len(label)) + [20]
    # [19] and [20] represent SOS token and EOS token respectively
    return img_tensor, omth_loader.just_return_it(args, label, seed, size).long()

def resize_height_and_pad_width(image, args, path, seed, size):
    """
    Resize the image to size: (args.resize_height, args.max_img_size) by padding or
    """
    if image is None:
        print(path)
    width, height = image.shape[1], image.shape[0]
    width = round(width / height * args.resize_height)
    if width < args.max_img_size:
        image = cv2.resize(image, (width, args.resize_height))
        # If the width of image is larger than max_img_size
        pad = np.ones((args.resize_height, args.max_img_size - width, args.img_channel),
                      dtype="uint8") * 128
        image = np.concatenate((image, pad), axis=1)
    elif args.max_img_size < width < args.max_img_size * 1.2:
        image = cv2.resize(image, (args.max_img_size, args.resize_height))
    else:
        print("%s has a width of %s after resize height exceed max length for %s percent"
              %(path, width, int(100 * width / args.max_img_size)))
        image = cv2.resize(image, (args.max_img_size, args.resize_height))
    return image

def fetch_data(args, sources, batch_size, batch_size_val, text_seperator=None, shuffle=False,
               txt_file=None, split_val=0.0, k_fold=1, pre_process=None, aug=None):
    if text_seperator is not None:
        args.text_seperator = text_seperator
    if txt_file:
        read_source = txt_file
    else:
        read_source = [_ + "/label.txt" for _ in sources]
    args.loading_threads = round(args.loading_threads * torch.cuda.device_count())
    if batch_size_val is None:
        batch_size_val = batch_size
    dataset = []
    for i, source in enumerate(sources):
        subset = Arbitrary_Dataset(args, sources=[read_source[i]], step_1=[get_path_and_label],
                                   step_2=[read_img_and_label], auxiliary_info=[source],
                                   pre_process=[pre_process], augmentation=[aug])
        subset.prepare()
        dataset.append(subset)

    """
    workers = args.loading_threads
    samples = sum([len(d) for d in dataset]) - 1
    kwargs = {'num_workers': workers, 'pin_memory': pin_memory} if torch.cuda.is_available() else {}
    if split_val > 0.0:
        train_index = random.sample(range(samples), samples - int(samples * split_val))
        val_index = random.sample(range(samples), int(samples * split_val))
        train_sampler = sampler.SubsetRandomSampler(train_index)
        validation_sampler = sampler.SubsetRandomSampler(val_index)
        train_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                                                shuffle=shuffle, sampler=train_sampler, **kwargs)
        val_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                                              shuffle=shuffle, sampler=validation_sampler, **kwargs)
        return train_set, val_set
    else:
        train_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                                                shuffle=shuffle, **kwargs)
        return train_set
    """
        
    if k_fold > 1:
        return util.k_fold_cross_validation(args, dataset, batch_size, batch_size_val, k_fold)
    else:
        if split_val > 0:
            return util.split_train_val_dataset(args, dataset, batch_size, batch_size_val, split_val)
        else:
            kwargs = {'num_workers': args.loading_threads, 'pin_memory': True}
            train_set = DataLoader(ConcatDataset(dataset), batch_size=batch_size,
                                   shuffle=shuffle, **kwargs)
            return [(train_set, None)]
        