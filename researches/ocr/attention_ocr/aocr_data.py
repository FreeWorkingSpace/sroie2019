import  random
import torch, cv2
import numpy as np
from torch.utils.data import *
from omni_torch.data.arbitrary_dataset import Arbitrary_Dataset
import omni_torch.data.data_loader as omth_loader
import researches.ocr.misc as ocr_misc

def read_img_and_label(args, path, seed, size, ops=None):
    path, label = path[0], path[1]
    img_tensor = omth_loader.read_image(args, path, seed, size, ops)
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
        pad = np.ones((args.resize_height, args.max_img_size - width, args.img_channel), dtype="uint8") * 128
        image = np.concatenate((image, pad), axis=1)
    elif args.max_img_size < width < args.max_img_size * 1.2:
        image = cv2.resize(image, (args.max_img_size, args.resize_height))
    else:
        print("%s has a width of %s after resize height exceed max length for %s percent"
              %(path, width, int(100 * width / args.max_img_size)))
        image = cv2.resize(image, (args.max_img_size, args.resize_height))
    return image

def fetch_data(args, sources, batch_size, text_seperator=None, shuffle=False, pin_memory=False,
               txt_file=None, split_val=0.0):
    if text_seperator is not None:
        args.text_seperator = text_seperator
    if txt_file:
        read_source = txt_file
    else:
        read_source = [_ + "/label.txt" for _ in sources]
    dataset = []
    for i, source in enumerate(sources):
        subset = Arbitrary_Dataset(args, sources=[read_source[i]], step_1=[ocr_misc.get_path_and_label],
                                   step_2=[read_img_and_label], auxiliary_info=[source],
                                   pre_process=[resize_height_and_pad_width])
        subset.prepare()
        dataset.append(subset)
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