from imgaug import augmenters

def aug_sroie():
    aug_list = []
    stage_0, stage_1, stage_2, stage_3 = 1536, 2048, 768, 512
    # Pad the height to stage_0
    aug_list.append(augmenters.PadToFixedSize(width=1, height=stage_0, pad_cval=255))
    # Resize its height to stage_1, note that stage_0 is smaller than stage_1
    # so that the font size could be increased for most cases.
    aug_list.append(augmenters.Resize(size={"height": stage_1, "width": "keep-aspect-ratio"}))
    # Crop a stage_2 x stage_2 area
    aug_list.append(augmenters.CropToFixedSize(width=stage_2, height=stage_2))
    # In case the width is not enough, pad it to stage_2 x stage_2
    aug_list.append(augmenters.PadToFixedSize(width=stage_2, height=stage_2, pad_cval=255))
    # Resize to stage_3 x stage_3
    aug_list.append(augmenters.Resize(size={"height": stage_3, "width": stage_3}))
    # Perform Flip
    aug_list.append(augmenters.Fliplr(0.33, name="horizontal_flip"))
    aug_list.append(augmenters.Flipud(0.33, name="vertical_flip"))
    return aug_list

def sroie_refine():
    aug_list = []
    stage_0, stage_1, stage_2, stage_3 = 1536, 2048, 768, 512
    # Pad the height to stage_0
    aug_list.append(augmenters.PadToFixedSize(width=1, height=stage_0, pad_cval=255))
    # Resize its height to stage_1, note that stage_0 is smaller than stage_1
    # so that the font size could be increased for most cases.
    aug_list.append(augmenters.Resize(size={"height": stage_1, "width": "keep-aspect-ratio"}))
    # Crop a stage_2 x stage_2 area
    aug_list.append(augmenters.CropToFixedSize(width=stage_2, height=stage_2))
    # In case the width is not enough, pad it to stage_2 x stage_2
    aug_list.append(augmenters.PadToFixedSize(width=stage_2, height=stage_2, pad_cval=255))
    # Resize to stage_3 x stage_3
    aug_list.append(augmenters.Resize(size={"height": stage_3, "width": stage_3}))
    # Perform Flip
    aug_list.append(augmenters.Fliplr(0.33, name="horizontal_flip"))
    aug_list.append(augmenters.Flipud(0.33, name="vertical_flip"))
    return aug_list


def aug_old():
    aug_list = []
    aug_list.append(augmenters.Crop(percent=0.25, sample_independently=True, keep_size=False))
    aug_list.append(augmenters.Resize(size={"height": 1024, "width": "keep-aspect-ratio"}))
    aug_list.append(augmenters.PadToFixedSize(width=768, height=768, pad_cval=255))
    #aug_list.append(augmenters.Fliplr(args.h_flip_prob, name="horizontal_flip"))
    #aug_list.append(augmenters.Flipud(args.v_flip_prob, name="vertical_flip"))
    aug_list.append(augmenters.Resize(size={"height": 768, "width": 768}))
    return aug_list