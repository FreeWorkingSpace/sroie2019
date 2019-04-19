from imgaug import augmenters

def aug_aocr(bg_color=255):
    aug_list = []
    aug_list.append(augmenters.Affine(
        scale={"x": (0.8, 1.0), "y": (0.8, 1.0)}, rotate=(-3, 3), cval=bg_color, fit_output=True),
    )
    aug_list.append(augmenters.AllChannelsCLAHE(tile_grid_size_px =(4, 12), per_channel=False))
    aug_list.append(augmenters.AddToHueAndSaturation(value=(-10, 10), per_channel=False))
    aug_list.append(augmenters.Resize(size={"height": 48, "width": "keep-aspect-ratio"}))
    aug_list.append(augmenters.PadToFixedSize(height=32, width=700, pad_cval=bg_color))
    return aug_list