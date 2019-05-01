from imgaug import augmenters

def aug_aocr(args, bg_color=255):
    aug_list = []
    aug_list.append(augmenters.Affine(
        scale={"x": (0.8, 1.0), "y": (0.8, 1.0)}, rotate=(-3, 3), cval=bg_color, fit_output=True),
    )
    # aug_list.append(augmenters.AllChannelsCLAHE(tile_grid_size_px =(4, 12), per_channel=False))
    # aug_list.append(augmenters.AddToHueAndSaturation(value=(-10, 10), per_channel=False))
    aug_list.append(augmenters.Resize(
        size={"height": args.resize_height, "width": "keep-aspect-ratio"})
    )
    aug_list.append(augmenters.PadToFixedSize(
        height=args.resize_height, width=args.max_img_size, pad_cval=bg_color)
    )
    aug_list.append(augmenters.CropToFixedSize(
        height=args.resize_height, width=args.max_img_size, position="left-top")
    )
    return aug_list