import os, time, sys, math, random, glob, datetime
sys.path.append(os.path.expanduser("~/Documents/sroie2019"))
import cv2, torch
import numpy as np
import imgaug
from imgaug import augmenters
import omni_torch.utils as util
import researches.ocr.textbox.tb_data as data
import researches.ocr.textbox.tb_preset as preset
import researches.ocr.textbox.tb_model as model
from researches.ocr.textbox.tb_utils import *
from researches.ocr.textbox.tb_preprocess import *
from researches.ocr.textbox.tb_augment import *
from researches.ocr.textbox.tb_vis import visualize_bbox, print_box
import omni_torch.visualize.basic as vb

TMPJPG = os.path.expanduser("~/Pictures/tmp.jpg")
cfg = model.cfg
args = util.get_args(preset.PRESET)
if not torch.cuda.is_available():
    raise RuntimeError("Need cuda devices")
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")


def test_rotation():
    def return_aug(transform_det, height_ori, width_ori, height, width):
        aug_list = []
        aug_list.append(
            augmenters.Resize(size={"height": height, "width": width})
        )
        if "rotation" in transform_det:
            aug_list.append(
                augmenters.Affine(rotate=-transform_det["rotation"], cval=args.aug_bg_color, fit_output=True),
            )
        aug = augmenters.Sequential(aug_list, random_order=False)
        return aug
    import imgaug
    from imgaug import augmenters

    # Load Model
    net = model.SSD(cfg, connect_loc_to_conf=True, fix_size=False,
                    incep_conf=True, incep_loc=True)
    net = net.cuda()
    net_dict = net.state_dict()
    weight_dict = util.load_latest_model(args, net, prefix="cv_1", return_state_dict=True)
    for key in weight_dict.keys():
        net_dict[key[7:]] = weight_dict[key]
    net.load_state_dict(net_dict)
    net.eval()

    img_list = glob.glob(os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019_test/*.jpg"))
    for i, img_file in enumerate(sorted(img_list)):
        # Get img and bbox infomation from local file
        img = cv2.imread(img_file)
        height_ori, width_ori = img.shape[0], img.shape[1]

        # detect rotation for returning the image back
        img, transform_det = estimate_angle(img, args, None, None, None)
        if transform_det["rotation"] != 0:
            rot_aug = augmenters.Affine(rotate=transform_det["rotation"],
                                         cval=args.aug_bg_color)
        else:
            rot_aug = None

        # Augment img and bbox, even if rotation exists, we only rotate img not bbox
        if rot_aug:
            rot_aug = augmenters.Sequential(
                augmenters.Affine(rotate=transform_det["rotation"], cval=args.aug_bg_color),
                random_order=False)
            #rot_bbox = rot_aug.augment_bounding_boxes([bbox])[0]
            image = rot_aug.augment_image(img)
        else:
            image = img
        height, width = image.shape[0], image.shape[1]
        if height != height_ori or width != width_ori:
            print("wrong rotation method")
        square = 1536
        resize_aug =augmenters.Sequential([
            augmenters.Resize(size={"height": square, "width": "keep-aspect-ratio"}),
            augmenters.PadToFixedSize(width=square, height=square, pad_cval=255),
        ])
        resize_aug = resize_aug.to_deterministic()
        image = resize_aug.augment_image(image)
        # Get the final size of resized input image
        height_final, width_final = image.shape[0], image.shape[1]
        # Generate prior boxes according to the input image size
        #cfg["feature_map_sizes"] = [[height_final/8, width_final/8], [height_final/16, width_final/16],
        #                             [height_final/32, width_final/32]]
        #net.prior = net.create_prior(input_size=(height_final, width_final)).cuda()
        # Collect bboxes inside the image
        #coord = extract_boxes(bbox, height_final, width_final, box_label)
        #rot_coord = extract_boxes(rot_bbox, height_final, width_final, box_label)

        # Prepare image tensor and test
        image = torch.Tensor(util.normalize_image(args, image)).unsqueeze(0)
        image = image.permute(0, 3, 1, 2).cuda()
        #visualize_bbox(args, cfg, image, [torch.Tensor(rot_coord).cuda()], net.prior, height_final/width_final)
        out = net(image, is_train=False)

        # Extract the predicted bboxes
        idx = out.data[0, 1, :, 0] >= 0.1
        text_boxes = out.data[0, 1, idx, 1:]
        pred = [[float(coor) for coor in area] for area in text_boxes]
        #BBox = [imgaug.imgaug.BoundingBox(box[0], box[1], box[2], box[3])
                #for box in pred]
        #BBoxes = imgaug.imgaug.BoundingBoxesOnImage(BBox, shape=(square, square))
        #bbox = aug_seq.augment_bounding_boxes([BBoxes])[0]
        print_box(blue_boxes=pred, idx=i, img=vb.plot_tensor(args, image, margin=0),
                  save_dir=args.val_log)
        print(i)
        """
        scale = torch.Tensor([h, w, h, w]).unsqueeze(0).repeat(text_boxes.size(0), 1)
        text_boxes = text_boxes.cpu() * scale

        r_aug = return_aug(transform_det, height_ori, width_ori, height, width)
        r_aug = r_aug.to_deterministic()
        image = r_aug.augment_image(image)
        bbox = r_aug.augment_bounding_boxes([bbox])[0]

        pred_bbox = [imgaug.imgaug.BoundingBox([float(coor) for coor in area]) for area in text_boxes]
        BBox = imgaug.imgaug.BoundingBoxesOnImage(BBox, shape=img.shape)
        bbox_aug = crop_aug.augment_bounding_boxes(bbox)
        """

        #print_box(pred, img=img, idx=i)

if __name__ == "__main__":
    with torch.no_grad():
        test_rotation()