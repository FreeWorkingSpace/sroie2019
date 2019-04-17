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
result_dir = os.path.join(args.path, args.code_name, "result")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def augment_back(transform_det, height_ori, width_ori, v_crop, h_crop):
    aug_list = []
    v_crop = round(v_crop)
    h_crop = round(h_crop)
    # counteract pading
    aug_list.append(
        # top, right, bottom, left
        augmenters.Crop(px=(v_crop, h_crop, v_crop, h_crop))
    )
    # counteract resizing
    aug_list.append(
        augmenters.Resize(size={"height": height_ori, "width": width_ori})
    )
    # counteract rotation, if exist
    if "rotation" in transform_det:
        aug_list.append(
            augmenters.Affine(rotate=-transform_det["rotation"], cval=args.aug_bg_color),
        )
    aug = augmenters.Sequential(aug_list, random_order=False)
    return aug


def test_rotation():
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
    
    # Enumerate test folder
    img_list = glob.glob(os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019_test/*.jpg"))
    for i, img_file in enumerate(sorted(img_list)):
        start = time.time()
        name = img_file[img_file.rfind("/") + 1 : -4]
        img = cv2.imread(img_file)
        height_ori, width_ori = img.shape[0], img.shape[1]

        # detect rotation for returning the image back
        img, transform_det = estimate_angle(img, args, None, None, None)
        if transform_det["rotation"] != 0:
            rot_aug = augmenters.Affine(rotate=transform_det["rotation"],
                                         cval=args.aug_bg_color)
        else:
            rot_aug = None

        # Perform Augmentation
        if rot_aug:
            rot_aug = augmenters.Sequential(
                augmenters.Affine(rotate=transform_det["rotation"], cval=args.aug_bg_color))
            image = rot_aug.augment_image(img)
        else:
            image = img
        # Resize the longer side to a certain length
        square = 1536
        if height_ori >= width_ori:
            resize_aug =augmenters.Sequential([
                augmenters.Resize(size={"height": square, "width": "keep-aspect-ratio"})])
        else:
            resize_aug = augmenters.Sequential([
                augmenters.Resize(size={"height": "keep-aspect-ratio", "width": square})])
        resize_aug = resize_aug.to_deterministic()
        image = resize_aug.augment_image(image)
        h_re, w_re = image.shape[0], image.shape[1]
        # Pad the image into a square image
        pad_aug = augmenters.Sequential(
            augmenters.PadToFixedSize(width=square, height=square, pad_cval=255, position="center")
        )
        pad_aug = pad_aug.to_deterministic()
        image = pad_aug.augment_image(image)
        h_final, w_final= image.shape[0], image.shape[1]

        # Prepare image tensor and test
        image_t = torch.Tensor(util.normalize_image(args, image)).unsqueeze(0)
        image_t = image_t.permute(0, 3, 1, 2).cuda()
        #visualize_bbox(args, cfg, image, [torch.Tensor(rot_coord).cuda()], net.prior, height_final/width_final)
        out = net(image_t, is_train=False)

        # Extract the predicted bboxes
        idx = out.data[0, 1, :, 0] >= 0.1
        text_boxes = out.data[0, 1, idx, 1:]
        pred = [[float(coor) for coor in area] for area in text_boxes]
        BBox = [imgaug.imgaug.BoundingBox(box[0] * w_final, box[1] * h_final, box[2] * w_final, box[3] * h_final)
                for box in pred]
        BBoxes = imgaug.imgaug.BoundingBoxesOnImage(BBox, shape=image.shape)
        return_aug = augment_back(transform_det, height_ori, width_ori, (h_final - h_re) / 2, (w_final - w_re) / 2)
        return_aug = return_aug.to_deterministic()
        img_ori = return_aug.augment_image(image)
        bbox = return_aug.augment_bounding_boxes([BBoxes])[0]
        #print_box(blue_boxes=pred, idx=i, img=vb.plot_tensor(args, image_t, margin=0),
                  #save_dir=args.val_log)
        
        f = open(os.path.join(result_dir, name + ".txt"), "w")
        for box in bbox.bounding_boxes:
            x1, y1, x2, y2 = int(round(box.x1)), int(round(box.y1)), int(round(box.x2)), int(round(box.y2))
            # 4-point to 8-point: x1, y1, x2, y1, x2, y2, x1, y2
            f.write("%d, %d, %d, %d, %d, %d, %d, %d\n"%(x1, y1, x2, y1, x2, y2, x1, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 105, 65), 2)
        cv2.imwrite(os.path.join(args.val_log, name + ".jpg"), img)
        f.close()
        print("%d th image cost %.2f seconds"%(i, time.time() - start))
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


if __name__ == "__main__":
    with torch.no_grad():
        test_rotation()