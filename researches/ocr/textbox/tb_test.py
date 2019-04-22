import os, time, sys, math, random, glob, datetime, argparse
sys.path.append(os.path.expanduser("~/Documents/sroie2019"))
import cv2, torch
import numpy as np
import imgaug
from imgaug import augmenters
import omni_torch.utils as util
import researches.ocr.textbox.tb_preset as preset
import researches.ocr.textbox.tb_model as model
from researches.ocr.textbox.tb_utils import *
from researches.ocr.textbox.tb_preprocess import *
from researches.ocr.textbox.tb_augment import *
from researches.ocr.textbox.tb_postprocess import combine_boxes
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
# Image will be resize to this size
square = 2048


def parse_arguments():
    parser = argparse.ArgumentParser(description='Textbox Detector Settings')
    ##############
    #        TRAINING        #
    ##############
    parser.add_argument(
        "-tdr",
        "--test_dataset_root",
        type=str,
        help="1 represent the latest model",
        default="~/Pictures/dataset/ocr/SROIE2019_test"
    )
    parser.add_argument(
        "-mpl",
        "--model_prefix_list",
        nargs='+',
        help="a list of model prefix to do the ensemble",
        default=["768"]
    )
    parser.add_argument(
        "-nth",
        "--nth_best_model",
        type=int,
        help="1 represent the latest model",
        default=1
    )
    parser.add_argument(
        "-dtk",
        "--detector_top_k",
        type=int,
        help="get top_k boxes from prediction",
        default=1500
    )
    parser.add_argument(
        "-dct",
        "--detector_conf_threshold",
        type=int,
        help="detector_conf_threshold",
        default=0.05
    )
    parser.add_argument(
        "-dnt",
        "--detector_nms_threshold",
        type=int,
        help="detector_nms_threshold",
        default=0.3
    )
    args = parser.parse_args()
    return args


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


def test_rotation(opt):
    # Load
    assert len(opt.model_prefix_list) <= torch.cuda.device_count(), \
        "number of models should not exceed the device numbers"
    nets = []
    for device_id, prefix in enumerate(opt.model_prefix_list):

        net = model.SSD(cfg, connect_loc_to_conf=True, fix_size=False,
                        incep_conf=True, incep_loc=True)
        net = net.to("cuda:%d"%(device_id))
        net_dict = net.state_dict()
        weight_dict = util.load_latest_model(args, net, prefix=prefix,
                                             return_state_dict=True, nth=opt.nth_best_model)
        loading_fail_signal = False
        for key in weight_dict.keys():
            if key[7:] in net_dict:
                if net_dict[key[7:]].shape == weight_dict[key].shape:
                    net_dict[key[7:]] = weight_dict[key]
                else:
                    print("Key: %s from disk has shape %s copy to the model with shape %s"%
                          (key[7:], str(weight_dict[key].shape), str(net_dict[key[7:]].shape)))
                    loading_fail_signal = True
            else:
                print("Key: %s does not exist in net_dict"%(key[7:]))
        if loading_fail_signal:
            raise RuntimeError('Shape Error happens, remove "%s" from your -mpl settings.'%(prefix))

        net.load_state_dict(net_dict)
        net.eval()
        nets.append(net)
        print("Above model loaded with out a problem")
    detector = model.Detect(num_classes=2, bkg_label=0,
                            top_k=opt.detector_top_k,
                            conf_thresh=opt.detector_conf_threshold,
                            nms_thresh=opt.detector_nms_threshold)
    
    # Enumerate test folder
    root_path = os.path.expanduser(opt.test_dataset_root)
    if not os.path.exists(root_path):
        raise FileNotFoundError("%s does not exists, please check your -tdr/--test_dataset_root settings"%(root_path))
    img_list = glob.glob(root_path + "/*.jpg")
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
        image_t = image_t.permute(0, 3, 1, 2)
        #visualize_bbox(args, cfg, image, [torch.Tensor(rot_coord).cuda()], net.prior, height_final/width_final)

        text_boxes = []
        for device_id, net in enumerate(nets):
            image_t = image_t.to("cuda:%d"%(device_id))
            out = net(image_t, is_train=False)
            loc_data, conf_data, prior_data = out
            prior_data = prior_data.to("cuda:%d"%(device_id))
            det_result = detector(loc_data, conf_data, prior_data)
            # Extract the predicted bboxes
            idx = det_result.data[0, 1, :, 0] >= 0.1
            text_boxes.append(det_result.data[0, 1, idx, 1:])
        text_boxes = torch.cat(text_boxes, dim=0)
        text_boxes = combine_boxes(text_boxes, w=w_final, h=h_final)
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
            f.write("%d,%d,%d,%d,%d,%d,%d,%d\n"%(x1, y1, x2, y1, x2, y2, x1, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 105, 65), 2)
        img_save_directory = os.path.join(args.path, args.code_name, "val+" + "-".join(opt.model_prefix_list))
        if not os.path.exists(img_save_directory):
            os.mkdir(img_save_directory)
        cv2.imwrite(os.path.join(img_save_directory, name + ".jpg"), img)
        f.close()
        print("%d th image cost %.2f seconds"%(i, time.time() - start))


if __name__ == "__main__":
    opt = parse_arguments()
    with torch.no_grad():
        test_rotation(opt)