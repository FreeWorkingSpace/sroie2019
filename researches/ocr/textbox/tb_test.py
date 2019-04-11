def test_rotation():
    def return_aug(transform_det, height_ori, width_ori, height, width):
        aug_list = []
        aug_list.append(
            augmenters.Resize(size={"height": height, "width": width})
        )
        if "crop" in transform_det:
            top_crop, right_crop, bottom, left = transform_det["crop"]
            left = (left + width/2) / width_ori
            bottom = (bottom + height/2) / height_ori
            aug_list.append(
                augmenters.PadToFixedSize(width=width_ori, height=height_ori, position=(left, bottom)),
            )
        if "rotation" in transform_det:
            aug_list.append(
                augmenters.Affine(rotate=-transform_det["rotation"], cval=args.aug_bg_color, fit_output=True),
            )
        aug = augmenters.Sequential(aug_list, random_order=False)
        return aug
    def extract_boxes(bboxes, h, w, box_label):
        coords = []
        #h, w = image.shape[0], image.shape[1]
        for i, bbox in enumerate(bboxes.bounding_boxes):
            condition_1 = bbox.x1 <= 0 and bbox.x2 <= 0
            condition_2 = bbox.y1 <= 0 and bbox.y2 <= 0
            condition_3 = bbox.x1 >= w - 1 and bbox.x2 >= w - 1
            condition_4 = bbox.y1 >= h - 1 and bbox.y2 >= h - 1
            if condition_1 or condition_2 or condition_3 or condition_4:
                # Eliminate bboxes outside the image
                continue
            horizontal_constrain = lambda x: max(min(w, x), 0)
            vertival_constrain = lambda y: max(min(h, y), 0)
            coords.append([horizontal_constrain(bbox.x1) / w, vertival_constrain(bbox.y1) / h,
                           horizontal_constrain(bbox.x2) / w, vertival_constrain(bbox.y2) / h, box_label[i]])
        return coords
    import imgaug
    from imgaug import augmenters
    net = model.SSD(cfg, connect_loc_to_conf=True, fix_size=False)
    net = net.cuda()
    net_dict = net.state_dict()
    weight_dict = util.load_latest_model(args, net, prefix="cv_1", return_state_dict=True)
    for key in weight_dict.keys():
        net_dict[key[7:]] = weight_dict[key]
    net.load_state_dict(net_dict)
    img_list = glob.glob(os.path.expanduser("~/Pictures/sroie_new/*.jpg"))
    for i, img_file in enumerate(sorted(img_list)):
        # Get img and bbox infomation from local file
        img, bbox, box_label = data.extract_bbox(args, [img_file, img_file[:-4] + ".txt"], None, None)
        height_ori, width_ori = img.shape[0], img.shape[1]
        # detect rotation and crop area and save it for returning the image back
        img, transform_det = estimate_angle_and_crop_area(img, args, None, None, None)
        if "rotation" in transform_det:
            rot_aug = augmenters.Affine(rotate=transform_det["rotation"],
                                         cval=args.aug_bg_color, fit_output=True)
        else:
            rot_aug = None
        if "crop" in transform_det:
            crop_aug = [augmenters.Crop(px=transform_det["crop"], keep_size=False)]
        else:
            crop_aug = None
        # Augment img and bbox, even if rotation exists, we only rotate img not bbox
        if rot_aug:
            rot_aug = augmenters.Sequential(rot_aug, random_order=False)
            rot_bbox = rot_aug.augment_bounding_boxes([bbox])[0]
            image = rot_aug.augment_image(img)
        else:
            image = img
            rot_bbox = None
        if crop_aug:
            crop_aug = augmenters.Sequential(crop_aug, random_order=False)
            crop_aug = crop_aug.to_deterministic()
            image = crop_aug.augment_image(image)
            bbox = crop_aug.augment_bounding_boxes([bbox])[0]
            if rot_bbox:
                rot_bbox = crop_aug.augment_bounding_boxes([rot_bbox])[0]
        height, width = image.shape[0], image.shape[1]
        # Resize the image to a number divideable by GCD
        # So as to estimate the feature map size
        gcd = 32
        height_resize = round(height / gcd) * gcd
        width_resize = round(width / gcd) * gcd
        resize_aug =augmenters.Sequential([
            augmenters.Resize(size={"height": 1472, "width": 512}),
            augmenters.CropToFixedSize(height=128, width=512),
        ])
        resize_aug = resize_aug.to_deterministic()
        image = resize_aug.augment_image(image)
        bbox = resize_aug.augment_bounding_boxes([bbox])[0]
        if rot_bbox:
            rot_bbox = resize_aug.augment_bounding_boxes([rot_bbox])[0]
        # Get the final size of resized input image
        height_final, width_final = image.shape[0], image.shape[1]
        # Generate prior boxes according to the input image size
        cfg["feature_map_sizes"] = [[height_final/8, width_final/8], [height_final/16, width_final/16],
                                     [height_final/32, width_final/32]]
        net.prior = net.create_prior(input_size=(height_final, width_final)).cuda()
        # Collect bboxes inside the image
        coord = extract_boxes(bbox, height_final, width_final, box_label)
        rot_coord = extract_boxes(rot_bbox, height_final, width_final, box_label)

        # Prepare image tensor and test
        image = torch.Tensor(util.normalize_image(args, image)).unsqueeze(0)
        image = image.permute(0, 3, 1, 2).cuda()
        visualize_bbox(args, cfg, image, [torch.Tensor(rot_coord).cuda()], net.prior, height_final/width_final)
        b, c, h, w = image.shape
        out = net(image, is_train=False)

        # Extract the predicted bboxes
        idx = out.data[0, 1, :, 0] >= 0.4
        text_boxes = out.data[0, 1, idx, 1:]
        scale = torch.Tensor([h, w, h, w]).unsqueeze(0).repeat(text_boxes.size(0), 1)
        text_boxes = text_boxes.cpu() * scale

        r_aug = return_aug(transform_det, height_ori, width_ori, height, width)
        r_aug = r_aug.to_deterministic()
        image = r_aug.augment_image(image)
        bbox = r_aug.augment_bounding_boxes([bbox])[0]

        pred_bbox = [imgaug.imgaug.BoundingBox([float(coor) for coor in area]) for area in text_boxes]
        BBox = imgaug.imgaug.BoundingBoxesOnImage(BBox, shape=img.shape)
        bbox_aug = crop_aug.augment_bounding_boxes(bbox)

        #print_box(pred, img=img, idx=i)
