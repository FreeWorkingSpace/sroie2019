import os, glob, math
import torch, cv2
import torch.nn.functional as F
import numpy as np
import omni_torch.utils as util


def detect_angle(img):
    def weighted_median(data, weights):
        """
        computes weighted median
        """
        midpoint = weights.sum() / 2
        if any(weights > midpoint):
            return (data[weights == np.max(weights)])[0]
        indsort = data.argsort()
        weights_sorted = weights[indsort]
        weight_sums = np.cumsum(weights_sorted)
        idx = np.where(weight_sums <= midpoint)[0][-1]
        if weight_sums[idx] == midpoint:
            return np.mean(data[indsort][idx:idx + 2])
        return data[indsort][idx + 1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
    lines = lsd.detect(img_gray)[0]
    if lines is None or not lines.any():
        return None
    angles = np.zeros(lines.shape[0])
    lengths = np.zeros(lines.shape[0])
    for i, (pos_x1, pos_y1, pos_x2, pos_y2) in enumerate(lines.reshape(-1, 4)):
        angles[i] = math.atan2(pos_y2 - pos_y1, pos_x2 - pos_x1)
        lengths[i] = math.sqrt((pos_y2 - pos_y1) ** 2 + (pos_x2 - pos_x1) ** 2)
    angles[angles < 0] += math.pi
    angles[angles >= math.pi / 2] -= math.pi

    angles_s = angles + math.pi / 2
    angles_s[angles_s >= math.pi / 2] -= math.pi

    angles_c = np.hstack([angles, angles_s])
    lengths = np.hstack([lengths, lengths])

    lengths = lengths[angles_c >= -math.pi / 4]
    angles_c = angles_c[angles_c >= -math.pi / 4]
    lengths = lengths[angles_c < math.pi / 4]
    angles_c = angles_c[angles_c < math.pi / 4]

    hist, bins = np.histogram(
        angles_c, bins=90, range=(-math.pi / 4, math.pi / 4), weights=lengths)
    angle_offset = bins[np.argmax(hist)]

    angles_c -= angle_offset
    angles_c[angles_c < -math.pi / 4] += math.pi / 2
    angles_c[angles_c >= math.pi / 4] -= math.pi / 2
    try:
        labels = cv2.kmeans(np.float32(angles_c.ravel()), 3, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1),
            10, cv2.KMEANS_PP_CENTERS )[1]
    except:
        # Sometime cv2 error will be raised by N>K in k-means
        return None
    masks = [(labels.ravel() == label) for label in range(3)]
    mask_sizes = [lengths[masks[label]].sum() for label in range(3)]
    label = np.argmax(mask_sizes)
    if mask_sizes[label] < labels.size / 2:
        return None
    angle = weighted_median(angles_c[masks[label]], lengths[masks[label]])
    angle += angle_offset
    if angle < -math.pi / 4:
        angle += math.pi / 2
    if angle >= math.pi / 4:
        angle -= math.pi / 2
    return angle * -1


def eatimate_angle(signal, args, path, seed, size, device=None):
    transform_det = {}
    signal = clahe_inv(signal, args, path, seed, size)
    original_size = signal.shape
    # Resize to small image for detect rotation angle
    #width = original_size[1] / original_size[0] * 1000
    #_signal = cv2.resize(signal, (int(width), 1000))
    angle = detect_angle(signal)
    if angle is not None and abs(angle) * 90 > 1:
        #print("angle: %s"%angle)
        transform_det.update({"rotation": angle * 90})
    return signal, transform_det


def estimate_angle_and_crop_area(signal, args, path, seed, size, device=None):
    """
    Pre-Process function for SROIE
    Remove the white sorrounding areas of input images
    """
    def norm_zero_one(x):
        min_v = torch.min(x)
        return (x - min_v) / (torch.max(x) - min_v)

    def rotate_image(img, angle):
        width = img.shape[1]
        height = img.shape[0]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, -angle / math.pi * 180, 1.0)
        new_width = int(abs(width * matrix[0, 0]) + abs(height * matrix[0, 1]))
        new_height = int(abs(height * matrix[0, 0]) + abs(width * matrix[0, 1]))
        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]
        return cv2.warpAffine(img, matrix, (new_width, new_height),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=np.median(img.reshape(-1, 3), axis=0)
                              )
    img = signal
    transform_det = {}
    threshold = 0.15
    if device is None:
        #device = args.device
        device = "cpu"
    gaussian_kernal = (0.1, 0.2, 0.4, 0.2, 0.1)
    ascend_kernel = (0.0, 0.25, 0.5, 0.75, 1.0)
    descend_kernel = (1.0, 0.75, 0.5, 0.25, 0.0)
    # Use CLAHE to enhance the contrast
    signal = clahe_inv(signal, args, path, seed, size)
    original_size = signal.shape
    # Resize to small size result to bad estimation
    #width = original_size[1] / original_size[0] * 500
    #_signal = cv2.resize(signal, (int(width), 500))
    angle = detect_angle(signal)
    if angle is not None and abs(angle) * 90 > 1:
        signal = rotate_image(signal, angle)
    # After rotation, the image size will change
    original_size = signal.shape
    if len(signal.shape) == 2:
        signal = np.expand_dims(signal, -1).astype(np.float32)
    else:
        signal = signal.astype(np.float32)
    signal = util.normalize_image(args, signal)
    # Transform (H, W, C) nd.array into (C, H, W) Tensor
    signal = torch.Tensor(signal).float().permute(2, 0, 1)
    # Convert to grey scale
    signal = torch.sum(signal, 0)
    # signal_x and signal_y represent the horizontal and vertical signal strength
    signal_x = (torch.sum(signal, dim=0) / signal.size(0)).unsqueeze(0).unsqueeze(0).to(device)
    signal_x = 1 - norm_zero_one(signal_x)
    signal_y = (torch.sum(signal, dim=1) / signal.size(1)).unsqueeze(0).unsqueeze(0).to(device)
    signal_y = 1 - norm_zero_one(signal_y)
    # Define the kernel
    #gaussian = args.gaussian
    #detector1 = args.detector1
    #detector2 = args.detector2
    gaussian = torch.tensor(gaussian_kernal).unsqueeze(0).unsqueeze(0).to(device)
    detector1 = torch.tensor(ascend_kernel).unsqueeze(0).unsqueeze(0).to(device)
    detector2 = torch.tensor(descend_kernel).unsqueeze(0).unsqueeze(0).to(device)

    start, end = [], []
    for signal in [signal_x, signal_y]:
        # Due to the size is very big, we do not need zero-padding
        smooth_signal = F.conv1d(signal, gaussian, stride=1, padding=0)
        # Calculate first derivative
        ascend_signal = F.conv1d(smooth_signal, detector1, stride=1, padding=0).squeeze()
        descend_signal = F.conv1d(smooth_signal, detector2, stride=1, padding=0).squeeze()

        # safe distance is 5% of current signal length
        safe_distance = int(0.05 * signal.size(-1))
        start_idx = (ascend_signal >= threshold).nonzero().squeeze(1)
        if start_idx.nelement() == 0:
            # Cannot find a ascend signal stronger than threshold
            _start = 0
        else:
            _start = max(0, int(start_idx[0]) - safe_distance)
        end_idx = (descend_signal >= threshold).nonzero().squeeze(1)
        if end_idx.nelement() == 0:
            _end = signal.size(-1)
        else:
            _end = min(signal.size(-1), int(end_idx[-1]) + safe_distance)
        if _end > _start + 300:
            start.append(_start)
            end.append(_end)
        else:
            print("assume some error happens in smart crop")
            start.append(0)
            end.append(signal.size(-1))
    if angle is not None and abs(angle) * 90 > 1:
        #print("angle: %s"%(angle))
        transform_det.update({"rotation": angle * 90})
    # 4 dimension means distance to top, right, bottom, left
    crop_area = (start[1], int(original_size[1] - end[0]), int(original_size[0] - end[1]), int(start[0]))
    if not crop_area == (0, 0, 0, 0):
        transform_det.update({"crop": crop_area})
    return img, transform_det


def clahe_inv(img, args, path, seed, size):
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 1, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_ab = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab_planes[0] =cv2.normalize(lab_planes[0], None, -25, 270, cv2.NORM_MINMAX)
    #lab_planes[1] = clahe_ab.apply(lab_planes[1])
    #lab_planes[2] = clahe_ab.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    #img = cv2.bilateralFilter(img, 5, 1, 1)
    return img


def refine_dataset():
    return

if __name__ == "__main__":
    def prepare_aug(transform_det):
        aug_list = []
        if "rotation" in transform_det:
            aug_list.append(
                augmenters.Affine(rotate=transform_det["rotation"], cval=args.aug_bg_color, fit_output=True),
            )
        if "crop" in transform_det:
            top_crop, right_crop, bottom, left = transform_det["crop"]
            aug_list.append(
                augmenters.Crop(px=(top_crop, right_crop, bottom, left), keep_size=False),
            )
        aug = augmenters.Sequential(aug_list, random_order=False)
        return aug

    def sroie_data_summary():
        width_max = 0
        width_min = 999999
        height_max = 0
        height_min = 999999
        width, height = [], []
        for img_file in sorted(glob.glob(os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019/*.jpg"))):
            print(img_file)
            img = cv2.imread(img_file)
            h, w = img.shape[0], img.shape[1]
            width.append(w)
            height.append(h)
            width_min = min(width_min, w)
            width_max = max(width_max, w)
            height_min = min(height_min, h)
            height_max = max(height_max, h)

        print("width_min: %s" % (width_min))
        print("height_min: %s" % (height_min))
        print("width_max: %s" % (width_max))
        print("height_max: %s" % (height_max))
        print("height_avg %s" % (sum(height) / len(height)))
        print("width_avg: %s" % (sum(width) / len(width)))
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
        ax[0].hist(np.asarray(width), bins=100)
        ax[1].hist(np.asarray(height), bins=100)
        plt.show()

    import matplotlib.pyplot as plt
    import time
    from imgaug import augmenters
    import researches.ocr.textbox.tb_preset as preset
    import researches.ocr.textbox.tb_augment as augment
    from random import shuffle
    #sroie_data_summary()
    args = util.get_args(preset.PRESET)
    start = 0
    i = 0
    aug = augmenters.Sequential(augment.aug_sroie())
    img_files = sorted(glob.glob(os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019/*.jpg")))
    #img_files = sorted(glob.glob(os.path.expanduser("~/Pictures/sroie_typical/*.jpg")))
    #shuffle(img_files)
    for img_file in img_files:
        start_time = time.time()
        num = img_file[img_file.rfind("/") + 1:-4]
        i += 1
        if i < start:
            continue
        img = cv2.imread(img_file)
        img, det = estimate_angle_and_crop_area(img, args, None, None, None, device="cpu")
        transform = prepare_aug(det)
        img = transform.augment_image(img)
        print(img.shape)
        img = aug.augment_image(img)
        print(img.shape)
        cv2.imwrite(os.path.expanduser("~/Pictures/%s.jpg" % (num)), img)
        print("%s cost %.3f seconds"%(img_file, time.time() - start_time))
        pass



