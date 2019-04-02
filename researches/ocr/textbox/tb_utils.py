#
# This file was copied from
# https://github.com/amdegroot/ssd.pytorch
#
import torch
import torch.nn.functional as F

def calculate_anchor_number(cfg, i):
    return len(cfg['box_ratios'][i]) + (0, len(cfg['box_ratios_large'][i]))[cfg['big_box']]


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    """
    # boxes[:, :2] represent the center_x and center_y
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1)


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2]], 1)


def box_jaccard(box_a, box_b):
    def calibration(x):
        """
        when x is 0 then y is: 0.49768748545490993
        when x is 0.5 then y is: 0.9113706829705495
        when x is 1 then y is: 1.1053614272370262
        when x is 5 then y is: 1.5603316608172944
        when x is 20 then y is: 1.9203769145103649
        when x is 50 then y is: 2.1389601886864544
        """
        mul = 0.7
        trans = 1
        y = torch.pow(x, 1/5)
        #y = torch.sqrt(torch.tanh(mul * (x + trans)) + torch.log(mul * (x + trans)))
        return y
    A = box_a.size(0)
    B = box_b.size(0)
    # Calculate intersect and union on x direction
    x1_min = torch.min(box_a[:, 0:1].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 0:1].unsqueeze(0).expand(A, B, 1))
    x1_max = torch.max(box_a[:, 0:1].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 0:1].unsqueeze(0).expand(A, B, 1))
    x2_min = torch.min(box_a[:, 2:3].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 2:3].unsqueeze(0).expand(A, B, 1))
    x2_max = torch.max(box_a[:, 2:3].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 2:3].unsqueeze(0).expand(A, B, 1))
    inter_x = torch.clamp((x2_min - x1_max), min=0).squeeze(-1)
    union_x = torch.clamp((x2_max - x1_min), min=0).squeeze(-1)
    j_x = (inter_x / union_x)

    # Calculate intersect and union on y direction
    y1_min = torch.min(box_a[:, 1:2].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 1:2].unsqueeze(0).expand(A, B, 1))
    y1_max = torch.max(box_a[:, 1:2].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 1:2].unsqueeze(0).expand(A, B, 1))
    y2_min = torch.min(box_a[:, 3:4].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 3:4].unsqueeze(0).expand(A, B, 1))
    y2_max = torch.max(box_a[:, 3:4].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 3:4].unsqueeze(0).expand(A, B, 1))
    inter_y = torch.clamp((y2_min - y1_max), min=0).squeeze(-1)
    union_y = torch.clamp((y2_max - y1_min), min=0).squeeze(-1)
    j_y = (inter_y / union_y)

    # Calculated intersected box ratio
    inter_r = inter_x / inter_y
    inter_r[inter_r != inter_r] = 0
    inter_r[inter_r == float("Inf")] = 0
    idx = inter_r == 0
    inter_r = calibration(inter_r)
    # enhance box when box is wide
    #inter_r = F.softmax(torch.stack([inter_r, 1/inter_r], 0), dim=0)
    # supress box when box is wide
    inter_r = F.softmax(torch.stack([1/inter_r, inter_r], 0), dim=0)

    j = torch.stack([j_x, j_y], 0)

    jac = torch.sum(inter_r * j, dim=0)
    jac[idx] = 0
    return jac


def get_box_size(box):
    """
    calculate the bound box size
    """
    return (box[:, 2]-box[:, 0]) * (box[:, 3]-box[:, 1])


def intersect(box_a, box_b):
    """ Calculate the intersection area of box_a & box_b
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    # supress the matching of very wide boxes
    # supressor = (inter[:, :, 0] / (inter[:, :, 1] + 1e-5)).clamp(min=1)
    return inter[:, :, 0] * inter[:, :, 1]#, torch.pow(supressor, 1/5)
    

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    jac = inter / (area_a + area_b - inter)
    #mul_jac = inter / suppressor / (area_a + area_b - inter / suppressor)
    # print(torch.sum(mul_jac > 0.9) - torch.sum(jac > 0.9))
    # return mul_jac
    return jac
    

def match(cfg, threshold, truths, priors, variances, labels, loc_t, conf_t, idx, visualize=False, jaccard=jaccard):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    if cfg['clip']:
        overlaps = jaccard(truths, point_form(priors).clamp_(max=1, min=0))
    else:
        overlaps = jaccard(truths, point_form(priors))
    # 找到与每个ground truth boxes最接近的prior boxes的IOU和index, length = num_gt
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # 找到与每个prior boxes最接近的ground truth boxes的IOU和index, lenght = num_prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # 剔除一些匹配度低的prior_idx
    _best_prior_idx = best_prior_idx[best_prior_overlap > 0.5]
    # 将剔除后的_best_prior_idx中所对应的overlap变为2
    best_truth_overlap.index_fill_(0, _best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    if visualize:
        return overlaps, conf
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def coord_to_rect(coord, height, width):
    """
    Convert 4 point boundbox coordinate to matplotlib rectangle coordinate
    """
    x1, y1, x2, y2 = coord[0], coord[1], coord[2] - coord[0], coord[3] - coord[1]
    return x1 * width, y1 * height, x2 * width, y2 * height


def get_parameter(param):
    """
    Convert input parameter to two parameter if they are lists or tuples
    Mainly used in tb_vis.py and tb_model.py
    """
    if type(param) is list or type(param) is tuple:
        assert len(param) == 2, "input parameter shoud be either scalar or 2d list or tuple"
        p1, p2 = param[0], param[1]
    else:
        p1, p2 = param, param
    return p1, p2


if __name__ == "__main__":
    a = torch.Tensor([[]])