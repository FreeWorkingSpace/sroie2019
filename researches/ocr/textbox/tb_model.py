import torch, sys, os, math
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torchvision.models import vgg16_bn
import omni_torch.networks.blocks as omth_blocks
import researches.ocr.textbox as init
from researches.ocr.textbox.tb_utils import *

cfg = {
    # Configuration for 512x512 input image
    'num_classes': 2,
    # Which conv layer output to use
    # The program create the prior box according to the length of conv_output
    # As long as its length does not exceed the length of other value
    # e.g. feature_map_sizes, box_height, box_height_large
    # Then it will be OK
    'conv_output': ["conv_4", "conv_5", "extra_2"],
    'feature_map_sizes': [64, 32, 16],
    # For static input size only, when Dynamic mode is turned out, it will not be used
    # Must be 2d list or tuple
    'input_img_size': [512, 512],
    # See the visualization result by enabling visualize_bbox in function fit of textbox.py
    # And change the settings according to the result
    # Some possible settings of box_height and box_height_large
    # 'box_height': [[16], [26], [36]],
    # 'box_height': [[10, 16], [26], [36]],
    # 'box_height': [[16], [26], []],
    'box_height': [[14], [24], [38]],
    'box_ratios': [[2, 4, 7, 11, 15, 20, 26], [1, 2, 5, 9, 13, 16, 18, 20], [1, 2, 5, 8, 10, 12]],
    # If big_box is True, then box_height_large and box_ratios_large will be used
    'big_box': True,
    'box_height_large': [[18], [30], [46]],
    'box_ratios_large': [[1, 2, 4, 7, 11, 15, 20], [0.5, 1, 3, 6, 9, 11, 13, 15], [1, 2, 4, 7, 9, 11]],
    # You can increase the stride when feature_map_size is large
    # especially at swallow conv layers, so as not to create lots of prior boxes
    'stride': [1, 1, 1],
    # Input depth for location and confidence layers
    'loc_and_conf': [512, 512, 512],
    # The hyperparameter to decide the Loss
    'variance': [0.1, 0.2],
    'var_updater': 1,
    'alpha': 1,
    'alpha_updater': 1,
    # Jaccard Distance Threshold
    'overlap_thresh': 0.45,
    # Whether to constrain the prior boxes inside the image
    'clip': True,
}


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output.cuda()


class SSD(nn.Module):
    def __init__(self, cfg, in_channel=512, batch_norm=nn.BatchNorm2d, fix_size=True,
                 connect_loc_to_conf=False, incep_loc=False, incep_conf=False, nms_thres=0.2):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        self.output_list = cfg['conv_output']
        self.conv_module = nn.ModuleList([])
        self.loc_layers = nn.ModuleList([])
        self.conf_layers = nn.ModuleList([])
        self.conf_concate = nn.ModuleList([])
        self.conv_module_name = []
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(self.num_classes, 0, 1500, 0.01, nms_thres)
        self.connect_loc_to_conf = connect_loc_to_conf
        self.fix_size = fix_size
        if fix_size:
            self.prior = self.create_prior()#.cuda()

        # Prepare VGG-16 net with batch normalization
        vgg16_model = vgg16_bn(pretrained=True)
        net = list(vgg16_model.children())[0]
        # Replace the maxout with ceil in vanilla vgg16 net
        ceil_maxout = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        net = [ceil_maxout if type(n) is nn.MaxPool2d else n for n in net]
        
        # Basic VGG Layers
        self.conv_module_name.append("conv_1")
        self.conv_module.append(nn.Sequential(*net[:6]))
        self.conv_module_name.append("conv_2")
        #block = omth_blocks.conv_block(64, filters=[64, 128],kernel_sizes=[3, 3], stride=[1, 1],
        #                               padding=[1, 1],batch_norm=batch_norm)
        #block.add_module("2", ceil_maxout)
        #self.conv_module.append(block)
        self.conv_module.append(nn.Sequential(*net[6:13]))
        self.conv_module_name.append("conv_3")
        #block = omth_blocks.conv_block(128, filters=[128, 128], kernel_sizes=[3, 1], stride=[1, 1],
        #                               padding=[1, 0], batch_norm=batch_norm)
        #block.add_module("3", ceil_maxout)
        #self.conv_module.append(block)
        self.conv_module.append(nn.Sequential(*net[13:23]))
        self.conv_module_name.append("conv_4")
        #block = omth_blocks.conv_block(128, filters=[256, 256], kernel_sizes=[3, 1], stride=[1, 1],
        #                               padding=[1, 0], batch_norm=batch_norm)
        #block.add_module("4", ceil_maxout)
        #self.conv_module.append(block)
        self.conv_module.append(nn.Sequential(*net[23:33]))
        self.conv_module_name.append("conv_5")
        #block = omth_blocks.conv_block(256, filters=[384, 384], kernel_sizes=[3, 1], stride=[1, 1],
        #                               padding=[1, 0], batch_norm=batch_norm)
        #block.add_module("5", ceil_maxout)
        #self.conv_module.append(block)
        self.conv_module.append(nn.Sequential(*net[33:43]))

        # Extra Layers
        self.conv_module_name.append("extra_1")
        self.conv_module.append(omth_blocks.conv_block(512, [512, 512],
                                                       kernel_sizes=[3, 1], stride=[1, 1], padding=[3, 0],
                                                       dilation=[3, 1], batch_norm=batch_norm))
        self.conv_module_name.append("extra_2")
        self.conv_module.append(omth_blocks.conv_block(512, [256, 512], kernel_sizes=[1, 3],
                                                       stride=[1, 2], padding=[0, 1], batch_norm=batch_norm))
        #self.conv_module.apply(init.init_cnn)
        
        # Location and Confidence Layer
        for i, in_channel in enumerate(cfg['loc_and_conf']):
            anchor = calculate_anchor_number(cfg, i)
            # Create Location Layer
            loc_layer = Loc_Layer(in_channel, anchor, cfg['stride'][i], incep_loc=incep_loc, in_wid=128)
            self.loc_layers.append(loc_layer)
            # Create Confidence Layer
            if self.connect_loc_to_conf:
                if incep_conf:
                    conf_layer = omth_blocks.InceptionBlock(in_channel, stride=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1]],
                                                            kernel_sizes=[[[7, 1], 3, 1], [[5, 1], 3, 1], [[3, 1], 3, 1], [3, 1]],
                                                            filters=[[128, 128, 64], [128, 128, 64], [128, 128, 64], [192, 64]],
                                                            padding=[[[3, 0], 1, 0], [[2, 0], 1, 0], [[1, 0], 1, 0], [1, 0]],
                                                            batch_norm=None, inner_maxout=None)
                else:
                    conf_layer = omth_blocks.conv_block(in_channel, filters=[in_channel, int(in_channel / 2)],
                                                        kernel_sizes=[1, 3], stride=[1, 1], padding=[0, 1], activation=None)
                conf_layer.apply(init.init_cnn)
                self.conf_layers.append(conf_layer)
                # In this layer, the output from loc_layer will be concatenated to the conf layer
                # Feeding the conf layer with regressed location, helping the conf layer
                # to get better prediction
                conf_concat = omth_blocks.conv_block(256 + anchor * 4,
                                                     filters=[int(in_channel / 4), anchor * 2], kernel_sizes=[1, 3],
                                                     stride=[1, cfg['stride'][i]], padding=[0, 1], activation=None)
                conf_concat.apply(init.init_cnn)
                self.conf_concate.append(conf_concat)
            else:
                print("incep_conf is turned off due to connect_loc_to_conf is False")
                conf_layer = omth_blocks.conv_block(in_channel, filters=[in_channel, int(in_channel / 2), anchor * 2],
                                                    kernel_sizes=[1, 3, 3], stride=[1, 1, cfg['stride'][i]],
                                                    padding=[0, 1, 1], activation=None)
                conf_layer.apply(init.init_cnn)
                self.conf_layers.append(conf_layer)

    def create_prior(self, feature_map_size=None, input_size=None):
        """
        :param feature_map_size:
        :param input_size: When input size is not None. which means Dynamic Input Size
        :return:
        """
        from itertools import product as product
        mean = []
        big_box = self.cfg['big_box']
        if feature_map_size is None:
            assert len(self.cfg['feature_map_sizes']) >= len(self.cfg['conv_output'])
            feature_map_size = self.cfg['feature_map_sizes']
        if input_size is None:
            input_size = cfg['input_img_size']
        assert len(input_size) == 2, "input_size should be either int or list of int with 2 elements"
        input_ratio = input_size[1] / input_size[0]
        for k in range(len(self.cfg['conv_output'])):
            # Get setting for prior creation from cfg
            h, w = get_parameter(feature_map_size[k])
            h_stride, w_stride = get_parameter(cfg['stride'][k])
            for i, j in product(range(0, int(h), int(h_stride)), range(0, int(w), int(w_stride))):
                # 4 point represent: center_x, center_y, box_width, box_height
                cx = (j + 0.5) / w
                cy = (i + 0.5) / h
                # Add prior boxes with different height and aspect-ratio
                for height in self.cfg['box_height'][k]:
                    s_k = height / input_size[0]
                    for box_ratio in self.cfg['box_ratios'][k]:
                        mean += [cx, cy, s_k * box_ratio, s_k]
                # Add prior boxes with different number aspect-ratio if the box is large
                if big_box:
                    for height in self.cfg['box_height_large'][k]:
                        s_k_big = height / input_size[0]
                        for box_ratio_l in self.cfg['box_ratios_large'][k]:
                            mean += [cx, cy, s_k_big * box_ratio_l, s_k_big]
        # back to torch land
        prior_boxes = torch.Tensor(mean).view(-1, 4)
        if self.cfg['clip']:
            #boxes = center_size(prior_boxes, input_ratio)
            prior_boxes.clamp_(max=1, min=0)
            #prior_boxes = point_form(boxes, input_ratio)
        return prior_boxes

    def forward(self, x, is_train=True, verbose=False):
        input_size = [x.size(2), x.size(3)]
        locations, confidences, conv_output = [], [], []
        feature_shape = []
        for i, conv_layer in enumerate(self.conv_module):
            x = conv_layer(x)
            # Get shape from each conv output so as to create prior
            if self.conv_module_name[i] in self.output_list:
                conv_output.append(x)
                feature_shape.append((x.size(2), x.size(3)))
                if verbose:
                    print("CNN output shape: %s" % (str(x.shape)))
                if len(conv_output) == len(self.output_list):
                    # Doesn't need to compute further convolutional output
                    break
        if not self.fix_size:
            self.prior = self.create_prior(feature_map_size=feature_shape, input_size=input_size).cuda()
        for i, x in enumerate(conv_output):
            loc = self.loc_layers[i](x)
            locations.append(loc.permute(0, 2, 3, 1).contiguous().view(loc.size(0), -1, 4))
            conf = self.conf_layers[i](x)
            if self.connect_loc_to_conf:
                _loc = loc.detach()
                conf = torch.cat([conf, _loc], dim=1)
                conf = self.conf_concate[i](conf)
            confidences.append(conf.permute(0, 2, 3, 1).contiguous().view(conf.size(0), -1, self.num_classes))
            if verbose:
                print("Loc output shape: %s\nConf output shape: %s" % (str(loc.shape), str(conf.shape)))
        locations = torch.cat(locations, dim=1)
        confidences = torch.cat(confidences, dim=1)
        if is_train:
            output = [locations, confidences, self.prior]
        else:
            output = self.detect(locations, self.softmax(confidences), self.prior)
        return output

class Loc_Layer(nn.Module):
    def __init__(self, in_channel, anchor, stride, incep_loc=False, in_wid=128, batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.loc_layer = nn.ModuleList([])
        self.loc_layer.append(omth_blocks.conv_block(in_channel, [in_channel, in_channel], kernel_sizes=[3, 1],
                                                     stride=[1, 1], padding=[3, 0], dilation=[3, 1], batch_norm=batch_norm))
        if incep_loc:
            self.loc_layer.append(omth_blocks.InceptionBlock(
                in_channel, filters=[[128, 128, in_wid], [128, 128, in_wid], [128, 128, in_wid], [192, in_wid]],
                kernel_sizes=[[[7, 1], 3, 1], [[5, 1], 3, 1], [[3, 1], 3, 1], [3, 1]],
                stride=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1]],
                padding=[[[3, 0], 1, 0], [[2, 0], 1, 0], [[1, 0], 1, 0], [1, 0]],
                batch_norm=None, inner_maxout=None)
            )
        input_channel = in_wid * 4 if incep_loc else in_channel
        self.loc_layer.append(omth_blocks.conv_block(
            input_channel, filters=[input_channel, int(input_channel / 2), anchor * 4],
            kernel_sizes=[3, 1, 3], stride=[1, 1, stride], padding=[0, 1, 1], activation=None)
        )
        self.loc_layer.apply(init.init_cnn)

    def forward(self, x):
        for layer in self.loc_layer:
            x = layer(x)
        return x
    
class Conf_Layer(nn.Module):
    def __init__(self, in_channel, anchor, stride, incep_conf=False, cancat_loc=True,
                 in_wid=128, batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.cancat_loc = cancat_loc
        self.conf_layer = nn.ModuleList([])
        self.conf_layer.append(omth_blocks.conv_block(
            in_channel, [in_channel, in_channel], kernel_sizes=[3, 1], stride=[1, 1],
            padding=[3, 0], dilation=[3, 1], batch_norm=batch_norm)
        )
        if self.connect_loc_to_conf:
            if incep_conf:
                conf_layer = omth_blocks.InceptionBlock(in_channel, stride=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1]],
                                                        kernel_sizes=[[[7, 1], 3, 1], [[5, 1], 3, 1], [[3, 1], 3, 1],
                                                                      [3, 1]],
                                                        filters=[[128, 128, 64], [128, 128, 64], [128, 128, 64],
                                                                 [192, 64]],
                                                        padding=[[[3, 0], 1, 0], [[2, 0], 1, 0], [[1, 0], 1, 0],
                                                                 [1, 0]],
                                                        batch_norm=None, inner_maxout=None)
            else:
                conf_layer = omth_blocks.conv_block(512, filters=[in_channel, int(in_channel / 2)],
                                                    kernel_sizes=[1, 3], stride=[1, 1], padding=[0, 1], activation=None)
            conf_layer.apply(init.init_cnn)
            self.conf_layers.append(conf_layer)
            # In this layer, the output from loc_layer will be concatenated to the conf layer
            # Feeding the conf layer with regressed location, helping the conf layer
            # to get better prediction
            self.conf_concate = omth_blocks.conv_block(256 + anchor * 4,
                                                 filters=[int(in_channel / 4), anchor * 2], kernel_sizes=[1, 3],
                                                 stride=[1, stride], padding=[0, 1], activation=None)
            self.conf_concate.apply(init.init_cnn)
        else:
            print("incep_conf is turned off due to connect_loc_to_conf is False")
            conf_layer = omth_blocks.conv_block(in_channel, filters=[in_channel, int(in_channel / 2), anchor * 2],
                                                kernel_sizes=[1, 3, 3], stride=[1, 1, stride],
                                                padding=[0, 1, 1], activation=None)
            conf_layer.apply(init.init_cnn)
            self.conf_layers.append(conf_layer)
        
        if incep_conf:
            self.loc_incep_layer = omth_blocks.InceptionBlock(in_channel, stride=[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1]],
                                                         kernel_sizes=[[[7, 1], 3, 1], [[5, 1], 3, 1], [[3, 1], 3, 1],[3, 1]],
                                                         filters=[[128, 128, in_wid], [128, 128, in_wid], [128, 128, in_wid], [192, in_wid]],
                                                         padding=[[[3, 0], 1, 0], [[2, 0], 1, 0], [[1, 0], 1, 0], [1, 0]],
                                                         batch_norm=None, inner_maxout=None)
            self.loc_incep_layer.apply(init.init_cnn)
        input_channel = in_wid * 4 if incep_conf else in_channel
        self.loc_layer = omth_blocks.conv_block(input_channel, filters=[input_channel, int(input_channel / 2), anchor * 4],
                               kernel_sizes=[3, 1, 3], stride=[1, 1, stride], padding=[0, 1, 1],
                               activation=None)
        self.loc_layer.apply(init.init_cnn)

    def forward(self, x):
        if self.incep_layer:
            x = self.loc_incep_layer(x)
        x = self.loc_layer(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512).to("cuda")
    #print(cfg)
    ssd = SSD(cfg, connect_loc_to_conf=True, in_channel=384, incep_conf=True, incep_loc=True).to("cuda")
    loc, conf, prior = ssd(x, verbose=True)
    print(loc.shape)
    print(conf.shape)
    print(prior.shape)
