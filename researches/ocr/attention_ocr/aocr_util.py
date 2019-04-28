import torch


def avg(list):
    return sum(list) / len(list)

def invert_dict(dict):
    new_dict = {}
    for key in sorted(dict.keys()):
        if dict[key] in new_dict:
            raise KeyError("The value (%s) of input dictionary should be unique"%(key))
        if key in ["SOS", "EOS"]:
            new_dict.update({dict[key]: ""})
        else:
            new_dict.update({dict[key]: key})
    return new_dict

def extract_string(invert_dict, outputs, label_batch):
    index = [outputs[:, :, i].topk(1)[1] for i in range(outputs.size(2))]
    index = torch.cat(index, dim=1)
    pred_str = []
    for idx in index:
        pred_str.append("".join([invert_dict[int(i)] for i in idx]))
    label_str = []
    for idx in label_batch:
        label_str.append("".join([invert_dict[int(i)] for i in idx]))
    return pred_str, label_str

def print_pred_and_label(pred, label, print_correct=False):
    for i, p in enumerate(pred):
        if p == label[i] and p is not '':
            print("Correct: %s => %s" % (p, label[i]))
        else:
            if not print_correct:
                print("%s => %s" % (p, label[i]))