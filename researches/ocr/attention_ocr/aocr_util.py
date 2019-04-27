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

def print_pred_and_label(pred, label, print_correct=False):
    for i, p in enumerate(pred):
        if p == label[i] and p is not '':
            print("Correct: %s => %s" % (p, label[i]))
        else:
            if not print_correct:
                print("%s => %s" % (p, label[i]))