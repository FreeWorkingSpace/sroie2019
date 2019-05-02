from os.path import *
import os, glob, json


def read_lines_from_txtfile(text_file):
    lines = []
    with open(text_file, "r") as txt_lines:
        try:
            for j, line in enumerate(txt_lines):
                line_element = line.split(",")
                text_label = ",".join(line_element[8:])
                lines.append(text_label.strip())
        except UnicodeDecodeError:
            print("Cannot read some characters in %s"%text_file)
    return lines


def read_line_from_json(json_file):
    lines = []
    with open(json_file, "r") as file:
        data = json.load(file)
        lines.append(data["company"].strip())
        try:
            lines.append(data["address"].strip())
        except KeyError:
            print(data)
    return lines


def create_bert_train_tsv(text_root, json_root, bert_root):
    file_names = [file for file in os.listdir(json_root) if file.endswith("txt")]
    train = open(join(bert_root, "sroie_data", "train.tsv"), "w")
    val = open(join(bert_root, "sroie_data",  "dev.tsv"), "w")
    samples = len(file_names)
    write_lines = 0
    for i, file_name in enumerate(file_names):
        if i > round(samples * 0.9):
            train.close()
            write = val
        else:
            write = train
        if not exists(join(text_root, file_name)):
            continue
        text_file = read_lines_from_txtfile(join(text_root, file_name))
        json_file = read_line_from_json(join(json_root, file_name))
        for j, text in enumerate(text_file):
            for key_info in json_file:
                if text in key_info:
                    # Write as true sample
                     write.write("data%d\t1\t*\t%s\n"%(write_lines, text))
                else:
                    # Write as false sample
                    write.write("data%d\t0\t*\t%s\n" % (write_lines, text))
                write_lines += 1
    val.close()
    return


if __name__ is "__main__":
    bert_root = expanduser("~/Documents/bert")
    bert_model = "uncased_L-12_H-768_A-12"
    task_1_2_text_root = expanduser("~/Pictures/dataset/ocr/SROIE2019")
    task_1_2_label_root = expanduser("~/Downloads/task_1_2_label")
    task_3_text_root = expanduser("~/Downloads/task_3_label")

    create_bert_train_tsv(task_1_2_text_root, task_1_2_label_root, bert_root)
