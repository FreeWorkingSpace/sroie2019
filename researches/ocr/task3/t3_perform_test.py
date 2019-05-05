import os, sys, glob, argparse
from os.path import *
import numpy as np
import scipy.io as sio
from researches.ocr.task3.t3_make_data import make_data
from researches.ocr.task3.t3_rule_base import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='Task 3 settings')
    parser.add_argument(
        "--make_data",
        action="store_true"
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
    )
    parser.add_argument(
        "-br",
        "--bert_root",
        type = str,
        default = "~/Documents/bert"
    )
    parser.add_argument(
        "-bm",
        "--bert_model",
        type=str,
        default="uncased_L-12_H-768_A-12"
    )
    parser.add_argument(
        "-tet",
        "--test_text",
        type=str,
        default="~/Downloads/task_3_label"
    )
    parser.add_argument(
        "-trt",
        "--train_text",
        type=str,
        default="~/Pictures/dataset/ocr/SROIE2019"
    )
    parser.add_argument(
        "-trki",
        "--train_key_info",
        type=str,
        default="~/Downloads/task_1_2_label"
    )
    parser.add_argument(
        "--sequence",
        nargs='+',
        help="a list folder/folders to use as training set",
        default=["address", "company", "date"]
    )
    args = parser.parse_args()
    return args


def do_predict(bert_root):
    os.chdir(bert_root)
    unchange_command = "python3 run_classifier.py --task_name=CoLA --do_predict=true " \
                       "--max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 " \
                       "--num_train_epochs=3.0"
    vocab_file = "--vocab_file=%s"%(join(bert_root, bert_model, "vocab.txt"))
    config_file = "-bert_config_file=%s"%(join(bert_root, bert_model, "bert_config.json"))
    checkpoint = "--init_checkpoint=%s"%(join(bert_root, bert_model, "bert_model.ckpt"))

    # perform prediction using BERT's CoLA Mode
    for task in args.sequence:
        data_dir = "--data_dir=%s/sroie_%s"%(bert_root, task)
        out_dir = "--output_dir=%s/tmp/cola_%s" % (bert_root, task)
        bert_predict_command = " ".join([unchange_command, vocab_file, config_file, checkpoint, data_dir, out_dir])
        os.system(bert_predict_command)

    print("##################################")
    print("#                                #")
    print("#      PREDICTION COMPLETED      #")
    print("#                                #")
    print("#   generating the json file...  #")
    print("#                                #")
    print("##################################")



args = parse_arguments()
if __name__ == "__main__":
    bert_root = expanduser(args.bert_root)
    bert_model = args.bert_model
    task_1_2_text_root = expanduser(args.train_text)
    task_1_2_label_root = expanduser(args.train_key_info)
    task_3_text = expanduser(args.test_text)
    month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
             "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    # generate data
    if args.make_data:
        make_data(task_1_2_text_root, task_1_2_label_root, task_3_text, bert_root)

    # Prepare command for test to run
    if args.do_predict:
        do_predict(bert_root)

    # Get line number from ground truth text label of test data
    num_list = []
    all_txt_lines = []
    text_files = sorted(glob.glob(task_3_text + "/*.txt"))
    for i, text_file in enumerate(text_files):
        txt_lines = open(text_file, "r").readlines()
        num_list.append(len(txt_lines))
        all_txt_lines.append(txt_lines)

    # Get the test result by BERT
    output = {}
    for task in args.sequence:
        output_file = join(bert_root, "tmp", "cola_%s"%task, "test_results.tsv")
        output_lines = open(output_file, "r").readlines()
        assert len(output_lines) == sum(num_list) - 1
        output.update({task: output_lines})

    # Get word frequency for company key info from train data
    company_set = get_key_info(task_1_2_label_root, keys=["company"],
                               split_word=True)[0]
    sort_dict = sorted(company_set.items(), key=lambda kv: (kv[1], kv[0]),
                       reverse=False)
    company_set = set([pair[0] for pair in sort_dict[300:]])

    # Generate JSON file
    start = 0
    for i, num in enumerate(num_list):
        #print("%d-th reciept:"%i)
        tmp_lines = [[[float(v) for v in val.strip().split("\t")]
                      for val in output[task][start : start + num]]
                     for task in args.sequence]
        idx = [np.argmax(np.asarray(lines), axis=1) for lines in tmp_lines]
        result = {}
        for j, task in enumerate(args.sequence):
            #print(np.sum(idx[j]))
            if np.sum(idx[j]) == 0:
                # Use rule-based method
                if task == "date":
                    for text in all_txt_lines[i]:
                        if is_date(text):
                            if "date" in result:
                                print("Another date: %s" % text)
                            else:
                                result.update({"date": text})
                    if "date" not in result:
                        # Split by dot
                        for text in all_txt_lines[i]:
                            dot_split = ",".join(text.strip().split(",")[8:]).split(".")
                            if len(dot_split) == 3 and len(dot_split[1]) > 1:
                                if all([char.isdigit() for char in dot_split[1]]) \
                                    or dot_split[1].upper() in month:
                                    result.update({"date": ",".join(text.strip().split(",")[8:])})
                    if "date" not in result:
                        # split by space
                        for text in all_txt_lines[i]:
                            space_split = ",".join(text.strip().split(",")[8:]).split(" ")
                            if any([split.upper() in month for split in space_split]):
                                result.update({"date": ",".join(text.strip().split(",")[8:])})
                    if "date" not in result:
                        print("%s: Nothing Detected" % (task))
                elif task == "company":
                    # has 81.83% accuracy that the first line of the reciept
                    # is the company name according to train set
                    result.update({"company": ",".join(all_txt_lines[i][0].strip().split(",")[8:])})
                else:
                    print("%s: Nothing Detected" % (task))
            else:
                # Use the prediction result of BERT
                for k, indicator in enumerate(idx[j]):
                    key_info = ",".join(all_txt_lines[i][k].strip().split(",")[8:])
                    if indicator == 1:
                        if task in result:
                            if task == "date":
                                print("Another date: %s"%key_info)
                                pass
                            elif task == "company":
                                print("Another company: %s"%key_info)
                                pass
                            else:
                                result[task] += (" " + key_info)
                        else:
                            result.update({task: key_info})
        for key in result.keys():
            print("%s: %s"%(key, result[key]))
        print("")
        start += num