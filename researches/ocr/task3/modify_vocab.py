from os.path import *
import glob, json

bert_root = expanduser("~/Documents/bert")
bert_model = "uncased_L-12_H-768_A-12"

task_1_2_text = expanduser("~/Pictures/dataset/ocr/SROIE2019")
task_1_2_label_root = expanduser("~/Downloads/task_1_2_label")
task_3_text_root = expanduser("~/Downloads/task_3_label")


# get top 1000 frequent words from task1_2
text_files = glob.glob(task_1_2_text + "/*.txt")
word_freq = {}
print("Enumerating through %d files in %s"%(len(text_files), task_1_2_text))
for i, text_file in enumerate(text_files):
    with open(text_file, "r", encoding="utf-8") as txt_lines:
        for j, line in enumerate(txt_lines):
            line_element = line.split(",")
            text_label = ",".join(line_element[8:])
            words = text_label.strip().split(" ")
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq.update({word: 1})
print("word_freq for task_1_2_text has %d keys."%len(word_freq.keys()))

# get top 1000 frequent words from task1_2
text_files = glob.glob(task_1_2_label_root + "/*.txt")
company_freq = {}
address_freq = {}
print("Enumerating through %d files in %s"%(len(text_files), task_1_2_label_root))
for i, text_file in enumerate(text_files):
    with open(text_file, "r") as file:
        data = json.load(file)
        words = data["company"].strip().split(" ")
        for word in words:
            if word in company_freq:
                company_freq[word] += 1
            else:
                company_freq.update({word: 1})
        try:
            words = data["address"].strip().split(" ")
        except KeyError:
            print(data)
            continue
        for word in words:
            if word in address_freq:
                address_freq[word] += 1
            else:
                address_freq.update({word: 1})
frequency_com = sorted(company_freq.items(), key = lambda kv:(kv[1], kv[0]))
print("company_freq for task_1_2_text has %d keys."%len(company_freq.keys()))
print("address_freq for task_1_2_text has %d keys."%len(address_freq.keys()))