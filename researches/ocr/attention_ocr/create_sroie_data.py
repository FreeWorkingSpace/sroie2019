import os, glob, time
import cv2


def create_dataset(root_path, new_dataset_path, img_operation=True):
    char_statistics = {}
    word_set = set([])
    f = open(os.path.join(new_dataset_path, "label.txt"), "w")
    txt_list = sorted(glob.glob(root_path + "/*.txt"))
    img_num = 0
    for i, txt_file in enumerate(txt_list):
        start = time.time()
        name = txt_file[txt_file.rfind("/") + 1 : -4]
        img_path = os.path.join(root_path, name + ".jpg")
        if not os.path.exists(img_path):
            print("image: %s does not exists"%(name + ".jpg"))
            continue
        if img_operation:
            img = cv2.imread(img_path)
        else:
            img = None
        with open(txt_file, "r", encoding="utf-8") as txt_lines:
            for j, line in enumerate(txt_lines):
                line_element = line.split(",")
                text_label = ",".join(line_element[8:])
                words = text_label.strip().split(" ")
                for word in words:
                    word_set.add(word)
                for char in text_label:
                    if char in char_statistics:
                        char_statistics[char] += 1
                    else:
                        char_statistics[char] = 1
                coord = [int(x) for x in line_element[:8]]
                x1, x2 = min(coord[::2]), max(coord[::2])
                y1, y2 = min(coord[1::2]), max(coord[1::2])
                if img_operation:
                    img_with_txt = img[y1: y2, x1: x2, :]
                    cv2.imwrite(os.path.join(new_dataset_path, str(img_num) + ".jpg"), img_with_txt)
                    write_line = str(img_num) + ".jpg:%s"%(text_label)
                    f.write(write_line)
                img_num += 1
        print("%d/%d completed, cost %.2f seconds"%(i, len(txt_list), time.time() - start))
    f.close()
    return word_set, char_statistics


if __name__ == "__main__":
    root_path = os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019")
    new_dataset_path = os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019_OCR_2")
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
    word_set, char_statistics = create_dataset(root_path, new_dataset_path, img_operation=False)
    print("|   Character   |    Number    |")
    for key in sorted(char_statistics.keys()):
        print("|  %s  |  %s |"%(str(key).ljust(6), str(char_statistics[key]).ljust(6)))
    char_list_upper = [key for key in sorted(char_statistics.keys())]
    char_list_lower = [char.lower() for char in char_list_upper]
    #print(char_list)
    #print(char_list_lower)
    char_list =  set(char_list_lower).union(char_list_upper)
    label_dict = ", ".join(["'%s': %s"%(key, i) for i, key in enumerate(sorted(char_list))])
    print("label_dict = {" + label_dict + "}")
    write_to_dictionary = False
    if write_to_dictionary:
        word_path = os.path.expanduser("~/Documents/TextRecognitionDataGenerator/"
                                       "TextRecognitionDataGenerator/dicts")
        word_dict = open(os.path.join(word_path, "sroie_words.txt"), "w")
        for item in word_set:
            word_dict.write(item + "\n")
        word_dict.close()