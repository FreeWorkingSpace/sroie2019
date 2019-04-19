import os, glob, time
import cv2


def create_dataset(root_path, new_dataset_path):
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
        img = cv2.imread(img_path)
        with open(txt_file, "r", encoding="utf-8") as txt_lines:
            for j, line in enumerate(txt_lines):
                line_element = line.split(",")
                text_label = ",".join(line_element[8:])
                coord = [int(x) for x in line_element[:8]]
                x1, x2 = min(coord[::2]), max(coord[::2])
                y1, y2 = min(coord[1::2]), max(coord[1::2])
                img_with_txt = img[y1: y2, x1: x2, :]
                cv2.imwrite(os.path.join(new_dataset_path, str(img_num) + ".jpg"), img_with_txt)
                write_line = str(img_num) + ".jpg:%s"%(text_label)
                f.write(write_line)
                img_num += 1
        print("%d/%d completed, cost %.2f seconds"%(i, len(txt_list), time.time() - start))
    f.close()


if __name__ == "__main__":
    root_path = os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019")
    new_dataset_path = os.path.expanduser("~/Pictures/dataset/ocr/SROIE2019_OCR_2")
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
    create_dataset(root_path, new_dataset_path)



