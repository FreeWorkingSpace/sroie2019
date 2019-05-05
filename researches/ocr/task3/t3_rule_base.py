from researches.ocr.task3.t3_util import *


def is_date(text):
  month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
  ind_text = text.replace("[", "").replace("]", "").replace(":", "")\
    .replace(" ", "")
  extra_indicator = len(ind_text.split("/")) == 3 or \
                    len(ind_text.split("-")) == 3

  if extra_indicator:
    if len(ind_text.split("/")) == 3:
      split = ind_text.split("/")
    if len(ind_text.split("-")) == 3:
      split = ind_text.split("-")
    if is_number(split[0][-2:]) and is_number(split[-1][:2]):
      pass
    elif split[1].upper() in month:
      pass
    else:
      return False
    if len(ind_text.split("-")) > 1:
      splited = ind_text.split("-")
      # make sure telphone number xx-xxxx-xxxx will be filtered out
      if all([num.isdigit() for num in splited[1]]) and len(splited[1]) == 4:
        return False
  return extra_indicator


def correct_date(text, split_mode=None):
  pass


def get_height_of_line(text):
  coord = text.strip().split(",")[:8]
  coord = [int(c) for c in coord]
  #x1, x2 = min(coord[::2]), max(coord[::2])
  y1, y2 = min(coord[1::2]), max(coord[1::2])
  return y2 - y1


if __name__ == "__main__":
  string = "DATE: 22-03-2018 04:01:20 PM"
  print(is_date(string))