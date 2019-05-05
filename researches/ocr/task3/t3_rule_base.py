from researches.ocr.task3.t3_util import *

month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

def is_date(text):
  ind_text = text.replace("[", "").replace("]", "").replace(":", "")\
    .replace(" ", "")
  extra_indicator = len(ind_text.split("/")) == 3 or \
                    len(ind_text.split("-")) == 3

  if extra_indicator:
    if len(ind_text.split("/")) == 3:
      split = ind_text.split("/")
    if len(ind_text.split("-")) == 3:
      split = ind_text.split("-")
    if any([char in [",", "."] for char in split[1]]):
      return False
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


def correct_date(text):

  def get_year(_split):
    if len(_split[2]) >= 4 and is_number(_split[2][:4]):
      year = _split[2][:4]
    elif len(_split[2]) < 4 and is_number(_split[2][:2]):
      year = _split[2][:2]
    else:
      if len(_split[2].split(" ")) > 1:
        if is_number(_split[2].split(" ")[0]):
          year = _split[2].split(" ")[0]
        else:
          year = ""
      else:
        year = ""
    return year

  if len(text.split("/")) == 3:
    split = text.split("/")
    year = get_year(split)
    return "/".join([split[0][-2:], split[1], year])
  elif len(text.split("-")) == 3:
    split = text.split("-")
    year = get_year(split)
    return "-".join([split[0][-2:], split[1], year])
  elif len(text.split(".")) >= 3:
      split = text.split(".")
      year = get_year(split)
      return ".".join([split[0][-2:], split[1], year])

  elif len(text.split(" ")) >= 3:
    split = text.split(" ")
    for i, string in enumerate(split):
      if string.upper() in month:
        return " ".join([split[i - 1], split[i], split[i + 1]])
    return text
  else:
    print("Invalid date format: %s" % text)
    return text


def correct_company(text):
  # Eliminate the (xxxxxxxxx) at the end
  text = text.replace(",", "").replace("  ", "")
  text = text.strip(" ")
  if len(text.split("(")) > 1:
    latter = text.split("(")[-1]
    if len(latter.split(" ")) == 1:
      # means (xxxxxxxxx) at the end
      return text.split("(")[0].strip(" ")
    else:
      return text
  else:
    return text


def get_height_of_line(text):
  coord = text.strip().split(",")[:8]
  coord = [int(c) for c in coord]
  #x1, x2 = min(coord[::2]), max(coord[::2])
  y1, y2 = min(coord[1::2]), max(coord[1::2])
  return y2 - y1


if __name__ == "__main__":
  print(is_date("16-667 0982 , 016-333"))