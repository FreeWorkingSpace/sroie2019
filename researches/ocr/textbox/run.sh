cd ~/Documents/sroie2019/researches/ocr/textbox/

# Allow small boxes inside a large box to be matched
python3 textbox.py -cdb -en 105 -bpg 4 -lt 4 -csw -cswc 0.4 -mpf 768 -mp ft_001 -azp 0.5 -azlb 1.2 -azhb 1.8
python3 textbox.py -cdb -en 105 -bpg 4 -lt 4 -csw -cswc 0.4 -mpf ft_001 -mp ft_001 -azp 0.3 -azlb 1.4 -azhb 1.6
python3 textbox.py -cdb -en 105 -bpg 4 -lt 4 -csw -cswc 0.4 -mpf ft_001 -mp ft_001 -azp 0.3 -azlb 1.0 -azhb 1.5

# Allow small boxes inside a large box to be matched, but without zoom in augmentation
python3 textbox.py -cdb -en 210 -bpg 4 -lt 4 -csw -cswc 0.4 -mpf 768 -mp ft_0012 -azp 0.0

# Does not allow small boxes inside a large box to be matched
python3 textbox.py -cdb -en 105 -bpg 4 -lt 4 -mpf 768 -mp ft_003 -azp 0.5 -azlb 1.2 -azhb 1.8
python3 textbox.py -cdb -en 105 -bpg 4 -lt 4 -mpf ft_003 -mp ft_003 -azp 0.3 -azlb 1.4 -azhb 1.6
python3 textbox.py -cdb -en 105 -bpg 4 -lt 4 -mpf ft_003 -mp ft_003 -azp 0.3 -azlb 1.0 -azhb 1.5
