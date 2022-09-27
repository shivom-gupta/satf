from transformers import pipeline
from os import walk
import csv
import re


p = pipeline("text-classification", model='./pipeline_pretrained_future_single_sentence', device=0)
data_folder = "./data/future_extraction/out/"
already_processed_txt_file_path = "processed_single_sentence.txt"
filenames = []

already_processed_file = open(already_processed_txt_file_path, 'r')
already_processed_file_content = already_processed_file.read()
already_processed = already_processed_file_content.splitlines()
print("Files already processed:" + str(len(already_processed)))
# reopen it in append mode
already_processed_file = open(already_processed_txt_file_path, 'a')

for (dirpath, dirnames, f_name) in walk(data_folder):
    filenames.extend(f_name)
    break

wtr = csv.writer(open ('../data/finalFutureStatements/final_single_sentence.csv', 'a'), delimiter=',', lineterminator='\n')
regex = re.compile(r'([^.]*future[^.]*)', re.I)


filenames = [x for x in filenames if x not in already_processed]
print("Files to process:" + str(len(filenames)))
counter = 0
for filename in filenames:
    result = []
    file = open(data_folder + filename)
    file_contents = file.read()
    contents_split = file_contents.splitlines()
    # prevent OOM errors for large data
    contents_split = list(filter(lambda x: len(x) < 10000, contents_split))
    for line in contents_split:
        future_lines = regex.findall(line)
        for future_line in future_lines:
            test = p(future_line.strip())
            if test[0]['label'] == 'LABEL_1':
                result.append([future_line.strip()])
    wtr.writerows(result)
    print("writing filename " + filename)
    already_processed_file.writelines(filename + "\n")
    already_processed_file.close()
    already_processed_file = open(already_processed_txt_file_path, 'a')
    counter += 1
    print("[" + str(counter) + "/" + str(len(filenames)) + "]")
already_processed_file.close()
print("End")
