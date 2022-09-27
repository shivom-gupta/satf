from transformers import pipeline
from os import walk
import csv


p = pipeline("text-classification", model='./pipeline_pretrained_future', device=0)
data_folder = "./data/future_extraction/out/"
already_processed_txt_file_path = "processed.txt"
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

wtr = csv.writer(open ('../data/finalFutureStatements/final.csv', 'a'), delimiter=',', lineterminator='\n')

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
        test = p(line)
        if test[0]['label'] == 'LABEL_1':
            result.append([line])
    wtr.writerows(result)
    #for r in result:
        #wtr.writerow([r])
    already_processed_file.writelines(filename + "\n")
    counter += 1
    print("[" + str(counter) + "/" + str(len(filenames)) + "]")
already_processed_file.close()
print("End")
