# main() function
import csv
import re
from preprocessing.regex_module import *
from preprocessing.preprocessing_module import *
import pandas as pd
from preprocessing.file_reader_module import *


# Defining main function
def main():
    with open('2050.csv', mode='r', errors='replace') as file:
        # reading the CSV file
        csvFile = csv.reader(file, delimiter=",")
        data = "".join(row[0] for row in csvFile if isinstance(row[0], str))
        data = pick_only_key_sentence(data, "future")

        result = []

        pd.set_option('display.max_rows', None)
        for string in data:
            #string = split_by_char(string)
            #string = remove_stopwords(string)
            string = remove_unicode(string)
            #string = split_by_char(string)
            result.append(string)

        df = pd.DataFrame(result, columns=["text"])
        df.loc[:, ["label"]] = 1
        df.to_csv('data_labeled/2050.csv', index=False)


if __name__ == "__main__":
    main()
