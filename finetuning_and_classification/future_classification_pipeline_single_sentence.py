from transformers import pipeline
from os import walk
import csv
import re

def main():
    # Load the pre-trained text classification model pipeline
    p = pipeline("text-classification", model='./pipeline_pretrained_future_single_sentence', device=0)

    # Define the folder containing the data files to process
    data_folder = "./data/future_extraction/out/"
    already_processed_txt_file_path = "processed_single_sentence.txt"

    # Initialize a list to store filenames
    filenames = []

    # Read the list of already processed files to avoid reprocessing
    with open(already_processed_txt_file_path, 'r') as already_processed_file:
        already_processed = already_processed_file.read().splitlines()

    print(f"Files already processed: {len(already_processed)}")

    # Reopen the processed file tracking file in append mode
    with open(already_processed_txt_file_path, 'a') as already_processed_file:
        # Walk through the data folder and collect filenames
        for (dirpath, dirnames, f_name) in walk(data_folder):
            filenames.extend(f_name)
            break

        # Prepare CSV writer for saving the results
        wtr = csv.writer(open('../data/finalFutureStatements/final_single_sentence.csv', 'a'), delimiter=',', lineterminator='\n')

        # Compile regex to find sentences containing the word "future"
        regex = re.compile(r'([^.]*future[^.]*)', re.I)

        # Filter out files that have already been processed
        filenames = [x for x in filenames if x not in already_processed]
        print(f"Files to process: {len(filenames)}")

        # Process each file in the folder
        counter = 0
        for filename in filenames:
            result = []
            file_path = data_folder + filename
            
            # Open and read the content of the file
            with open(file_path) as file:
                file_contents = file.read()
            
            # Split the file contents into individual lines
            contents_split = file_contents.splitlines()
            # Filter out excessively long lines to prevent OOM errors
            contents_split = list(filter(lambda x: len(x) < 10000, contents_split))
            
            # Process each line in the file
            for line in contents_split:
                # Find all sentences containing the word "future"
                future_lines = regex.findall(line)
                for future_line in future_lines:
                    # Classify each extracted sentence
                    test = p(future_line.strip())
                    if test[0]['label'] == 'LABEL_1':  # Assuming LABEL_1 is the label for future statements
                        result.append([future_line.strip()])
            
            # Write the results to the CSV file
            wtr.writerows(result)
            print(f"Writing filename: {filename}")
            
            # Record the processed filename
            already_processed_file.write(filename + "\n")
            
            # Update the counter and print progress
            counter += 1
            print(f"[{counter}/{len(filenames)}]")

    print("End")

if __name__ == "__main__":
    main()
