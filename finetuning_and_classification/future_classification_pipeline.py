from transformers import pipeline
from os import walk
import csv

def main():
    # Load the pre-trained text classification model pipeline
    p = pipeline("text-classification", model='./pipeline_pretrained_future', device=0)

    # Define the data folder and file tracking processed files
    data_folder = "./data/future_extraction/out/"
    already_processed_txt_file_path = "processed.txt"

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
        with open('../data/finalFutureStatements/final.csv', 'a', newline='') as csvfile:
            wtr = csv.writer(csvfile, delimiter=',', lineterminator='\n')

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
                    # Classify each line
                    test = p(line)
                    if test[0]['label'] == 'LABEL_1':  # Assuming LABEL_1 is the target label
                        result.append([line])

                # Write the results to the CSV file
                wtr.writerows(result)

                # Record the processed filename
                already_processed_file.write(filename + "\n")

                # Update the counter and print progress
                counter += 1
                print(f"[{counter}/{len(filenames)}] Processed: {filename}")

    print("End")

if __name__ == "__main__":
    main()
