import csv
import pandas as pd
from preprocessing.regex_module import *
from preprocessing.preprocessing_module import *
from preprocessing.file_reader_module import *

# Defining the main function
def main():
    # Open and read the CSV file
    with open('2050.csv', mode='r', errors='replace') as file:
        csvFile = csv.reader(file, delimiter=",")
        
        # Extract and concatenate the relevant data
        data = "".join(row[0] for row in csvFile if isinstance(row[0], str))
        
        # Filter sentences containing the word "future"
        data = pick_only_key_sentence(data, "future")

        result = []

        # Set pandas options to display all rows
        pd.set_option('display.max_rows', None)
        
        # Process each string in the data
        for string in data:
            string = remove_unicode(string)  # Remove unicode characters
            result.append(string)

        # Convert the result list to a DataFrame
        df = pd.DataFrame(result, columns=["text"])
        df.loc[:, ["label"]] = 1  # Add a label column with value 1

        # Save the processed DataFrame to a new CSV file
        df.to_csv('data_labeled/2050.csv', index=False)

# Execute the main function
if __name__ == "__main__":
    main()
