# Future Statements



## Please refer to Future_statements_report.pdf for detailed information.




This repository is ment to extract and classify future statements from WARC files.
It is based on the  [WARC-DL repository](https://github.com/webis-de/WARC-DL/) for extracting data from WARC-files. 
For more information on how to run the basic WARC extraction process, refer to the repository. Here we have added
our own regular expression based extraction in the [examples section](examples/future_extraction). Which extracts 
single lines that contain the regular expression 
```regexp
"in the future|In the future"
```
The run script of that example is modified so it takes an argument and splices the files array of the WARC-repository 
and executes on a batch of only 1000 entries. This was done due to encountering errors in the process at some point.

## Data
### Extracted
Some data stored from this process can be found in the [data folder](data/future_extraction/out).

### Labeled
Some of this data was labeled manually as a valid future statement (1) or a false future statement (0). In addition to 
manually labeled data, some data was scraped from websites containing future statements. The labeled data is located in 
the [finetuning_and_classification/data_labeled folder](finetuning_and_classification/data_labeled). While the 
web-scraping code can be found in [webScraping](webScraping).

## Finetuning a pre - trained model
The finetuning of a pretrained model for whole lines is defined in the 
[finetuning_and_classification/finetuning.py file](finetuning_and_classification/finetuning.py) and for single sentences
containing the keyword 'future' is defined in the
[finetuning_and_classification/finetuning_single_sentence.py file](finetuning_and_classification/finetuning_single_sentence.py)
file. A pipeline of the model is saved upon finetuning.

## Classification of candidates
Like the fine tuning process, the classification process is also defined for both whole extracted lines and only the 
sentences containing the keyword future. They can be found in the same folder 
[finetuning_and_classification](finetuning_and_classification). The finetuned pipeline is applied to the data extracted
in the data folder and candidates classified as valid future statements are stored in the 
[finalFutureStatements subfolder](data/finalFutureStatements).


## Analysis
The analysis applied to the candidates classified as valid future statement can be found in the 
[analysis](analysis) folder.

## Extracted Dataset
The final, annotated and extracted data set is located in 
[data/finalFutureStatements/annotated.csv](data/finalFutureStatements/annotated.csv) and it's 
[Dataset card](data/finalFutureStatements/README.md) in the same folder