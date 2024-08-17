# Future Statements

This repository is designed to extract, classify, and analyze future statements from WARC files, providing insights into how people perceive and predict the future. It builds upon the [WARC-DL repository](https://github.com/webis-de/WARC-DL/) for data extraction from WARC files, with additional methods for fine-tuning models and conducting sentiment analysis, subjectivity assessment, and topic modeling.

## Project Overview

In today’s era of Big Data and Machine Learning, text classification plays a vital role in deriving insights from diverse data sources. This project focuses on creating a database of future statements extracted from web archives, classifying them based on their content, and analyzing the key themes and sentiments. The primary goal is to understand how people perceive the future, the common themes in future predictions, and the sentiment associated with these predictions.

## Methodology

### 1. **Data Extraction**
   - **WARC File Parsing:** We utilize the `Fastwarc` library to parse WARC files and extract lines containing the phrase "in the future".
   - **Regular Expressions:** Regular expressions are used to identify and extract potential future statements from a large corpus.

### 2. **Preprocessing**
   - **Data Cleaning:** The extracted sentences undergo preprocessing steps such as removing unicode characters to ensure clean and consistent data for analysis.
   - **Manual Labeling:** Extracted sentences are manually labeled to determine if they represent valid future predictions.

### 3. **Model Fine-Tuning**
   - **Tokenization:** Sentences are tokenized using `DistilBertTokenizer` for processing by the `DistilBERT` model.
   - **Fine-Tuning:** The labeled data is used to fine-tune a pre-trained transformer model, `DistilBERT`, optimizing it for sequence classification.
   - **Evaluation:** The model’s performance is evaluated using metrics such as accuracy, precision, and recall.

### 4. **Classification and Analysis**
   - **Future Statement Classification:** The fine-tuned model classifies sentences as valid or invalid future statements. Approximately 49,035 sentences were classified as future statements after processing.
   - **Sentiment and Subjectivity Analysis:** Sentiment analysis using `TextBlob` assesses whether future predictions are positive, negative, or neutral. Subjectivity analysis determines whether these predictions are factual or opinionated.
   - **Topic Modeling:** `BERTopic` is used to identify the main topics discussed in these future predictions, revealing trends such as water, climate, and elections.

## Results

- **Model Performance:** The model achieved an accuracy of 0.78 on the validation dataset, with varying performance on different subsets.
- **Sentiment Analysis:** The average sentiment of future predictions was found to be neutral, with a mean sentiment score of 0.07.
- **Subjectivity:** The predictions tended to be more factual, with a mean subjectivity score of 0.33.
- **Common Topics:** Key topics included climate, elections, and technology, reflecting concerns about future developments in these areas.

## Usage

To run the extraction, classification, and topic modeling processes:

```bash
# Extract and classify future statements
python run_extraction.py --input path/to/warc/files --output path/to/save/extracted/statements

# Fine-tune the model
python finetuning_and_classification/finetuning.py

# Classify and analyze the data
python finetuning_and_classification/classification.py

# Perform topic modeling
python topic_modeling.py
```

## Future Work

Future improvements could include enhancing the extraction process by incorporating additional keywords, refining the model with a more balanced dataset, and expanding the topic modeling to cover more diverse themes.

## Credits

This project was developed by Nikolai Kortenbruck, Shivom Gupta, and Alexander Pavlovski, with contributions in conceptualization, software development, and data analysis.

## Contact

For more information, please refer to the [Future_statements_report.pdf](Future_statements_report.pdf) or contact the project contributors via GitHub.
