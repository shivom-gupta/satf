from bertopic import BERTopic
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Load the statements from the CSV file
statements = []
with open('../data/finalFutureStatements/final_single_sentence.csv', 'r') as file:
    reader = csv.reader(file)
    statements = [row[0] for row in reader]

# Initialize the BERTopic model
topic_model = BERTopic(language='english', calculate_probabilities=True, verbose=True, low_memory=False)

# Fit the model on the statements and transform the data to extract topics
topics, probs = topic_model.fit_transform(statements)

# Print topic information
print(topic_model.get_topic_info())
print(topic_model.get_topic(3))  # Print details of a specific topic

# Visualize the most frequent topics in a bar chart
topic_model.visualize_barchart()
plt.show()
