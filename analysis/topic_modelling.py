from bertopic import BERTopic
import pandas as pd
import csv
import matplotlib.pyplot as plt
statements = []
with open('../data/finalFutureStatements/final_single_sentence.csv', 'r') as file:
    reader = csv.reader(file)
    statements = [row[0] for row in reader]

topic_model = BERTopic(language = 'english', calculate_probabilities= True, verbose = True, low_memory=False)
topic, probs = topic_model.fit_transform(statements)
print(topic_model.get_topic_info())
print(topic_model.get_topic(3))
topic_model.visualize_barchart()
plt.show()