import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DataCollatorWithPadding, pipeline

# Define paths for dataset, logs, checkpoints, and saved models
data_set_path = ["./finetuning_and_classification/data_labeled/single_sentence_statements.csv"]
tensorboard_logdir_path = "./model/logs/single_sentence"
model_checkpoint_callback_path = "./modelCheckpoint_single_sentence/"
pipeline_save_path = './finetuning_and_classification/pipeline_pretrained_future_single_sentence'

# Load the dataset from the CSV file
dataset = load_dataset("csv", data_files=data_set_path)

# Split the dataset into training and testing/validation sets
train_ds, testing_validation_data = train_test_split(dataset['train'], test_size=0.2, random_state=25)
test_ds, val_ds = train_test_split(Dataset.from_dict(testing_validation_data), test_size=0.5, random_state=123)

# Load raw test dataset for later evaluation
raw_test_dataset = load_dataset("csv", data_files=["./finetuning_and_classification/data_labeled/test_single_raw.csv"])

# Load the pre-trained DistilBERT tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)

# Function to tokenize input text data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# Convert training data to pandas DataFrame for analysis and print label distribution
pandas = Dataset.from_dict(train_ds)
pandas.set_format(type="pandas")
pandas_df = pandas[:]
print(pandas_df['label'].value_counts())

# Print raw test dataset details
print(raw_test_dataset)

# Tokenize the datasets (train, validation, test, and raw test)
tokenized_train_ds = Dataset.from_dict(train_ds).map(tokenize, batched=True, batch_size=None)
tokenized_val_ds = Dataset.from_dict(val_ds).map(tokenize, batched=True, batch_size=None)
tokenized_test_ds = Dataset.from_dict(test_ds).map(tokenize, batched=True, batch_size=None)
tokenized_raw_test_dataset = raw_test_dataset['train'].map(tokenize, batched=True, batch_size=None)

# Print tokenized validation dataset for verification
print(tokenized_val_ds)

# Set batch size for TensorFlow datasets
BATCH_SIZE = 1

# Data collator for dynamic padding in batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# Convert tokenized datasets to TensorFlow datasets
tf_train_dataset = tokenized_train_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                    label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)

tf_val_dataset = tokenized_val_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)

tf_test_dataset = tokenized_test_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                  label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)

tf_raw_test_dataset = tokenized_raw_test_dataset.to_tf_dataset(columns=tokenizer.model_input_names,
                                                               label_cols=['label'], shuffle=False,
                                                               batch_size=BATCH_SIZE)

# Print TensorFlow training dataset details for verification
print("Tensorflow dataset:")
print(tf_train_dataset)

# Set up callbacks for TensorBoard logging and model checkpoints
tensorboard_callback = TensorBoard(log_dir=tensorboard_logdir_path)
model_checkpoint_callback = ModelCheckpoint(
    filepath=model_checkpoint_callback_path,
    save_weights_only=False,
)
callbacks = [tensorboard_callback, model_checkpoint_callback]

# Load pre-trained DistilBERT model for sequence classification
model = TFDistilBertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Compile the model with optimizer, loss function, and accuracy metric
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.metrics.SparseCategoricalAccuracy()
              )

# Train the model on the training dataset and validate on the validation dataset
history = model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=20, callbacks=callbacks)

# Evaluate the model on the test dataset
_, accuracy = model.evaluate(tf_test_dataset)

# Print classification report for the test dataset
print("Split test:")
y_pred = model.predict(tf_test_dataset, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred.logits, axis=1)
print(classification_report(tokenized_test_ds['label'], y_pred_bool))

# Print classification report for the raw test dataset
print("Raw test:")
y_pred = model.predict(tf_raw_test_dataset, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred.logits, axis=1)
print(classification_report(tokenized_raw_test_dataset['label'], y_pred_bool))

# Save the trained model pipeline for future use
p = pipeline("text-classification", model=model, device=-1, tokenizer=tokenizer)
p.save_pretrained(pipeline_save_path)

# Exit the script
exit()
