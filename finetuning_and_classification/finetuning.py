import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DataCollatorWithPadding, pipeline
from sklearn.metrics import classification_report

dataset = load_dataset("csv", data_files=["./data_labeled/2050.csv", "./data_labeled/train_ds1.csv"])

train_ds, testing_validation_data = train_test_split(dataset['train'], test_size=0.2, random_state=25)
test_ds, val_ds = train_test_split(Dataset.from_dict(testing_validation_data), test_size=0.5, random_state=123)

model_checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


pandas = Dataset.from_dict(train_ds)
pandas.set_format(type="pandas")
pandas_df = pandas[:]
print(pandas_df['label'].value_counts())


tokenized_train_ds = Dataset.from_dict(train_ds).map(tokenize, batched=True, batch_size=None)
tokenized_val_ds = Dataset.from_dict(val_ds).map(tokenize, batched=True, batch_size=None)
tokenized_test_ds = Dataset.from_dict(test_ds).map(tokenize, batched=True, batch_size=None)
print(tokenized_val_ds)
BATCH_SIZE = 1
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_train_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                  label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)

tf_val_dataset = tokenized_val_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                  label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)

tf_test_dataset = tokenized_test_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                  label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)
print("Tensorflow dataset:")
print(tf_train_dataset)


tensorboard_callback = TensorBoard(log_dir="../model/logs")
model_checkpoint_callback = ModelCheckpoint(
    filepath="../modelCheckpoint/",
    save_weights_only=False,
)
callbacks = [tensorboard_callback, model_checkpoint_callback]


model = TFDistilBertForSequenceClassification.from_pretrained(model_checkpoint,
                                                              num_labels=2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.metrics.SparseCategoricalAccuracy()
)

history = model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=5, callbacks=callbacks)
_, accuracy = model.evaluate(tf_test_dataset)

p = pipeline("text-classification", model=model, device=-1, tokenizer=tokenizer)
p.save_pretrained('./pipeline_pretrained_future')

exit()
