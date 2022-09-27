import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DataCollatorWithPadding, pipeline

# Paths relative to workdir:
data_set_path = ["./finetuning_and_classification/data_labeled/single_sentence_statements.csv"]
tensorboard_logdir_path = "./model/logs/single_sentence"
model_checkpoint_callback_path = "./modelCheckpoint_single_sentence/"
pipeline_save_path = './finetuning_and_classification/pipeline_pretrained_future_single_sentence'

dataset = load_dataset("csv", data_files=data_set_path)
train_ds, testing_validation_data = train_test_split(dataset['train'], test_size=0.2, random_state=25)
test_ds, val_ds = train_test_split(Dataset.from_dict(testing_validation_data), test_size=0.5, random_state=123)
raw_test_dataset = load_dataset("csv", data_files=["./finetuning_and_classification/data_labeled/test_single_raw.csv"])
model_checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


pandas = Dataset.from_dict(train_ds)
pandas.set_format(type="pandas")
pandas_df = pandas[:]
print(pandas_df['label'].value_counts())
print(raw_test_dataset)
tokenized_train_ds = Dataset.from_dict(train_ds).map(tokenize, batched=True, batch_size=None)
tokenized_val_ds = Dataset.from_dict(val_ds).map(tokenize, batched=True, batch_size=None)
tokenized_test_ds = Dataset.from_dict(test_ds).map(tokenize, batched=True, batch_size=None)
tokenized_raw_test_dataset = raw_test_dataset['train'].map(tokenize, batched=True, batch_size=None)
print(tokenized_val_ds)
BATCH_SIZE = 1
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_train_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                    label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)

tf_val_dataset = tokenized_val_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)

tf_test_dataset = tokenized_test_ds.to_tf_dataset(columns=tokenizer.model_input_names,
                                                  label_cols=['label'], shuffle=False, batch_size=BATCH_SIZE)
tf_raw_test_dataset = tokenized_raw_test_dataset.to_tf_dataset(columns=tokenizer.model_input_names,
                                                               label_cols=['label'], shuffle=False,
                                                               batch_size=BATCH_SIZE)
print("Tensorflow dataset:")
print(tf_train_dataset)

tensorboard_callback = TensorBoard(log_dir=tensorboard_logdir_path)
model_checkpoint_callback = ModelCheckpoint(
    filepath=model_checkpoint_callback_path,
    save_weights_only=False,
)
callbacks = [tensorboard_callback, model_checkpoint_callback]

model = TFDistilBertForSequenceClassification.from_pretrained(model_checkpoint,
                                                              num_labels=2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.metrics.SparseCategoricalAccuracy()
              )

history = model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=20, callbacks=callbacks)
_, accuracy = model.evaluate(tf_test_dataset)

print("Split test:")
y_pred = model.predict(tf_test_dataset, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred.logits, axis=1)
print(classification_report(tokenized_test_ds['label'], y_pred_bool))

print("Raw test:")
y_pred = model.predict(tf_raw_test_dataset, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred.logits, axis=1)
print(classification_report(tokenized_raw_test_dataset['label'], y_pred_bool))

p = pipeline("text-classification", model=model, device=-1, tokenizer=tokenizer)
p.save_pretrained(pipeline_save_path)

exit()
