from utils.DatasetBuilder import DatasetBuilder
import os
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

path = os.path.abspath('./data/data.json')

dataset_builder = DatasetBuilder()
dataset_builder.parse_json(path)

input_ids, token_type_ids, attention_masks, categories = dataset_builder.create_dataset()


def train_dataset_generator():
    for input_id, token_type_id, attention_mask, category in zip(input_ids[:5000], token_type_ids[:5000], attention_masks[:5000],
                                                                 categories[:5000]):

        yield ({'input_ids': input_id,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_id},
               np.argmax(category))


train_dataset = tf.data.Dataset.from_generator(train_dataset_generator,
                                               ({'input_ids': tf.int32,
                                                 'attention_mask': tf.int32,
                                                 'token_type_ids': tf.int32},
                                                tf.int32),
                                               ({'input_ids': tf.TensorShape([50]),
                                                 'attention_mask': tf.TensorShape([50]),
                                                 'token_type_ids': tf.TensorShape([50])},
                                                tf.TensorShape([])))

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())

train_dataset = train_dataset.shuffle(30000).batch(32)


# Train and evaluate model
model.fit(train_dataset, epochs=5)
