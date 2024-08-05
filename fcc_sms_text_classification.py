# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds


print(tf.__version__)

# get data files
url = 'https://cdn.freecodecamp.org/project-data/sms/train-data.tsv'
url = 'https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv'

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_data = pd.read_csv('train-data.tsv', sep='\t', names=['type', 'text'])
test_data = pd.read_csv('valid-data.tsv', sep='\t', names=['type', 'text'])

train_labels = train_data.pop('type')
test_labels = test_data.pop('type')

vocab = {'ham': 0, 'spam': 1}

train_labels = train_labels.map(vocab)
test_labels = test_labels.map(vocab)

max_features = 10000
sequence_length = 250
    
vectorize_layer = layers.TextVectorization(standardize='lower',
                                               max_tokens=max_features,
                                               output_mode='int',
                                               output_sequence_length=sequence_length)
vectorize_layer.adapt(train_data['text'])
    
embedding_dim = 16
    
model = tf.keras.Sequential([
    vectorize_layer,
    layers.Embedding(max_features, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(2)])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam', metrics=['accuracy'])

epochs = 50
model.fit(train_data.astype(object), train_labels, epochs=epochs, verbose=0)
print(model.evaluate(test_data.astype(object), test_labels))
    
# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    output = model.predict(x=np.array([pred_text]).astype(object),verbose=0)
    print(output)
    if output[0][0] > output[0][1]:
        prediction = [output[0][0], 'ham']
    elif output[0][1] > output[0][0]:
        prediction = [output[0][1], 'spam']
    

    return (prediction)

#pred_text = "how are you doing today?"

#prediction = predict_message(pred_text)
#print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    print(prediction)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
