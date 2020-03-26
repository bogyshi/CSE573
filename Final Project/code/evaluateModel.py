import numpy as np
import pickle as pk
import scipy as spatial
import matplotlib as plt
import sklearn
import pandas as pd
import requests
import time
import os
from io import BytesIO, StringIO
from zipfile import ZipFile
from tensorflow.keras import layers , activations , models , preprocessing , utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D,LSTM,Conv1D, MaxPooling1D, Embedding,RepeatVector,TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
import tensorflow as tf
import pdb
from tensorflow.keras.initializers import Constant


truncDataSize=0.1 # if set to 1, use all data available
max_length_targ = max_length_inp=20
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def translate(sentence,il,tl):
  result, sentence, attention_plot = evaluate(sentence,il,tl)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  #attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  #plot_attention(attention_plot, sentence.split(' '), result.split(' '))

def evaluate(sentence,inp_lang,targ_lang):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  #sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


class BahdanauAttention(tf.keras.layers.Layer):
    # stolen from https://www.tensorflow.org/tutorials/text/nmt_with_attention

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Encoder(tf.keras.Model):
# stolen from https://www.tensorflow.org/tutorials/text/nmt_with_attention

  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))



def convert(lang, tensor):
    # stolen from https://www.tensorflow.org/tutorials/text/nmt_with_attention
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

@tf.function
def train_step(inp, targ, enc_hidden,targ_lang):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss



def failedModel1():
    print('Training model.')

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(20*50,activation='relu'))#20*5 cause 20 words, 50 dimensional word vector per?
    model.add(RepeatVector(3))
    model.add(LSTM(20*5, return_sequences=True))
    model.add(TimeDistributed(Dense(20)))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_val, y_val))


def useBigStuf():
    zip_file = ZipFile('./glove.6B.zip')
    files = zip_file.namelist()
    redFile = files[0]
    #df = pd.read_csv(zip_file.open(redFile),sep=" ",header=None,delimiter='\n')
    rawdata = str(zip_file.read(redFile),'utf-8').splitlines()
    data=[]
    for f in rawdata:
        data.append(f.split())
    embeddings = pd.DataFrame(data)
    print(embeddings.head())
    print(embeddings.dtypes)
    uniqueWords = embeddings[0].values
    res = set(vocab.keys()).intersection(set(uniqueWords))
    print(len(res))# there are about 4000 out of 14000 tokens missing from the bigger corpus(lkely misspellings and slang)
    print(len(vocab.keys()))

with open('./vocab.pk','rb') as f:
    vocab = pk.load(f)
with open('./conversation.pk','rb') as f:
    dialogue = pk.load(f)
    newDialogue = []
    for m in dialogue:
        newDialogue.append('<start> ' + m.strip() + ' <end>')
dialogue=newDialogue
'''with open('./vectors.txt','r') as f:
    myVecs = pd.read_csv(f,sep=' ',header=None)'''
embeddings_index = {}
with open('./vectors.txt','r') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
'''
with open('./convo.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(dialogue))
    '''
#lets load our pretrained embeddings and tokenize and start doing our matching
MAX_NUM_WORDS=1000
MAX_SEQUENCE_LENGTH=10
EMBEDDING_DIM=50
VALIDATION_SPLIT=0.2
labels=[0,1,2]
'''counter = 0
for x in dialogue:
    labels.append(counter)VAL
    counter+=1'''
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='') # the max num words works, but it still keeps track of everything
# this will only change later when it actually fits wiht its matrix reps
tokenizer.fit_on_texts(dialogue)
sequences = tokenizer.texts_to_sequences(dialogue)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#pdb.set_trace()

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = data[1:,:]
data = data[:-1,:]
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0]*truncDataSize).astype(int)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

#limitedTokenizer = Tokenizer().fit_on_texts(dialogue)


x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

checkpoint_dir = './training_checkpoints'

BUFFER_SIZE = len(x_train)
BATCH_SIZE = 64
steps_per_epoch = len(x_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(tokenizer.word_index)+1
vocab_tar_size = len(tokenizer.word_index)+1

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
for x in tokenizer.sequences_to_texts(x_val):
    #print()
    translate(x,tokenizer,tokenizer)
