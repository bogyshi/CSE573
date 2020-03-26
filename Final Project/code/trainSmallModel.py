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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

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
    labels.append(counter)
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

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print ("Input Language; index to word mapping")
convert(tokenizer, x_train[0])
print ("Target Language; index to word mapping")
convert(tokenizer, y_train[0])

BUFFER_SIZE = len(x_train)
BATCH_SIZE = 64
steps_per_epoch = len(x_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(tokenizer.word_index)+1
vocab_tar_size = len(tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden,tokenizer)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    #print("prog") its working, but taking quite some time
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


print('DONEZO')

'''
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu',data_format = 'channels_first')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu',data_format = 'channels_first')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu',data_format = 'channels_first')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(MAX_SEQUENCE_LENGTH)(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))

#print(uniqueWords)

'''
