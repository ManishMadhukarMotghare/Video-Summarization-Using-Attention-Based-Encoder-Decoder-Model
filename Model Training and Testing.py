import numpy as np

def knapsack(s, w, c): #shot, weights, capacity


    shot = len(s) + 1 #number of shots
    cap = c + 1 #capacity threshold

    #matching the modified size by adding 0 at 0th index
    s = np.r_[[0], s] #adding 0 at 0th index (concatinating)
    w = np.r_[[0], w] #adding 0 at 0th index (concatinating)

    #Creating and Filling Dynamic Programming Table with zeros in shot x cap dimensions
    dp = [] #creating empty list or table
    for j in range(shot):
        dp.append([]) #s+1 rows
        for i in range(cap):
            dp[j].append(0) #c+1 columns

    #Started filling values from (2nd row, 2nd column) till (shot X cap) and keeping the values for 0th indexes as 0
    #following dynamic programming approach to fill values
    for i in range(1,shot):
        for j in range(1,cap):
            if w[i] <= j:
                dp[i][j] = max(s[i] + dp[i-1][j-w[i]], dp[i-1][j])
            else:
                dp[i][j] = dp[i-1][j]

    #choosing the optimal pair of keyshots
    choice = []
    i = shot - 1
    j = cap - 1
    while i > 0 and j > 0:
        if dp[i][j] != dp[i-1][j]: #starting from last element and going further
            choice.append(i-1)
            j = j - w[i]
            i = i - 1
        else:
            i = i - 1

    return dp[shot-1][cap-1], choice



def eval_metrics(y_pred, y_true):
    '''Returns precision, recall and f1-score of given prediction and true value'''
    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return [precision, recall, fscore]



def select_keyshots(video_info, pred_score):
    '''Returns predicted scores(upsampled), selected keyshots indices, predicted summary of given video'''
    vidlen = video_info['length'][()]
    cps = video_info['change_points'][:]
    weight = video_info['n_frame_per_seg'][:]
    pred_score = np.array(pred_score)
    pred_score = upsample(pred_score, vidlen)
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
    _, selected = knapsack(pred_value, weight, int(0.2 * vidlen))
    selected = selected[::-1]
    key_labels = np.zeros((vidlen,))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1

    return pred_score.tolist(), selected, key_labels.tolist()




def upsample(down_arr, vidlen):
    '''Upsamples a given predicted score array to the size of video length'''
    up_arr = np.zeros(vidlen)
    ratio = vidlen // 320
    l = (vidlen - ratio * 320) // 2
    i = 0
    while i < 320:
        up_arr[l:l+ratio] = np.ones(ratio, dtype = int) * down_arr[0][i]
        l += ratio
        i += 1

    return up_arr



import numpy as np
import json
import os
from tqdm import tqdm, trange
import h5py
from prettytable import PrettyTable

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate
from tensorflow.keras.layers import Attention
from tensorflow.keras import Input, Model

def evaluate(epoch_i):
    '''Evaluates the model for given epoch on test dataset'''
    out_dict = {}
    eval_arr = []
    table = PrettyTable()
    table.title = 'Evaluation Result of epoch {}'.format(epoch_i)
    table.field_names = ['ID', 'Precision', 'Recall', 'F-Score']
    table.float_format = '1.5'

    with h5py.File('fcsn_tvsum.h5') as data_file:
        for feature, label, index in tqdm(test_dataset, desc = 'Evaluate', ncols = 90, leave = False):

            pred_score = model.predict(feature.reshape(-1,320,1024))
            video_info = data_file['video_'+str(index)]
            pred_score, pred_selected, pred_summary = select_keyshots(video_info, pred_score)
            true_summary_arr = video_info['user_summary'][:]
            eval_res = [eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr]
            eval_res = np.mean(eval_res, axis = 0).tolist()

            eval_arr.append(eval_res)
            table.add_row([index] + eval_res)

            out_dict[str(index)] = {
            'pred_score' : pred_score,
            'pred_selected' : pred_selected,
            'pred_summary' : pred_summary
            }
    eval_mean = np.mean(eval_arr, axis = 0).tolist()
    table.add_row(['mean'] + eval_mean)
    tqdm.write(str(table))



# Baseline implementation of algorithm mentioned in the paper (with only one extra BiLSTM layer) with some hyperparameter tuning

import numpy as np
import json
import os
from tqdm import tqdm, trange
import h5py
from prettytable import PrettyTable

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate
from tensorflow.keras.layers import Attention
from tensorflow.keras import Input, Model

encoder_inputs = Input(shape = (320, 1024))

encoder_BidirectionalLSTM = Bidirectional(LSTM(128, return_sequences = True, return_state = True))
encoder_BidirectionalLSTM1 = Bidirectional(LSTM(128, return_sequences = True, return_state = True))
encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM(encoder_inputs)
encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM1(encoder_out)
sh = Concatenate()([fh, bh])
ch = Concatenate()([fc, bc])
encoder_states = [sh, ch]

decoder_LSTM = LSTM(256, return_sequences = True, dropout = 0.01, recurrent_dropout = 0.01)
decoder_out = decoder_LSTM(encoder_out, initial_state = encoder_states)
decoder_out = decoder_LSTM(decoder_out)
attn_layer = Attention(name="Attention_Layer")
attn_out =  attn_layer([encoder_out, decoder_out])

decoder_concat_input = Concatenate(axis = -1, name = 'concat_layer')([decoder_out, attn_out])

dense = TimeDistributed(Dense(1, activation = 'sigmoid'))
decoder_pred = dense(decoder_concat_input)

model = Model(inputs = encoder_inputs, outputs = decoder_pred)

opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1 = 0.8, beta_2 = 0.8)
#opt = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()
t = trange(10, desc = 'Epoch', ncols = 90)
for epoch_i in t:

    model.fit(train_loader)
    evaluate(epoch_i)




# Improved with two extra BiLSTM layers, one extra LSTM layer and two more Time Distributed layers and some hyperparameter tuning

import numpy as np
import json
import os
from tqdm import tqdm, trange
import h5py
from prettytable import PrettyTable

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate
from tensorflow.keras.layers import Attention
from tensorflow.keras import Input, Model
from keras.layers import LeakyReLU

encoder_inputs = Input(shape = (320, 1024))

encoder_BidirectionalLSTM = Bidirectional(LSTM(64, return_sequences = True, return_state = True))
encoder_BidirectionalLSTM1 = Bidirectional(LSTM(64, return_sequences = True, return_state = True))
encoder_BidirectionalLSTM2 = Bidirectional(LSTM(64, return_sequences = True, return_state = True))
encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM(encoder_inputs)
encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM1(encoder_out)
encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM2(encoder_out)
sh = Concatenate()([fh, bh])
ch = Concatenate()([fc, bc])
encoder_states = [sh, ch]

decoder_LSTM = LSTM(128, return_sequences = True)
decoder_out = decoder_LSTM(encoder_out, initial_state = encoder_states)
decoder_out = decoder_LSTM(decoder_out)

attn_layer = Attention(name="Attention_Layer")
attn_out =  attn_layer([encoder_out, decoder_out])

decoder_concat_input = Concatenate(axis = -1, name = 'concat_layer')([decoder_out, attn_out])


dense = TimeDistributed(Dense(42, activation = 'relu'))
decoder_pred = dense(decoder_concat_input)

dense = TimeDistributed(Dense(14, activation = 'tanh'))
decoder_pred = dense(decoder_pred)

dense = TimeDistributed(Dense(1, activation = 'sigmoid'))
decoder_pred = dense(decoder_pred)

model = Model(inputs = encoder_inputs, outputs = decoder_pred)

opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.8, beta_2 = 0.8)
#opt = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
model.summary()
t = trange(1, desc = 'Epoch', ncols = 90)
for epoch_i in t:

    model.fit(train_loader)
    evaluate(epoch_i)
