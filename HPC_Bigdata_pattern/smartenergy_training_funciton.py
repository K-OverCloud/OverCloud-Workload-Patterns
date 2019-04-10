import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

tf.set_random_seed(777)

# Hyper Parameters
input_data_column_cnt = 5  # 입력데이터의 컬럼 개수
output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

seq_length = 10  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
forget_bias = 1.0  # 망각편향(기본값 1.0)
num_stacked_layers = 1  # stacked LSTM layers 개수
keep_prob = 1.0  # Dropout Ratio

epoch_num = 3000
learning_rate = 0.01

# Loads Input Data (CSV from Smart Energy Service)
input_file_name = './input/input.csv'
encoding = 'euc-kr'
names = ['date','aircon_temp','ext_humidity','ext_temperature','dev1.humidity','dev1.temperature','weather_stat']
raw_dataframe = pd.read_csv(input_file_name, names=names, encoding=encoding)
raw_dataframe.info() #Display Data Info

del raw_dataframe['date'] # Delete Time
service_data = raw_dataframe.values[1:].astype(np.float) # Casting String to Float

input_data = service_data[:,1:]
print("input_data[0]: ", input_data[0])


target_data = service_data[:,:1]
print("target_data[0]: ", target_data[0])

x = np.concatenate((input_data, target_data), axis=1) # Concatenate

y = x[:, [-1]]
x = x[:, :-1]
print(x)
print(y)

dataX = [] # Input Dataata
dataY = [] # Target Data


for i in range(0, len(y) - seq_length):
    _x = x[i : i+seq_length]
    _y = y[i + seq_length]
    if i is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# Training Data Ratio ( 70% )
train_size = int(len(dataY) * 0.7)
# Test Data Ratio ( 30% )
test_size = len(dataY) - train_size

# Store Training Data
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

# Store Test Data
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

# Model Input --> X
X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)

# Model Output --> Y
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

# Target --> For validation
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)

#Prediction --> For validation
predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)

# LSTM MODEL
# num_units: 각 Cell 출력 크기
# forget_bias:  to the biases of the forget gate
#              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
# state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
# state_is_tuple: False ==> they are concatenated along the column axis.
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


# Stacked RNNs with num_stacked_layers
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

# Connect RNN Cells
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

# one to many [-:.1]
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

# loss function
loss = tf.reduce_sum(tf.square(hypothesis - Y))

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE(Root Mean Square Error)
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions)))
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

train_error_summary = []
test_error_summary = []
test_predict = ''

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# Start Training
start_time = datetime.datetime.now()  # Training Start Time
print('Training Start....')
for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):
        # 학습용데이터로 rmse오차를 구한다
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)

        # 테스트용데이터로 rmse오차를 구한다
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)

        # Print Error Rate
        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1, train_error, test_error,
                                                                                 test_error - train_error))
end_time = datetime.datetime.now()  # Training End Time
elapsed_time = end_time - start_time  # Traiing elapsed Time
print('elapsed_time:', elapsed_time)
print('elapsed_time per epoch:', elapsed_time / epoch_num)

# Print Hyper Parameters
print('input_data_column_cnt:', input_data_column_cnt, end='')
print(',output_data_column_cnt:', output_data_column_cnt, end='')

print(',seq_length:', seq_length, end='')
print(',rnn_cell_hidden_dim:', rnn_cell_hidden_dim, end='')
print(',forget_bias:', forget_bias, end='')
print(',num_stacked_layers:', num_stacked_layers, end='')
print(',keep_prob:', keep_prob, end='')

print(',epoch_num:', epoch_num, end='')
print(',learning_rate:', learning_rate, end='')

print(',train_error:', train_error_summary[-1], end='')
print(',test_error:', test_error_summary[-1], end='')
print(',min_test_error:', np.min(test_error_summary))

# Show Results with Graphs
plt.figure(1)
plt.plot(train_error_summary, 'gold')
plt.plot(test_error_summary, 'b')
plt.xlabel('Epoch(x100)')
plt.ylabel('Root Mean Square Error')

plt.figure(2)
plt.plot(testY, 'r')
plt.plot(test_predict, 'b')
plt.xlabel('Time Period')
plt.ylabel('Air Con. temp')
plt.show()

# sequence length만큼의 가장 최근 데이터를 슬라이싱
recent_data = np.array([x[len(x) - seq_length:]])
print("recent_data.shape:", recent_data.shape)
print("recent_data:", recent_data)

# 추천 에어컨 온도를 예측한다.
result_predict = tf.placeholder(tf.float32, [None, None])
result_predict = sess.run(hypothesis, feed_dict={X: recent_data})

print("Recommended Air Con. temp", result_predict)
print("Recommended Air Con. temp", np.round(result_predict[0]))
# test_predict = reverse_min_max_scaling(price,test_predict) # 금액데이터 역정규화한다
# print("Tomorrow's stock price", test_predict[0]) # 예측한 주가를 출력한다

# Export model
builder = tf.saved_model.builder.SavedModelBuilder('./trained_model')

# Build the signature_def_map.
classification_inputs = tf.saved_model.utils.build_tensor_info(X)
classification_output = tf.saved_model.utils.build_tensor_info(Y)

classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(
    inputs={ tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs },
    outputs={ tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_output},
    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
tensor_info_y = tf.saved_model.utils.build_tensor_info(tf.convert_to_tensor(result_predict[0]))

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={tf.saved_model.signature_constants.PREDICT_INPUTS: tensor_info_x},
        outputs={tf.saved_model.signature_constants.PREDICT_OUTPUTS: tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_images': prediction_signature,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature,
      },
    main_op=tf.tables_initializer(),
    strip_default_attrs=True)

builder.save()