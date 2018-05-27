import tflearn
import speech_data

learning_rate = 0.0001
steps = 300000

batch = word_atch = speech_data.mfcc_batch_generator(64)
X, Y = next(batch)
trainX, trainY = X,Y
testX, testY, X, Y

net = tflearn.input_data([None, 20, 80])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tf.Learn.fully_connected(net, 10, activation='softmax')
net = tf.learn.regression(net, optimizer='adam', learning_rate= learning_rate, loss= 'categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)
while
