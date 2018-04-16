import tensorflow as tf
import numpy as np
import pandas as pd
import io

from tensorflow.contrib import rnn
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#from google.colab import files
from datetime import datetime

tf.reset_default_graph()

# to read csv file from the drive
#uploaded = files.upload()
#df = pd.read_csv(io.StringIO(uploaded['Suez Cement55.csv'].decode('utf-8')))
df = pd.read_csv("C:\\Users\\mohamed ismail\\Desktop\\GP And Data\\Suez Cement.csv")
del df["SYMBOL_CODE"]
del df['TRADE_DATE']

#d= df
#print(d.axes)
# training from 80->60 and test from 40->20

df = df.values
df = df.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))


# split into train and test sets 1286
train_size = 800 #int(len(df) * 0.67)
test_size = len(df) - train_size
train, test = df[0:train_size,:], df[train_size:len(df),:]

train = scaler.fit_transform(train)
test = scaler.fit_transform(test)




'''
plt.plot(d['CLOSE_PRICE'],label='line one')
plt.plot(d['OPEN_PRICE'],label='line two')
plt.xlabel("x-axes")
plt.ylabel("y_axes")
plt.title("try and erorr")
plt.legend()
plt.show()

'''





n_train=800
batch_size=20
num_units=20
n_classes=1
time_steps=2
input_size=3
n_epoch=10

# none here refere to batch size
x=tf.placeholder("float",[None,time_steps,input_size])
y=tf.placeholder("float")



def rnn_output_layer(x):
  layer={'weights':tf.Variable(tf.random_normal([num_units,n_classes])),
         "biases":tf.Variable(tf.random_normal([n_classes]))}
         
  
  # to convert the shape from [batch size , time_steps, input size] to list of tensor[batch size , input size] of length time_steps
  input = tf.unstack(x,time_steps,1)
  
  # create lstm cell that have num_units neurons in the layer
  lstm_layer = rnn.BasicLSTMCell(num_units,forget_bias=1)
  outputs,_ = rnn.static_rnn(lstm_layer,input,dtype="float")  #output is list of tensors we consider only to the last one of shabe[batch_size,num_units]

  #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
  out_layer=tf.add( tf.matmul(outputs[-1],layer['weights']),layer['biases'])

  
  return out_layer

def modell_netw():
  
  prediction = rnn_output_layer(x)
  
  #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
  cost = tf.reduce_mean( tf.squared_difference(prediction,y) )
  
  optimizer = tf.train.AdamOptimizer().minimize(cost) 
  
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for epoch in range(n_epoch):
      epoch_loss=0
      start=0
      
      for _ in range(int(n_train/batch_size)):
        
        end = start + batch_size
        
        x_epoch =train[start:end,:-1] 
        y_epoch =train[start:end,-1]
        
       
        x_epoch = x_epoch.reshape((batch_size,time_steps,input_size))        
        y_epoch=y_epoch.reshape((batch_size,n_classes))
     
            
        _,c = sess.run([optimizer,cost],feed_dict={x:x_epoch , y:y_epoch})
        epoch_loss += c
        start = end
        
      print("the epoch loss is ",epoch_loss,"\n")  
      
    
    correct=tf.squared_difference(prediction,y)
    accuracy = tf.reduce_mean(correct)
    print('mse ',accuracy.eval({x:test[:,:-1].reshape((-1,time_steps,input_size)),y:test[:,-1].reshape(-1,1)}))

    xx=prediction
    xc=xx.eval({x:test[:,:-1].reshape((-1,time_steps,input_size)),y:test[:,-1].reshape(-1,1)})
    
    #print(xc,test[:,-1])
    plt.plot(xc)
    plt.plot(test[:,-1])
    plt.show()
  
modell_netw()  
  
