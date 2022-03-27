import numpy as np
import tensorflow as tf
from keras.models import load_model

meta_sample = 1000
max_del_branch_num = 2


def get_topo_data():
    meta_X = np.zeros((meta_sample,max_del_branch_num))
    for meta_i in range(meta_sample):
        data = np.load(r"./data/topo%s.npy"%(meta_i+1),allow_pickle=True).item()
        del_branches = data['del_branches']
        if len(del_branches) == 1:del_branches.append(0)
        meta_X[meta_i,:] = np.array(del_branches).reshape(1,max_del_branch_num)
    return  meta_X 

def get_Ymeta():
    for meta_i in range(meta_sample):
        print('\r%s/%s'%(meta_i,meta_sample),end='\r')
        model = load_model(r'./models/base_model_%s.h5'%meta_i) 
        weights1 = model.layers[1].get_weights()[0]
        weights2 = model.layers[1].get_weights()[1]
        weights3 = model.layers[2].get_weights()[0]
        weights4 = model.layers[2].get_weights()[1]
        weights5 = model.layers[3].get_weights()[0]
        weights6 = model.layers[3].get_weights()[1]
        weights1 = weights1.reshape(1,weights1.shape[0]*weights1.shape[1])
        weights2 = weights2.reshape(1,weights2.shape[0])
        weights3 = weights3.reshape(1,weights3.shape[0]*weights3.shape[1]) 
        weights4 = weights4.reshape(1,weights4.shape[0])   
        weights5 = weights5.reshape(1,weights5.shape[0]*weights5.shape[1]) 
        weights6 = weights6.reshape(1,weights6.shape[0])                      
        theta = np.hstack((weights1,weights2,weights3,weights4,weights5,weights6))
        if meta_i == 0:
            Ymate = theta
        else:
            Ymate = np.vstack((Ymate,theta))
    return Ymate
        
# In[]
meta_X = get_topo_data()
Ymate = get_Ymeta()
# In[]
X = meta_X
Y = Ymate
input_shape = X.shape
output_shape = Y.shape
input0 = tf.keras.Input(shape=(input_shape[1],))
x = tf.keras.layers.Embedding(45, 8, input_length=2)(input0)
x = tf.keras.layers.Reshape((16,))(x)
x = tf.keras.layers.Dense(256,name='dense1',activation='gelu')(x)
x = tf.keras.layers.Dense(256,name='dense2',activation='gelu')(x)
output = tf.keras.layers.Dense(output_shape[1],name='dense3')(x)
model = tf.keras.Model(input0, output)
model.compile(optimizer='Adam',loss='MSE')
model.predict(meta_X[0,:].reshape(1,2))
#
# In[]
model.fit(X,Y,epochs=1000)
model.compile(optimizer='Adam',loss='MSE')

# In[]
xx0 = model.predict(meta_X[0,:].reshape(1,2))
xx1=model.predict(meta_X[1,:].reshape(1,2))