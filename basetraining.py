import numpy as np
import tensorflow as tf
from keras.models import load_model


meta_sample = 1
base_sample = 1000
N = 39
col_num = 58

def get_std0_cols(nparray):
    output_list = []
    for i in range(nparray.shape[0]):
        if nparray[i] == 0:
            output_list.append(i)
    return output_list

def load_data(data_label):
    data = np.load(r"./data/topo%s.npy"%data_label,allow_pickle=True).item()
    base_X = np.zeros((base_sample,N*2))
    base_Y = np.zeros((base_sample,N*2))
    for i in range(base_sample):
        base_X[i,0:39] = np.real(data['PF_samples'][i]['S']).reshape(39,)
        base_X[i,39:78] = np.imag(data['PF_samples'][i]['S']).reshape(39,)
        base_Y[i,0:39] = data['PF_samples'][i]['Va'].reshape(39,)
        base_Y[i,39:78] = data['PF_samples'][i]['Vm'].reshape(39,)
    X_std = base_X.std(axis=0)
    base_X = np.delete(base_X,get_std0_cols(X_std), axis = 1) 
    base_X_mean = base_X.mean(axis=0)
    base_X_std = base_X.std(axis=0)
    base_X = (base_X-base_X_mean)/base_X_std
    meta_X = np.array(data['del_branches'])
    return base_X,base_Y,base_X_mean,base_X_std

def base_train(X,Y,base_label):
    input_shape = X.shape
    output_shape = Y.shape
    input0 = tf.keras.Input(shape=(input_shape[1],))
    x = tf.keras.layers.Dense(64,name='dense1',activation='gelu')(input0)
    x = tf.keras.layers.Dense(32,name='dense2',activation='gelu')(x)
    output = tf.keras.layers.Dense(output_shape[1],name='dense3')(x)
    model = tf.keras.Model(input0, output)
    model.compile(optimizer='Adam',loss='MSE')
    model.fit(X,Y,epochs=100,batch_size=32,verbose=0)
    model.save(r'./models/base_model_%s.h5'%base_label)
    return model
    
    

if __name__ == '__main__':
    total_base_Xmean = np.zeros((meta_sample,col_num))
    total_base_Xstd = np.zeros((meta_sample,col_num))
    for base_i in range(meta_sample):
        base_X,base_Y,base_X_mean,base_X_std = load_data(base_i+1)
        total_base_Xmean[base_i,:] = base_X_mean
        total_base_Xstd[base_i,:] = base_X_std
        print('-'*5+'base learner-%s is training'%base_i+'-'*5)
        model = base_train(base_X,base_Y,base_i)
        Y_hat = model.predict(base_X)
    print('Done')


