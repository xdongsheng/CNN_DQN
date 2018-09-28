import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()

#   1
# sess=tf_reset()
#
# a=tf.constant(1.0)
# b=tf.constant(2.0)
#
# c=a+b
#
# c_run=sess.run(c)
#
# print('c={0}'.format(c_run))

#    2
# sess=tf_reset()
# a=tf.placeholder(dtype=tf.float32,shape=[1],name='a_placeholder')
# b=tf.placeholder(dtype=tf.float32,shape=[1],name='b_placeholder')
# c=a+b
# c0_run=sess.run(c,feed_dict={a:[1.0],b:[2.0]})
# c1_run=sess.run(c,feed_dict={a:[2.0],b:[4.0]})
# print('c0={0}'.format(c0_run))
# print('c1={0}'.format(c1_run))
# print('c0={0},c1={1}'.format(c0_run,c1_run))

#   3
# sess=tf_reset()
#
# a=tf.placeholder(dtype=tf.float32,shape=[None],name='a_placeholder')
# b=tf.placeholder(dtype=tf.float32,shape=[None],name='b_placeholder')
# c=a+b
#
# c0_run=sess.run(c,feed_dict={a:[1.0],b:[2.0]})
# c1_run=sess.run(c,feed_dict={a:[1.0,2.0],b:[2.0,3.0]})
# print(a)
# print(b)
# print('c0={0},c1={1}'.format(c0_run,c1_run))

#    4
# sess=tf_reset()
#
# a=tf.constant([[-1.],[-2.],[-3.]],dtype=tf.float32)
# b=tf.constant([[1.,2.,3.]],dtype=tf.float32)
#
# a_run,b_run=sess.run([a,b])
# print(a_run,b_run)
#
# c_elementwise=b*a
# c_matmul=tf.matmul(b,a)
# c_elementwise,c_matmul=sess.run([c_elementwise,c_matmul])
# print('{0}\n'.format(c_elementwise))
# print(c_matmul)
#
# b_mean=tf.reduce_mean(b)
# b_mean_run=sess.run(b_mean)
# print('\nb的reduceman值为：{0}'.format(b_mean_run))

# how to create variables
# sess=tf_reset()
#
# b=tf.constant([[1.,2.,3.]],dtype=tf.float32)
#
# b_run=sess.run(b)
# print('b:\n{0}'.format(b_run))
#
# var_init_value=[[2.0,4.0,6.0]]
# var=tf.get_variable(name='my_var',shape=[1,3],dtype=tf.float32,initializer=tf.constant_initializer(var_init_value))
# print(var)
# c=b+var
# print(b)
# print(var)
# print(c)
#
# sess.run(tf.global_variables_initializer())
# c_run=sess.run(c)
# print(sess.run(var))
# print(c_run)

# How to train a neural network for a simple regression problem
# generate the data
inputs=np.linspace(-2*np.pi,2*np.pi,10000)[:,None]
outputs=np.sin(inputs)+0.05*np.random.normal(size=[len(inputs),1])
plt.scatter(inputs[:,0],outputs[:,0],s=0.1,c='k',marker='o')
#plt.show()
#print('inputs:{0}\noutputs:{1}'.format(inputs[:,0],outputs))

sess=tf_reset()

def crate_model():
    # create inputs
    input_ph=tf.placeholder(dtype=tf.float32,shape=[None,1])
    output_ph=tf.placeholder(dtype=tf.float32,shape=[None,1])
    # create variables
    w0=tf.get_variable(name='w0',shape=[1,20],initializer=tf.contrib.layers.xavier_initializer())
    w1=tf.get_variable(name='w1',shape=[20,20],initializer=tf.contrib.layers.xavier_initializer())
    w2=tf.get_variable(name='w3',shape=[20,1],initializer=tf.contrib.layers.xavier_initializer())
    b0=tf.get_variable(name='b0',shape=[20],initializer=tf.constant_initializer(0.))
    b1=tf.get_variable(name='b1',shape=[20],initializer=tf.constant_initializer(0.))
    b2=tf.get_variable(name='b2',shape=[1],initializer=tf.constant_initializer(0.))
    weights=[w0,w1,w2]
    biases=[b0,b1,b2]
    activations=[tf.nn.relu,tf.nn.relu,None]
    # create computation graph
    layer=input_ph
    for W,b,activation in zip(weights,biases,activations):
        layer=tf.matmul(layer,W)+b
        if activation is not None:
            layer=activation(layer)
    output_pre=layer
    return input_ph,output_ph,output_pre

input_ph,output_ph,output_pred=crate_model()

# create loss
mse=tf.reduce_mean(0.5*tf.square(output_ph-output_pred))
# create optimizer
opt=tf.train.AdamOptimizer().minimize(mse)
# initialize variables
sess.run(tf.global_variables_initializer())
# create saver to save model variables
saver=tf.train.Saver()

# run training
# batch_size=32
# for train_step in range(10000):
#     # get a random subset of the training data
#     indices=np.random.randint(low=0,high=len(inputs),size=batch_size)
#
#     input_batch=inputs[indices]
#     output_batch=outputs[indices]
#
#     # run the optimizer and get the mse
#     _,mse_run=sess.run([opt,mse],feed_dict={input_ph:input_batch,output_ph:output_batch})
#     # print the mse every so often
#     if train_step%1000==0:
#         print('{0:04d} mse:{1:.3f}'.format(train_step,mse_run))
#         saver.save(sess,'/tmp/model.ckpt')

# now that the neural network is trained ,we can use it to make predictions:
sess=tf_reset()
# creat the model
input_ph,output_ph,output_pred=crate_model()
# restore the saved model
saver=tf.train.Saver()
saver.restore(sess,'/tmp/model.ckpt')

output_pred_run=sess.run(output_pred,feed_dict={input_ph:inputs})
plt.scatter(inputs[:,0],outputs[:,0],c='k',marker='o',s=0.1)
plt.scatter(inputs[:,0],output_pred_run[:,0],c='r',marker='o',s=0.1)
plt.show()