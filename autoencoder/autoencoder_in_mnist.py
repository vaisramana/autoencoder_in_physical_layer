
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imread, imsave, imresize
import numpy as np

PARAM_FILE_NAME = "./param/autoencoder"
#PARAM_FILE_PATH = "./param/"
DEFAULT_DATASET_PATH = "/home/nokia/machinelearning/mnist/"
OUTPUT_IMAGE_PATH = "./image/"
TRAINING_EPOCH_NUMBER = 10000

def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.1), name)
def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1, shape = shape), name)

def translate_lable(vector):
    labelNum = 0
    for i in vector:
        if i:
            return labelNum
        labelNum += 1
    return "INVALID"

def autoencoder(input_dataset_path, restart_training_flag):
    #input dataset
    mnist = input_data.read_data_sets(input_dataset_path, one_hot = True)
    
    #construct network
    x = tf.placeholder(tf.float32, shape = [None, 784])
    e_W_1 = weight_variable([784, 300], "e_W_1")
    e_b_1 = bias_variable([300], "e_b_1")
    e_layer1 = tf.nn.relu(tf.matmul(x, e_W_1) + e_b_1)
    e_W_2 = weight_variable([300, 100], "e_W_2")
    e_b_2 = bias_variable([100], "e_b_2")
    e_layer2 = tf.nn.relu(tf.matmul(e_layer1, e_W_2) + e_b_2)
    e_W_3 = weight_variable([100, 20], "e_W_3")
    e_b_3 = bias_variable([20], "e_b_3")
    code_layer = tf.nn.relu(tf.matmul(e_layer2, e_W_3) + e_b_3)
    d_W_1 = weight_variable([20, 100], "d_W_1")
    d_b_1 = bias_variable([100], "d_b_1")
    d_layer1 = tf.nn.relu(tf.matmul(code_layer, d_W_1) + d_b_1)
    d_W_2 = weight_variable([100, 300], "d_W_2")
    d_b_2 = bias_variable([300], "d_b_2")
    d_layer2 = tf.nn.relu(tf.matmul(d_layer1, d_W_2) + d_b_2)
    d_W_3 = weight_variable([300, 784], "d_W_3")
    d_b_3 = bias_variable([784], "d_b_3")
    output_layer = tf.nn.relu(tf.matmul(d_layer2, d_W_3) + d_b_3)
    
    loss = tf.reduce_mean(tf.pow(output_layer - x, 2))
    optimizer = tf.train.RMSPropOptimizer(0.002).minimize(loss)
    init_op = tf.global_variables_initializer()
    
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    if restart_training_flag: 
        print "start training from beginning"
        sess.run(init_op)
        for i in range(TRAINING_EPOCH_NUMBER):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                print("step %d, loss %g"%(i, loss.eval(feed_dict={x:batch[0]})))
            optimizer.run(feed_dict={x: batch[0]})
        print("save network to %s" %(PARAM_FILE_NAME))
    else:
        print ("restore network from %s" %(PARAM_FILE_NAME))
        saver.restore(sess, PARAM_FILE_NAME)
    print("final loss %g" % loss.eval(feed_dict={x: mnist.test.images}))
    saver.save(sess, PARAM_FILE_NAME)
    
    
    output_nd = output_layer.eval(feed_dict={x:mnist.train.images})
    for i in [0, 1, 2, 3, 4]:
        img_ori = np.reshape(mnist.train.images[i, :], (28, 28)) # 28 by 28 matrix 
        img_rcs = np.reshape(output_nd[i,:], (28, 28))
        imsave(OUTPUT_IMAGE_PATH+"index"+str(i)+"_lable"+str(translate_lable(mnist.train.labels[i]))+"_original.jpeg", img_ori)
        imsave(OUTPUT_IMAGE_PATH+"index"+str(i)+"_lable"+str(translate_lable(mnist.train.labels[i]))+"_reconstructed.jpeg", img_rcs)
    
    	
    	


def test_network_save():
    '''
    x = tf.placeholder(tf.float32, shape = [None, 10])
    W = weight_variable([10, 5], "W")
    b = bias_variable([5], "b")
    output = tf.nn.relu(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(tf.pow(output, 2))
    optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)
    init_op = tf.global_variables_initializer()
    
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(init_op)
    print saver.save(sess, "./param/test")
    print saver.last_checkpoints
    print W
    print b
    '''
    W = tf.Variable([[1,1,1],[2,2,2]],dtype = tf.float32,name='w')  
    b = tf.Variable([[0,1,2]],dtype = tf.float32,name='b')  

    init = tf.initialize_all_variables()  
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        sess.run(init)  
        save_path = saver.save(sess,"save/model.ckpt")  

def test_network_restore():
    '''
    x = tf.placeholder(tf.float32, shape = [None, 10])
    W = weight_variable([10, 5], "W")
    b = bias_variable([5], "b")
    output = tf.nn.relu(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(tf.pow(output, 2))
    optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)
    init_op = tf.global_variables_initializer()
    
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    print saver.last_checkpoints
    print saver.restore(sess, "./param/test")
    print W
    print b
    '''
    W = tf.Variable(tf.truncated_normal(shape=(2,3)),dtype = tf.float32,name='w')  
    b = tf.Variable(tf.truncated_normal(shape=(1,3)),dtype = tf.float32,name='b')  

    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        saver.restore(sess,"save/model.ckpt") 


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', "--input_dataset_path", help="input dataset path", default=DEFAULT_DATASET_PATH)
  parser.add_argument("-r", "--restart_training", help="restart traning network", action="store_true")
  args = parser.parse_args()
  
  #test_network_save()
  #test_network_restore()
  autoencoder(args.input_dataset_path, args.restart_training)