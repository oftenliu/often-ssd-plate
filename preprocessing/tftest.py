import tensorflow as tf
import numpy as np
def get_uniform():
    num = np.random.uniform(0,1)
    num = tf.convert_to_tensor(num)
    # num = tf.random_uniform((),0,1)
    # tf.cond()
    # if tf.greater(num, tf.constant(0.4)):
    #     print ('11111')
    # else:
    #     print ('00000')
    return num

tensornum = get_uniform()
with tf.Session() as sess:
    for i in range (100):
        tn = sess.run(tensornum)
        print (tn)

