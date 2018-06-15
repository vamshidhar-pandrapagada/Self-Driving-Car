# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:37:24 2018

@author: vpandrap
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:56:55 2018

@author: Vamshidhar P
"""
import tensorflow as tf
import numpy as np
from Evaluate_Model import evaluate_model
from sklearn.model_selection import train_test_split


def train_Convnet(X_train, X_test, y_train, y_test, image_shape, batch_size, epochs, lr, dropout_prob):

    """
        : arma_model: List of input features
        : arma_labels: list_of_labels
    """
    save_model_path = './model_save/tf_model_convnet'
    
    def one_hot_encode(x):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """
        one_hot_array=[]
        for l in x:
            holder = np.repeat(0,2)
            np.put(holder,l,1)
            one_hot_array.append(holder)

        return np.array(one_hot_array)
    

    def batches(batch_size, features, labels):
        n_batches = len(features)//batch_size
                       
        # only full batches    
        features = features[:n_batches*batch_size]
        for i in range(0, len(features), batch_size):
            batch_X = features[i:i + batch_size]
            batch_Y = labels[i:i + batch_size]
            yield batch_X, batch_Y
            
    def batches_test(batch_size, features):
        for i in range(0, len(features), batch_size):
            batch_X = features[i:i + batch_size]
            yield batch_X
    
    def print_epoch_stats(epoch_i, sess, last_features, last_labels):
        """
        Print cost and validation accuracy of an epoch
        """
        current_cost = sess.run(cost,feed_dict={features: last_features, labels: last_labels, keep_prob: dropout_prob})
        training_accuracy = sess.run(accuracy,feed_dict={features: last_features, labels: last_labels, keep_prob: dropout_prob})
        valid_accuracy = sess.run(accuracy,feed_dict={features: X_Val, labels: y_val, keep_prob: 1.0})
        print('Epoch: {:<4} - Cost: {:<8.3} Training Accuracy: {:<5.3} Validation Accuracy: {:<5.3}'.format(epoch_i,current_cost, training_accuracy, valid_accuracy))
        
    def batchnorm(Ylogits, batch_norm_TF, iteration, offset):
        # adding the iteration prevents from averaging across non-existing iterations
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) 
        bnepsilon = 1e-5

        mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(batch_norm_TF, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(batch_norm_TF, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages            
    
   
    
    def conv_layer(features_tensor, conv_num_outputs, conv_ksize, conv_strides, padding = 'SAME', drop_out = False,
                   max_pool = False, pool_ksize = None, pool_strides = None):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        image_depth = features_tensor.get_shape().as_list()[-1]
        conv_depth = conv_num_outputs
        wghts = list(conv_ksize)
        wghts.extend([image_depth,conv_depth])
        weights = tf.Variable(tf.truncated_normal(wghts,stddev=0.1))
        bias = tf.Variable(tf.truncated_normal([conv_depth],stddev=0.1))
        
        conv_ksize = list(conv_ksize)
        conv_ksize.insert(0,1)
        conv_ksize.insert(len(conv_ksize),1)
        
        conv_strides = list(conv_strides)
        conv_strides.insert(0,1)
        conv_strides.insert(len(conv_strides),1)
        
        x = tf.nn.conv2d (input = features_tensor, filter = weights, strides = conv_strides, padding = padding)
        x = tf.nn.bias_add(x, bias)
        conv_x = tf.nn.relu(x)
        #print(conv_x.get_shape().as_list())   
            
        
        if max_pool:
            pool_ksize = list(pool_ksize)
            pool_ksize.insert(0,1)
            pool_ksize.insert(len(pool_ksize),1)
            
            pool_strides = list(pool_strides)
            pool_strides.insert(0,1)
            pool_strides.insert(len(pool_strides),1)     
               
            conv_x = tf.nn.max_pool(value = conv_x, ksize = pool_ksize, strides = pool_strides, padding = padding )
        #print(conv_maxpool.get_shape().as_list())        
        
        return conv_x 
    
    def flatten(x_tensor):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """
        
        shape = x_tensor.get_shape().as_list()
        reshape_to = np.prod(shape[1:])
        
        return (tf.reshape(x_tensor,[-1,reshape_to]))
            
    def fully_connected(features_tensor, keep_prob, num_outputs, num_inputs = None):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : features_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        if (num_inputs != None):
            inputs = num_inputs
        else:
            inputs = features_tensor.get_shape().as_list()[1]
        weights = tf.Variable(tf.truncated_normal([inputs, num_outputs],stddev=0.1))
        bias = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.1))        
        fc = tf.add(tf.matmul(features_tensor,weights),bias)
        
        #fc = tf.nn.softmax(fc)
        fc = tf.nn.relu(fc)
        fc = tf.nn.dropout(fc, keep_prob = keep_prob)
        return fc
    
    def output(features_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : features_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        inputs = features_tensor.get_shape().as_list()[1]
        weights = tf.Variable(tf.truncated_normal([inputs, num_outputs],stddev=0.1))
        bias = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.1))        
        fc = tf.add(tf.matmul(features_tensor,weights),bias)            
        return fc       
    
    
    
    # Split Test data into randomized validation and test sets
    rand_state = np.random.randint(0, 100)
    print('Splitting Features into Train and test Setsl')
    X_Val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = rand_state)
    
        
    print("One-Hot Encode Target") 
    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val) 
    y_test = one_hot_encode(y_test) 
    print("One-Hot Encode Target Complete") 
    
    
    print('Train, Val and Test Shapes: ', X_train.shape, X_Val.shape, X_test.shape)
    print('Train Labels , Validation Labels and Test Labels Shapes: ', y_train.shape, y_val.shape, y_test.shape)
    
       
    
    tf.reset_default_graph()
    
    n_classes = y_train.shape[1]

       
    # Features and Labels
    image_shape = list(image_shape)
    image_shape.insert(0, None)   
    features = tf.placeholder(tf.float32, image_shape, name ='x')
    print('Features Tensor Shape')
    print(features.get_shape().as_list())
    
    labels = tf.placeholder(tf.float32, [None, n_classes] , name = 'y')
    
    # True when training
    #is_training = tf.placeholder(tf.bool, shape =(), name = 'is_training')
    # Learning Rate
    learning_rate = tf.placeholder(tf.float32)
    
    # Drop out Probability
    keep_prob = tf.placeholder(tf.float32, name = 'kp')
    
    conv_layer1 = conv_layer(features, conv_num_outputs = 24, conv_ksize = (5,5), conv_strides = (2,2))
    conv_layer2 = conv_layer(conv_layer1, conv_num_outputs = 36, conv_ksize = (5,5), conv_strides = (2,2))
    conv_layer3 = conv_layer(conv_layer2, conv_num_outputs = 48, conv_ksize = (5,5), conv_strides = (2,2))
    conv_layer4 = conv_layer(conv_layer3, conv_num_outputs = 64, conv_ksize = (3,3), conv_strides = (1,1))
    conv_layer5 = conv_layer(conv_layer4, conv_num_outputs = 64, conv_ksize = (3,3), conv_strides = (1,1))
    flat = flatten(conv_layer5)
    
    fc1 = fully_connected(flat, keep_prob, num_outputs = 100)
    fc2 = fully_connected(fc1, keep_prob, num_outputs = 50)
    fc3 = fully_connected(fc2, keep_prob, num_outputs = 10)
    
    Ylogits = output(fc3, num_outputs = n_classes)
    Ylogits = tf.identity(Ylogits, name='logits')
    
    Y = tf.nn.softmax(Ylogits)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels=labels))
    #cost = tf.reduce_mean(tf.reduce_sum(-labels * tf.log(Ylogits) - (1 - labels) * tf.log(1 - Ylogits), 1))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name = 'accuracy')

    #NUM_CORES = 4  # Choose how many cores to use.
    #sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,intra_op_parallelism_threads=NUM_CORES))    

    init = tf.global_variables_initializer()
    
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)) as sess:
    
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        print("Training 9 layer Convolutional Neural Network")
        for epoch_i in range(epochs+1):
            train_batches = batches(batch_size, X_train, y_train)
            # Loop over all batches
            for batch_features, batch_labels in train_batches:
                train_feed_dict = {
                        features: batch_features,
                        labels: batch_labels,
                        learning_rate: lr,
                        keep_prob: dropout_prob}
                sess.run(optimizer, feed_dict=train_feed_dict)                

                #Print cost and validation accuracy for every 10 iterations
                #if (epoch_i%10 == 0):
            print_epoch_stats(epoch_i, sess, batch_features, batch_labels)
                
        saver = tf.train.Saver()
        saver.save(sess, save_model_path)
        saver.export_meta_graph(save_model_path + '.meta')
        
        # If you just want to predict without input labels
        #pred =  loaded_logits
        test_model_feed_dict = {features: X_test, keep_prob: 1.0}
        softmax_predictions = sess.run(tf.nn.softmax(Ylogits),feed_dict = test_model_feed_dict)
        
    y_pred_proba = np.array(softmax_predictions)[:, 1]
    y_pred = []
    for i in range(len(softmax_predictions)):
        if softmax_predictions[i][1] >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = np.array(y_pred)
    evaluate_model(y_pred, y_test, 'Neural_Network Test Set',  
                   pred_proba = y_pred_proba, cold_encode = True)