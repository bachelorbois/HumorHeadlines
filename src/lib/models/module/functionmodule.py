import tensorflow as tf

def sigmoid_3(inputs):
    """Activation for 3 x Sigmoid
    
    Arguments:
        in {float} -- Neuron value
    
    Returns:
        float -- Scaled value between 0 - 3
    """
    return tf.sigmoid(inputs)

def RMSE(y_true, y_pred):
    """Root Mean Squared Error
    
    Arguments:
        y_true {float[]} -- True values
        y_pred {float[]} -- Predicted values
    
    Returns:
        float -- Loss of predictions
    """
    mse = tf.losses.MSE(y_true, y_pred)
    return tf.sqrt(mse)