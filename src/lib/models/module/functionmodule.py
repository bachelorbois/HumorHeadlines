import tensorflow as tf
from tensorflow.keras import backend
import numpy as np

def sigmoid_3(inputs):
    """Activation for 3 x Sigmoid
    
    Arguments:
        in {float} -- Neuron value
    
    Returns:
        float -- Scaled value between 0 - 3
    """
    return tf.multiply(tf.sigmoid(inputs), 3)

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

@tf.function
def StepActivation(x):
    if x <= 0.2:
        return 0.2
    elif x <= 0.4:
        return 0.4
    elif x <= 0.6: 
        return 0.6
    elif x <= 0.8:
        return 0.8
    elif x <= 1.0:
        return 1.0
    elif x <= 1.2:
        return 1.2
    elif x <= 1.4:
        return 1.4
    elif x <= 1.6: 
        return 1.6
    elif x <= 1.8:
        return 1.8
    elif x <= 2.0:
        return 2.0
    elif x <= 2.2:
        return 2.2
    elif x <= 2.4:
        return 2.4
    elif x <= 2.6: 
        return 2.6
    elif x <= 2.8:
        return 2.8
    elif x <= 3.0:
        return 3.0
    else:
        return x

Vector = np.vectorize(StepActivation)
Stepy = lambda x: Vector(x).astype(np.float32)