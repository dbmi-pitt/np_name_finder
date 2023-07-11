import matplotlib.pyplot as plt
from typing import Literal, List
plt.rcParams['patch.force_edgecolor'] = True


import tensorflow as tf
print("tf.version: ", tf.version.VERSION)
print("tf.keras.version: ", tf.keras.__version__)
print("tf.config.devices: ", tf.config.list_physical_devices())

# tf.compat.v1.enable_eager_execution()
# tf.compat.v1.disable_eager_execution()
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert(tf.test.gpu_device_name())

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.
tf.executing_eagerly()

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.figure(figsize=(15,10))
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["Train", "Validation"], loc="upper left")
    plt.title(title, fontsize='xx-large', weight="bold")
    plt.ylabel(metric.capitalize(), fontsize='large', weight="bold")
    plt.xlabel("Epoch", fontsize='large', weight="bold")
    if metric == "accuracy":
        plt.axhline(y=max(history["val_" + metric]), color="gray", linestyle="--")
    elif metric == "loss":
        plt.axhline(y=min(history["val_" + metric]), color="gray", linestyle="--")
    plt.show()

class CosineSimilarity(tf.keras.layers.Layer):
    '''Cosine similarity to be calculated as sum(x*y)/(sqrt(sum(x))*sqrt(sum(y))).
    This is achieved through Tensorflow functions to retain performance.
    
    Parameters
    ----------
    vects: tf.TensorArray
    
    Returns
    -------
    cosine_similarity: tf.TensorArray
       The result of the cosine similarity between the vectors.    
    '''
    __name__ = 'CosineSimilarity'
    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__()
       
    @tf.function  # The decorator converts `cosine_similarity` into a tensolflow `Function`.
    def call(self, vects: tf.TensorArray) -> tf.TensorArray:
        x, y = vects
        return tf.math.divide(tf.reduce_sum(tf.multiply(x,y), axis=1, keepdims=True), tf.multiply(tf.norm(x, ord=2, axis=1, keepdims=True), tf.norm(y, ord=2, axis=1, keepdims=True)))

    def get_config(self):
        return super(CosineSimilarity, self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CosineDistance(tf.keras.layers.Layer):
    '''Cosine distance to be calculated as 1-(cosine similarity).
    Where cosine similarity equals sum(x*y)/(sqrt(sum(x))*sqrt(sum(y))).
    This is achieved through Tensorflow functions to retain performance.
    
    Parameters
    ----------
    vects: tf.TensorArray
        
    Returns
    -------
    cosine_distance: tf.TensorArray
        The result of 1-cosine similarity between the vectors. 
    '''
    __name__ = 'CosineDistance'
    def __init__(self, **kwargs):
        super(CosineDistance, self).__init__()
       
    @tf.function  # The decorator converts `cosine_distance` into a tensorflow `Function`.
    def call(self, vects: tf.TensorArray) -> tf.TensorArray:
        x, y = vects
        return  tf.math.subtract(tf.constant([1.0]), tf.math.divide(tf.reduce_sum(tf.multiply(x,y), axis=1, keepdims=True), tf.multiply(tf.norm(x, ord=2, axis=1, keepdims=True), tf.norm(y, ord=2, axis=1, keepdims=True))))

    def get_config(self):
        return super(CosineDistance, self).get_config()
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class ContrastiveLoss(tf.keras.losses.Loss):
    '''Returns a value between 0 and 1 representing the average error of the y_pred vector by comparing it to the y_true.
    '''
    __name__ = 'ContrastiveLoss'
    def __init__(self, margin: tf.float32 = 1.0, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = tf.constant(margin)
        
    @tf.function # The decorator converts `loss` into a tensolflow `Function`.
    def call(self, y_true: tf.TensorArray, y_pred: tf.TensorArray) -> tf.Tensor:
        return tf.math.reduce_mean((1 - y_true) * tf.math.square(y_pred) + (y_true) * tf.math.square(tf.math.maximum(self.margin - (y_pred), 0.0)), axis = -1)
    
    def get_config(self):
        config = super(ContrastiveLoss, self).get_config()
        config.update({
            "margin": str(self.margin)
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def get_model(model_type: Literal["lstm", "gru", "rnn"], maxlen: int, embedding_dim: int, num_rnn_node: int, bidirectional: bool, num_dense_node: int, num_layer: int, activation_fn: str, learning_rate: float, optimizer_fn: Literal["Adam", "RMSprop", "SGD"], margin: float, output_activation: str, random_seed: int) -> tf.keras.Model: 
    '''Specifies the architecture of the model to be trained.
    '''

#     tf.random.set_seed(random_seed)

    input_x = tf.keras.layers.Input(maxlen)
    input_1 = tf.keras.layers.Input(maxlen)
    input_2 = tf.keras.layers.Input(maxlen)
    embedding = tf.keras.layers.Embedding(input_dim=28, output_dim=embedding_dim, mask_zero=True)
    x = tf.keras.layers.BatchNormalization()(embedding(input_x))
    
    match model_type:
        case "lstm":
            if bidirectional:
                x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_rnn_node))(x)
            else:
                x = tf.keras.layers.LSTM(num_rnn_node)(x)
        case "gru":
            if bidirectional:
                x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_node))(x)
            else:
                x = tf.keras.layers.GRU(num_rnn_node)(x)
        case "rnn":
            if bidirectional:
                x = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(num_rnn_node))(x)
            else:
                x = tf.keras.layers.SimpleRNN(num_rnn_node)(x)
    
        
    num = num_dense_node
    for _ in range(num_layer):
        x = tf.keras.layers.Dense(num, activation=activation_fn)(x)
        num /= 2
        
    embedding_network = tf.keras.Model(input_x, x)

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

#     cosine_similarity = CosineSimilarity()
#     merge_layer = tf.keras.layers.Lambda(cosine_similarity)([tower_1, tower_2])
    cosine_distance = CosineDistance()
    merge_layer = tf.keras.layers.Lambda(cosine_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = tf.keras.layers.Dense(1, activation= output_activation)(normal_layer)
    contr = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    
    match optimizer_fn:
        case "Adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        case "RMSprop":                
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        case "SGD":                
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    contr.compile(loss=ContrastiveLoss(margin = margin), optimizer=opt, metrics=["accuracy"])
    
    return contr


# vectors = [[[1.0,2.0,3.0], [1.0,2.0,3.0]] , [[3.0,4.0,5.0],[3.0,4.0,5.0]]]

# cosine_similarity = CosineSimilarity()
# cosine_similarity.call(vectors)

# cosine_distance = CosineDistance()
# cosine_distance.call(vectors)

# 1 - 0.9827076 = 0.017292399999999986

# a = ContrastiveLoss(margin = 1.0)
# print(a.get_config())
# tf.assert_ a.call(np.array([1,0,1]), np.array([1,0,1])).numpy() == 0.0
# tf.assert_ a.call(np.array([0,0,0]), np.array([1,0,1])).numpy() >= 0.666
# tf.assert_ a.call(np.array([0,1,0]), np.array([1,0,1])).numpy() >= 1.0