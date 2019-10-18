from keras.engine.topology import Layer
from keras import backend as K, initializers, regularizers, constraints


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionLayer(Layer):
    """
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, weights)`.
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
#                                 name='{}_W'.format(self.name),
#                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
#                                     name='{}_b'.format(self.name),
#                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
#                                 name='{}_u'.format(self.name),
#                                 name='u',
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        return a

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

class Self_Attention(Layer):
    
    def __init__(self, output_dim, **kwargs):        
        self.output_dim = output_dim        
        super(Self_Attention, self).__init__(**kwargs)    
    
    def build(self, input_shape):      
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)      
        self.kernel = self.add_weight(name='kernel',    
                                      shape=(3,input_shape[2], self.output_dim),    
                                      initializer='uniform',                     
                                      trainable=True)      
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它   
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x):      
        WQ = K.dot(x, self.kernel[0])        
        WK = K.dot(x, self.kernel[1])  
        WV = K.dot(x, self.kernel[2])     
        print("WQ.shape",WQ.shape)        
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)  
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))   
        QK = QK / (64**0.5)     
        QK = K.softmax(QK)      
        print("QK.shape",QK.shape)   
        V = K.batch_dot(QK,WV)    
        
        return V    
    
    def compute_output_shape(self, input_shape):    
        return (input_shape[0],input_shape[1],self.output_dim)

