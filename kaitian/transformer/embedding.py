#! -*- coding: utf-8 -*-
# 自定义层

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import Model
from functools import partial
import json


from kaitian.transformer.position import TransformerCoordinateEmbedding
from kaitian.transformer.transformer import TransformerACT, TransformerBlock
from kaitian.transformer.transformer import gelu


class FactorizedEmbedding(Layer):
    """基于低秩分解的Embedding层
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, **kwargs):
        super(FactorizedEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        if hidden_dim is None:
            self.hidden_dim = output_dim
        else:
            self.hidden_dim = hidden_dim

    def build(self, input_shape):
        super(FactorizedEmbedding, self).build(input_shape)
        self._embeddings = self.add_weight(name='embeddings',
                                           shape=(self.input_dim,
                                                  self.hidden_dim),
                                           initializer='uniform')
        self._project_kernel = self.add_weight(name='project_kernel',
                                               shape=(self.hidden_dim,
                                                      self.output_dim),
                                               initializer='glorot_uniform')
        self.embeddings = K.dot(self._embeddings, self._project_kernel)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        outputs = K.gather(self._embeddings, inputs)
        outputs = K.dot(outputs, self._project_kernel)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim, )




def gelu_erf(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))

def gelu_tanh(x):
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf

gelu_version = 'erf'

def get_gelu():
    if gelu_version == 'erf':
        return gelu_erf
    else:
        return gelu_tanh

def set_gelu(version):
    """提供gelu版本切换功能，默认为Erf版本
    """
    global gelu_version
    gelu_version = version

def padding_mask(seq_q, seq_k):
    """
    A sentence is filled with 0, which is not what we need to pay attention to
    :param seq_k: shape of [N, T_k], T_k is length of sequence
    :param seq_q: shape of [N, T_q]
    :return: a tensor with shape of [N, T_q, T_k]
    """

    q = K.expand_dims(K.ones_like(seq_q, dtype="float32"), axis=-1)  # [N, T_q, 1]
    k = K.cast(K.expand_dims(K.not_equal(seq_k, 0), axis=1), dtype='float32')  # [N, 1, T_k]
    return K.batch_dot(q, k, axes=[2, 1])


class BertModel(object):
    """构建跟Bert一样结构的Transformer-based模型
    这是一个比较多接口的基础类，然后通过这个基础类衍生出更复杂的模型
    """
    def __init__(
            self,
            vocab_size,  # 词表大小
            max_position_embeddings,  # 序列最大长度
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate,  # Dropout比例
            embedding_size=None,  # 是否指定embedding_size
            with_mlm=False,  # 是否包含MLM部分
            keep_words=None,  # 要保留的词ID列表
            block_sharing=False,  # 是否共享同一个transformer block
            max_depth: int = 2,
            num_heads: int = 2,
            model_dim: int = 128,
            embedding_dropout: float = 0.0,
            residual_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            output_dropout: float = 0.0,
            confidence_penalty_weight: float = 0.1,
            l2_reg_penalty: float = 1e-6,
            compression_window_size: int = None,
            max_seq_len: int=150,
    ):
        if keep_words is None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(keep_words)
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        if embedding_size:
            self.embedding_size = embedding_size
        else:
            self.embedding_size = hidden_size
        self.with_mlm = with_mlm
        if hidden_act == 'gelu':
            self.hidden_act = get_gelu()
        else:
            self.hidden_act = hidden_act
        self.keep_words = keep_words
        self.block_sharing = block_sharing
        self.additional_outputs = []

        self.num_heads = num_heads
        self.max_depth = max_depth

        self.model_dim = model_dim
        self.embedding_dropout = embedding_dropout
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.confidence_penalty_weight = confidence_penalty_weight
        self.l2_reg_penalty = l2_reg_penalty
        self.compression_window_size = compression_window_size
        self.max_seq_len = max_seq_len

    def build(self):
        """Bert模型构建函数
        """
        x_in = Input(shape=(self.max_seq_len, ), name='Input-Token')
        s_in = Input(shape=(None, ), name='Input-Segment')
        x, s = x_in, s_in

        # 自行构建Mask
        sequence_mask = K.cast(K.greater(x, 0), 'float32')

        # Embedding部分
        if self.embedding_size == self.hidden_size:
            token_embedding = Embedding(input_dim=self.vocab_size,
                                        output_dim=self.embedding_size,
                                        name='Embedding-Token')
        else:
            token_embedding = FactorizedEmbedding(
                input_dim=self.vocab_size,
                hidden_dim=self.embedding_size,
                output_dim=self.hidden_size,
                name='Embedding-Token')
        output = token_embedding(x)

        mask = Lambda(lambda x: padding_mask(x, x))(x_in)

        output = self.__encoder(output, mask)

        # s = Embedding(input_dim=2,
        #               output_dim=self.hidden_size,
        #               name='Embedding-Segment')(s)
        # x = Add(name='Embedding-Token-Segment')([x, s])


        # x = PositionEmbedding(input_dim=self.max_position_embeddings,
        #                       output_dim=self.hidden_size,
        #                       name='Embedding-Position')(x)
        # if self.dropout_rate > 0:
        #     x = Dropout(rate=self.dropout_rate, name='Embedding-Dropout')(x)
        # x = LayerNormalization(name='Embedding-Norm')(x)
        # layers = None
        #
        # # 主要Transformer部分
        # for i in range(self.num_hidden_layers):
        #     attention_name = 'Encoder-%d-MultiHeadSelfAttention' % (i + 1)
        #     feed_forward_name = 'Encoder-%d-FeedForward' % (i + 1)
        #     x, layers = self.transformer_block(
        #         inputs=x,
        #         sequence_mask=sequence_mask,
        #         attention_mask=self.compute_attention_mask(i, s_in),
        #         attention_name=attention_name,
        #         feed_forward_name=feed_forward_name,
        #         input_layers=layers)
        #     x = self.post_processing(i, x)
        #     if not self.block_sharing:
        #         layers = None
        #
        # if self.with_mlm:
        #     # Masked Language Model 部分
        #     x = Dense(self.hidden_size,
        #               activation=self.hidden_act,
        #               name='MLM-Dense')(x)
        #     x = LayerNormalization(name='MLM-Norm')(x)
        #     x = EmbeddingDense(token_embedding, name='MLM-Proba')(x)

        if self.additional_outputs:
            self.model = Model([x_in, s_in], [output] + self.additional_outputs)
        else:
            self.model = Model([x_in, s_in], output)

    def __encoder(self, emb_inputs, mask):

        transformer_enc_layer = TransformerBlock(
            name='transformer_enc',
            num_heads=self.num_heads,
            residual_dropout=self.residual_dropout,
            attention_dropout=self.attention_dropout,
            compression_window_size=self.compression_window_size,
            use_masking=False,
            vanilla_wiring=True)
        coordinate_embedding_layer = TransformerCoordinateEmbedding(name="coordinate_emb1",
                                                                    max_transformer_depth=self.max_depth)
        transformer_act_layer = TransformerACT(name='adaptive_computation_time1')

        emb_project_layer = Conv1D(self.model_dim, activation="linear",
                                   kernel_size=1,
                                   name="emb_project")

        next_step_input = emb_project_layer(emb_inputs)
        act_output = next_step_input

        for step in range(self.max_depth):
            next_step_input = coordinate_embedding_layer(next_step_input, step=step)
            next_step_input = transformer_enc_layer(next_step_input)
            next_step_input, act_output = transformer_act_layer(next_step_input)

        transformer_act_layer.finalize()

        next_step_input = act_output

        return next_step_input

    def load_weights_from_checkpoint(self, checkpoint_file):
        """从预训练好的Bert的checkpoint中加载权重
        """
        model = self.model
        loader = partial(tf.train.load_variable, checkpoint_file)

        if self.keep_words is None:
            keep_words = slice(0, None)
        else:
            keep_words = self.keep_words

        if self.embedding_size == self.hidden_size:
            model.get_layer(name='Embedding-Token').set_weights([
                loader('bert/embeddings/word_embeddings'),
            ])
            #model.get_layer(name='Embedding-Token').trainable = False
        else:
            model.get_layer(name='Embedding-Token').set_weights([
                loader('bert/embeddings/word_embeddings'),
                loader('bert/embeddings/word_embeddings_2'),
            ])
            #model.get_layer(name='Embedding-Token').trainable = False

        #print(model.get_layer(name='Embedding-Token').get)
        # model.get_layer(name='Embedding-Position').set_weights([
        #     loader('bert/embeddings/position_embeddings'),
        # ])
        # model.get_layer(name='Embedding-Segment').set_weights([
        #     loader('bert/embeddings/token_type_embeddings'),
        # ])
        # model.get_layer(name='Embedding-Norm').set_weights([
        #     loader('bert/embeddings/LayerNorm/gamma'),
        #     loader('bert/embeddings/LayerNorm/beta'),
        # ])
        #
        # for i in range(self.num_hidden_layers):
        #     try:
        #         model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
        #     except ValueError as e:
        #         continue
        #     try:
        #         layer_name = 'layer_%d' % i
        #         loader('bert/encoder/%s/attention/self/query/kernel' % layer_name)
        #     except:
        #         layer_name = 'layer_shared'
        #         loader('bert/encoder/%s/attention/self/query/kernel' % layer_name)
        #     model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
        #         loader('bert/encoder/%s/attention/self/query/kernel' % layer_name),
        #         loader('bert/encoder/%s/attention/self/query/bias' % layer_name),
        #         loader('bert/encoder/%s/attention/self/key/kernel' % layer_name),
        #         loader('bert/encoder/%s/attention/self/key/bias' % layer_name),
        #         loader('bert/encoder/%s/attention/self/value/kernel' % layer_name),
        #         loader('bert/encoder/%s/attention/self/value/bias' % layer_name),
        #         loader('bert/encoder/%s/attention/output/dense/kernel' % layer_name),
        #         loader('bert/encoder/%s/attention/output/dense/bias' % layer_name),
        #     ])
        #     model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
        #         loader('bert/encoder/%s/attention/output/LayerNorm/gamma' % layer_name),
        #         loader('bert/encoder/%s/attention/output/LayerNorm/beta' % layer_name),
        #     ])
        #     model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
        #         loader('bert/encoder/%s/attention/output/LayerNorm/gamma' % layer_name),
        #         loader('bert/encoder/%s/attention/output/LayerNorm/beta' % layer_name),
        #     ])
        #     model.get_layer(
        #         name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
        #             loader('bert/encoder/%s/intermediate/dense/kernel' % layer_name),
        #             loader('bert/encoder/%s/intermediate/dense/bias' % layer_name),
        #             loader('bert/encoder/%s/output/dense/kernel' % layer_name),
        #             loader('bert/encoder/%s/output/dense/bias' % layer_name),
        #         ])
        #     model.get_layer(
        #         name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
        #             loader('bert/encoder/%s/output/LayerNorm/gamma' % layer_name),
        #             loader('bert/encoder/%s/output/LayerNorm/beta' % layer_name),
        #         ])
        #
        # if self.with_mlm:
        #     model.get_layer(name='MLM-Dense').set_weights([
        #         loader('cls/predictions/transform/dense/kernel'),
        #         loader('cls/predictions/transform/dense/bias'),
        #     ])
        #     model.get_layer(name='MLM-Norm').set_weights([
        #         loader('cls/predictions/transform/LayerNorm/gamma'),
        #         loader('cls/predictions/transform/LayerNorm/beta'),
        #     ])
        #     model.get_layer(name='MLM-Proba').set_weights([
        #         loader('cls/predictions/output_bias')[keep_words],
        #     ])


def load_pretrained_model(config_path,
                          checkpoint_file,
                          with_mlm=False,
                          seq2seq=False,
                          keep_words=None,
                          albert=False):
    """根据配置文件和checkpoint文件来加载模型
    """
    config = json.load(open(config_path))

    # if seq2seq:
    #     Bert = Bert4Seq2seq
    # else:
    Bert = BertModel

    bert = Bert(vocab_size=config['vocab_size'],
                max_position_embeddings=config['max_position_embeddings'],
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size'],
                hidden_act=config['hidden_act'],
                dropout_rate=config['hidden_dropout_prob'],
                embedding_size=config.get('embedding_size'),
                with_mlm=with_mlm,
                keep_words=keep_words,
                block_sharing=albert)

    bert.build()
    bert.load_weights_from_checkpoint(checkpoint_file)

    return bert.model

if __name__ == '__main__':
    maxlen = 100

    config_path = 'E:\\Research\\Data\\albert_base_zh\\bert_config.json'
    checkpoint_path = 'E:\\Research\\Data\\albert_base_zh\\bert_model.ckpt'
    dict_path = 'E:\\Research\\Data\\albert_base_zh\\vocab.txt'

    model = load_pretrained_model(
        config_path,
        checkpoint_path,
        #keep_words=keep_words,
        albert=True
    )

    model.summary()

    #model.s