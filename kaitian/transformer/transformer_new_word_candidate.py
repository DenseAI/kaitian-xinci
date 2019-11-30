import json
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock

import keras
import keras.backend as K
from keras import Input, Model, regularizers
from keras.layers import Dropout, Conv1D, Lambda, Flatten, Dense, Add
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from kaitian.transformer.position import TransformerCoordinateEmbedding
from kaitian.transformer.transformer import TransformerACT, TransformerBlock
from kaitian.transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from kaitian.common.utils import load_dictionary


def label_smoothing_loss(y_true, y_pred):
    shape = K.int_shape(y_pred)
    n_class = shape[2]
    eps = 0.1
    y_true = y_true * (1 - eps) + eps / n_class
    return categorical_crossentropy(y_true, y_pred)


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


class Transformer_Segmenter:

    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 ws_num_classes: int,
                 pos_num_classes: int,
                 max_seq_len: int,
                 embedding_size_word: int = 300,
                 model_dim: int = 128,
                 num_filters: int = 128,
                 max_depth: int = 8,
                 num_heads: int = 8,
                 embedding_dropout: float = 0.0,
                 residual_dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 output_dropout: float = 0.0,
                 confidence_penalty_weight: float = 0.1,
                 l2_reg_penalty: float = 1e-6,
                 compression_window_size: int = None,
                 use_crf: bool = True,
                 optimizer=Adam(),
                 src_tokenizer: Tokenizer = None,
                 tgt_tokenizer: Tokenizer = None,
                 weights_path: str = None,
                 sparse_target: bool = False,
                 use_universal_transformer: bool = True,
                 num_gpu: int = 1):

        """

        :param src_vocab_size:  源字库大小
        :param tgt_vocab_size:  目标标签数量
        :param max_seq_len:     最大输入长度
        :param model_dim:       Transformer 模型维度
        :param max_depth:       Universal Transformer 深度
        :param num_heads:       多头注意力头数
        :param embedding_dropout: 词嵌入失活率
        :param residual_dropout:  残差失活率
        :param attention_dropout: 注意力失活率
        :param confidence_penalty_weight: confidence_penalty 正则化，仅在禁用CRF时有效
        :param l2_reg_penalty:  l2 正则化率
        :param compression_window_size: 压缩窗口大小
        :param use_crf:     是否使用crf
        :param optimizer:   优化器
        :param src_tokenizer: 源切割器
        :param tgt_tokenizer: 目标切割器
        :param weights_path: 权重路径
        :param num_gpu: 使用gpu数量
        """

        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.ws_num_classes = ws_num_classes
        self.pos_num_classes = pos_num_classes
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.max_depth = max_depth
        self.num_gpu = num_gpu
        self.embedding_size_word = embedding_size_word
        self.model_dim = model_dim
        self.num_filters = num_filters
        self.embedding_dropout = embedding_dropout
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.confidence_penalty_weight = confidence_penalty_weight
        self.l2_reg_penalty = l2_reg_penalty
        self.compression_window_size = compression_window_size
        self.use_crf = use_crf
        self.sparse_target = sparse_target
        self.use_universal_transformer = use_universal_transformer

        self.model, self.parallel_model = self.__build_model()

        if weights_path is not None:
            try:
                self.model.load_weights(weights_path)
            except Exception as e:
                logging.error(e)
                logging.info("Fail to load weights, create a new model!")

    def __build_model(self):
        assert self.max_depth >= 1, "The parameter max_depth is at least 1"
        word_ids = Input(shape=(self.max_seq_len,), dtype='int32', name='word_ids')

        l2_regularizer = (regularizers.l2(self.l2_reg_penalty) if self.l2_reg_penalty else None)

        embedding_layer = ReusableEmbedding(
            self.src_vocab_size,           #vocabulary_size,
            self.embedding_size_word,      #word_embedding_size,
            input_length=self.max_seq_len,
            name='bpe_embeddings',
            # Regularization is based on paper "A Comparative Study on
            # Regularization Strategies for Embedding-based Neural Networks"
            # https://arxiv.org/pdf/1508.03721.pdf
            embeddings_regularizer=l2_regularizer)

        coordinate_embedding_layer = TransformerCoordinateEmbedding(
            self.max_depth if self.use_universal_transformer else 1,
            name='coordinate_embedding')

        # 分词向量
        next_step_input, embedding_matrix = embedding_layer(word_ids)
        emb_project_layer = Conv1D(self.model_dim, activation="linear",
                                   kernel_size=1,
                                   name="emb_project")
        emb_dropout_layer = Dropout(self.embedding_dropout, name="emb_dropout")
        next_step_input = emb_project_layer(emb_dropout_layer(next_step_input))


        if self.use_universal_transformer:
            # Building a Universal Transformer (2018)
            act_layer = TransformerACT(name='adaptive_computation_time')
            transformer_block = TransformerBlock(
                name='transformer',
                num_heads=self.num_heads,
                residual_dropout=self.residual_dropout,
                attention_dropout=self.attention_dropout,
                # Allow bi-directional attention
                use_masking=False)

            act_output = next_step_input
            for i in range(self.max_depth):
                next_step_input = coordinate_embedding_layer(next_step_input, step=i)
                #next_step_input = add_segment_layer([next_step_input, segment_embeddings])
                next_step_input = transformer_block(next_step_input)
                next_step_input, act_output = act_layer(next_step_input)

            act_layer.finalize()
            next_step_input = act_output
        else:
            # Building a Vanilla Transformer (described in
            # "Attention is all you need", 2017)
            next_step_input = coordinate_embedding_layer(next_step_input, step=0)
            #next_step_input = add_segment_layer([next_step_input, segment_embeddings])
            for i in range(self.max_depth):
                next_step_input = (
                    TransformerBlock(
                        name='transformer' + str(i),
                        num_heads=self.num_heads,
                        residual_dropout=self.residual_dropout,
                        attention_dropout=self.attention_dropout,
                        use_masking=False,  # Allow bi-directional attention
                        vanilla_wiring=True)
                    (next_step_input))

        cls_node_slice = (
            # selecting the first output position in each sequence
            # (responsible for classification)
            Lambda(lambda x: x[:, 0], name='cls_node_slicer')
            (next_step_input))
        class_prediction = (
            Dense(1, name='class_prediction', activation='sigmoid')
            (cls_node_slice))
        model = Model(inputs=[word_ids],outputs=[class_prediction])

        parallel_model = model
        if self.num_gpu > 1:
            parallel_model = multi_gpu_model(model, gpus=self.num_gpu)

        parallel_model.compile(optimizer=self.optimizer, loss=[binary_crossentropy], metrics=['accuracy'])

        return model, parallel_model


    def decode_sequences(self, sequences):
        sequences = self._seq_to_matrix(sequences)
        output = self.model.predict_on_batch([sequences])  # [N, -1, chunk_size + 1]
        return output


    def decode_texts(self, texts, sequences):
        output = self.decode_sequences(sequences)
        return output

    def _seq_to_matrix(self, sequences):
        # max_len = len(max(sequences, key=len))
        return pad_sequences(sequences, maxlen=self.max_seq_len, padding="post")

    def get_config(self):
        return {
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'ws_num_classes': self.ws_num_classes,
            'pos_num_classes': self.pos_num_classes,
            'max_seq_len': self.max_seq_len,
            'max_depth': self.max_depth,
            'model_dim': self.model_dim,
            'embedding_size_word': self.embedding_size_word,
            'confidence_penalty_weight': self.confidence_penalty_weight,
            'l2_reg_penalty': self.l2_reg_penalty,
            'embedding_dropout': self.embedding_dropout,
            'residual_dropout': self.residual_dropout,
            'attention_dropout': self.attention_dropout,
            'compression_window_size': self.compression_window_size,
            'num_heads': self.num_heads,
            'use_crf': self.use_crf
        }

    __singleton = None
    __lock = Lock()

    @staticmethod
    def get_or_create(config, src_dict_path=None,
                      tgt_dict_path=None,
                      weights_path=None,
                      num_gpu=1,
                      optimizer=Adam(),
                      encoding="utf-8"):
        Transformer_Segmenter.__lock.acquire()
        try:
            if Transformer_Segmenter.__singleton is None:
                if type(config) == str:
                    with open(config, encoding=encoding) as file:
                        config = dict(json.load(file))
                elif type(config) == dict:
                    config = config
                else:
                    raise ValueError("Unexpect config type!")

                if src_dict_path is not None:
                    src_tokenizer = load_dictionary(src_dict_path, encoding)
                    config['src_tokenizer'] = src_tokenizer
                if tgt_dict_path is not None:
                    config['tgt_tokenizer'] = load_dictionary(tgt_dict_path, encoding)

                config["num_gpu"] = num_gpu
                config['weights_path'] = weights_path
                config['optimizer'] = optimizer
                Transformer_Segmenter.__singleton = Transformer_Segmenter(**config)
        except Exception:
            traceback.print_exc()
        finally:
            Transformer_Segmenter.__lock.release()
        return Transformer_Segmenter.__singleton


get_or_create = Transformer_Segmenter.get_or_create


def save_config(obj, config_path, encoding="utf-8"):
    with open(config_path, mode="w+", encoding=encoding) as file:
        json.dump(obj.get_config(), file)