import tensorflow as tf

from modules import *

class Transformer():

    def __init__(self, sequence_length, word_vocab_size, hidden_units, tag_vocab_size, num_blocks, num_heads, margin):

        self.sequence_length = sequence_length
        self.word_vocab_size = word_vocab_size
        self.hidden_units = hidden_units
        self.tag_vocab_size = tag_vocab_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.margin = margin


        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_q")
        self.input_x_11 = tf.placeholder(tf.int32, [None, sequence_length], name="input_q_tag")

        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos")
        self.input_x_22 = tf.placeholder(tf.int32, [None, sequence_length], name="input_pos_tag")

        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_neg")
        self.input_x_33 = tf.placeholder(tf.int32, [None, sequence_length], name="input_neg_tag")

        self.dropout_prob = tf.placeholder_with_default(0.0, [], name="dropout_prob")
        self.is_training = tf.placeholder_with_default(False, [], name="is_training")

        self.enc_q = embedding(self.input_x_1,
                               vocab_size=word_vocab_size,
                               num_units=hidden_units,
                               scale=True,
                               scope="Embedding")

        self.enc_pos = embedding(self.input_x_2,
                               vocab_size=word_vocab_size,
                               num_units=hidden_units,
                               scale=True,
                               scope="Embedding",
                               reuse=True)

        self.enc_neg = embedding(self.input_x_3,
                                 vocab_size=word_vocab_size,
                                 num_units=hidden_units,
                                 scale=True,
                                 scope="Embedding",
                                 reuse=True)

        self.enc_q += embedding(self.input_x_11,
                                vocab_size=tag_vocab_size,
                                num_units=hidden_units,
                                scale=True,
                                scope="Tag_Embedding")

        self.enc_pos += embedding(self.input_x_22,
                                vocab_size=tag_vocab_size,
                                num_units=hidden_units,
                                scale=True,
                                scope="Tag_Embedding",
                                reuse=True)

        self.enc_neg += embedding(self.input_x_33,
                                  vocab_size=tag_vocab_size,
                                  num_units=hidden_units,
                                  scale=True,
                                  scope="Tag_Embedding",
                                  reuse=True)

        self.enc_q += embedding(
                                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_x_1)[1]), 0), [tf.shape(self.input_x_1)[0], 1]),
                                  vocab_size=sequence_length,
                                  num_units=hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="Position_Embedding")

        self.enc_pos += embedding(
                                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_x_2)[1]), 0), [tf.shape(self.input_x_2)[0], 1]),
                                vocab_size=sequence_length,
                                num_units=hidden_units,
                                zero_pad=False,
                                scale=False,
                                scope="Position_Embedding",
                                reuse=True)

        self.enc_neg += embedding(
                                    tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_x_3)[1]), 0), [tf.shape(self.input_x_3)[0], 1]),
                                    vocab_size=sequence_length,
                                    num_units=hidden_units,
                                    zero_pad=False,
                                    scale=False,
                                    scope="Position_Embedding",
                                    reuse=True)

        self.enc_q = tf.layers.dropout(self.enc_q, rate=self.dropout_prob, training=self.is_training)
        self.enc_pos = tf.layers.dropout(self.enc_pos, rate=self.dropout_prob, training=self.is_training)
        self.enc_neg = tf.layers.dropout(self.enc_neg, rate=self.dropout_prob, training=self.is_training)

        for i in range(num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                self.enc_q = multihead_attention(queries=self.enc_q,
                                                 keys=self.enc_q,
                                                 num_units=hidden_units,
                                                 num_heads=num_heads,
                                                 dropout_rate=self.dropout_prob,
                                                 is_training=self.is_training,
                                                 causality=False)
                self.enc_q = feedforward(self.enc_q,
                                         num_units=[4*hidden_units, hidden_units])

                self.enc_pos = multihead_attention(queries=self.enc_pos,
                                                 keys=self.enc_pos,
                                                 num_units=hidden_units,
                                                 num_heads=num_heads,
                                                 dropout_rate=self.dropout_prob,
                                                 is_training=self.is_training,
                                                 causality=False,
                                                 reuse=True)
                self.enc_pos = feedforward(self.enc_pos,
                                         num_units=[4 * hidden_units, hidden_units],
                                         reuse=True)

                self.enc_neg = multihead_attention(queries=self.enc_neg,
                                                   keys=self.enc_neg,
                                                   num_units=hidden_units,
                                                   num_heads=num_heads,
                                                   dropout_rate=self.dropout_prob,
                                                   is_training=self.is_training,
                                                   causality=False,
                                                   reuse=True)
                self.enc_neg = feedforward(self.enc_neg,
                                           num_units=[4 * hidden_units, hidden_units],
                                           reuse=True)


        with tf.variable_scope("Out"):
            pred_pos = self.cos(self.enc_q, self.enc_pos)
            pred_neg = self.cos(self.enc_q, self.enc_neg)

            self.output_prob = tf.identity(pred_pos, name="output_prob")

            self.loss = tf.reduce_mean(tf.maximum(0., pred_neg + self.margin - pred_pos))

        self._model_stats()



    def cos(self, input_a, input_b):
        norm_a = tf.nn.l2_normalize(input_a, dim=1)
        norm_b = tf.nn.l2_normalize(input_b, dim=1)
        cos_sim = tf.expand_dims(tf.reduce_sum(tf.multiply(norm_a, norm_b), 1), -1)
        return cos_sim


    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))



