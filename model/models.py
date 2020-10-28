from layers import *
from utils import *
from inits import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def pairwise_inner_product(mat_1, mat_2):
    n, _ = mat_1.shape.as_list()
    mat_expand = tf.expand_dims(mat_2, 0)
    mat_expand = tf.tile(mat_expand, [n, 1, 1])
    mat_expand = tf.transpose(mat_expand, [1, 0, 2])
    inner_prod = tf.multiply(mat_expand, mat_1)
    inner_prod = tf.reduce_sum(inner_prod, axis=-1)
    return inner_prod

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):

        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)
        self._log_vars()

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class MVURE(Model):
    """ Multi-View Urban Region Embedding Model. """
    def __init__(self, placeholders, input_dim, output_dim, **kwargs):
        super(MVURE, self).__init__(**kwargs)

        self.inputs = placeholders['feature']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        with tf.name_scope("loss"):
            # Weight decay loss
            # for var in self.layers[0].vars.values():
            #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

            self.mob_loss = self._mob_loss(self.s_out[0], self.t_out[0], self.placeholders["mob_adj"][0])
            self.poi_loss = self._adj_loss(self.poi_out[0], self.placeholders["poi_adj"], "poi_loss")
            self.chk_loss = self._adj_loss(self.chk_out[0], self.placeholders["chk_adj"], "chk_loss")
            self.loss =  self.mob_loss + self.poi_loss + self.chk_loss

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("mob loss", self.mob_loss)
        tf.summary.scalar("poi loss", self.poi_loss)
        tf.summary.scalar("chk loss", self.chk_loss)


    def _mob_loss(self, s_embeddings, t_embeddings, mob, namespace="mob_loss"):
        with tf.name_scope(namespace):
            inner_prod = pairwise_inner_product(s_embeddings, t_embeddings)
            phat = tf.nn.softmax(inner_prod, axis=-1)
            loss = tf.reduce_sum(-tf.multiply(mob, tf.log(phat)))
            inner_prod = pairwise_inner_product(t_embeddings, s_embeddings)
            phat = tf.nn.softmax(inner_prod, axis=-1)
            loss += tf.reduce_sum(-tf.multiply(tf.transpose(mob), tf.log(phat)))
            return loss

    def _adj_loss(self, embeddings, adj, namespace="adj_loss"):
        with tf.name_scope(namespace):
            inner_prod = pairwise_inner_product(embeddings, embeddings)
            return tf.losses.mean_squared_error(inner_prod, adj)

    def _build(self):

        self.layers = []
        s_out = self._gat(self.inputs, 12, 12, "SMOB-0", self.placeholders["s_bias"], tf.nn.relu)
        t_out = self._gat(self.inputs, 12, 12, "TMOB-0", self.placeholders["t_bias"], tf.nn.relu)

        poi_out = self._gat(self.inputs, 12, 12, "POI-0", self.placeholders["poi_bias"], tf.nn.relu)
        chk_out = self._gat(self.inputs, 12, 12, "CHK-0", self.placeholders["chk_bias"], tf.nn.relu)

        self.fused_outs = self.self_attn([s_out, t_out, poi_out, chk_out], self.output_dim, "1")

        alpha = 0.8
        s_out = alpha * self.fused_outs[0] + (1 - alpha) * s_out
        t_out = alpha * self.fused_outs[1] + (1 - alpha) * t_out
        poi_out = alpha * self.fused_outs[2] + (1 - alpha) * poi_out
        chk_out = alpha * self.fused_outs[3] + (1 - alpha) * chk_out

        self.out = self.mv_attn([s_out, t_out, poi_out, chk_out])

        beta = 0.5
        self.s_out = beta*s_out + (1-beta)*self.out
        self.t_out = beta*t_out + (1-beta)*self.out
        self.poi_out = beta*poi_out + (1-beta)*self.out
        self.chk_out = beta*chk_out + (1-beta)*self.out


    def mv_attn(self, views):
        input_dim = views[0].shape.as_list()[-1]
        with tf.name_scope("MV-ATTN"):
            attn_weight_1 = glorot([input_dim, input_dim], name='mv-weights')
            attn_bias = uniform([input_dim], name='mv-bias')
            weights = tf.convert_to_tensor([
                tf.sigmoid(tf.matmul(view[0], attn_weight_1)+attn_bias)
                for view in views])
            self.view_weights = tf.unstack(weights, axis=0)
            attns = [tf.multiply(view, weight) for weight, view in zip(self.view_weights, views)]
            out = tf.add_n(attns)
        return out

    def self_attn(self, views, outdim, name):
        views = tf.convert_to_tensor(views)
        with tf.name_scope("SELF-ATTN-"+name):
            outs = []
            for i in range(4):
                Q = tf.layers.dense(views, 48)
                K = tf.layers.dense(views, 48)
                V = views
                attention = tf.matmul(Q, K, transpose_b=True)
                d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
                attention = tf.divide(attention, tf.sqrt(d_k))
                attention = tf.nn.softmax(attention, axis=-1)
                output = tf.matmul(attention, V)
                output = tf.squeeze(output)
                outs.append(output)
            output = tf.reduce_mean(outs, axis=0)
            return tf.unstack(output, axis=0)

    def _gat(self, inputs, output_dim, num_head, namespace, bias, act, final=False):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, 0)
        with tf.name_scope(namespace):
            x = inputs
            input_dim = x.shape.as_list()[-1]
            attns = []
            for i in range(num_head):
                gattn = GraphAttention(input_dim=input_dim,
                                       output_dim=output_dim,
                                       act=act,
                                       input_dropout=0,
                                       coef_dropout=0.2,
                                       bias=bias,
                                       logging=self.logging)
                self.layers.append(gattn)
                attns.append(gattn(x))
            if final:
                out = tf.add_n(attns)/num_head
            else:
                out = tf.concat(attns, axis=-1)
            return out
