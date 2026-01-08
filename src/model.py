import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    """返回默认的超参数对象"""
    return HParams(
        n_vocab=0,       # 词汇表大小
        n_ctx=1024,      # 上下文窗口大小（序列长度）
        n_embd=768,      # 嵌入层维度（隐藏层大小）
        n_head=12,       # 注意力头的数量
        n_layer=12,      # Transformer 层数（Block 数量）
    )

def shape_list(x):
    """
    优雅地处理 TensorFlow 中的动态形状。
    返回张量形状的列表，静态维度保持为整数，动态维度保持为 tf.Tensor。
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    """自定义 Softmax 函数"""
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    """
    GELU (Gaussian Error Linear Unit) 激活函数。
    它是 ReLU 的平滑版本，常用于 Transformer 模型（如 GPT-2, BERT）。
    近似公式：0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """
    层归一化 (Layer Normalization)。
    将输入归一化为均值 0，方差 1，然后进行对角仿射变换 (scale * x + bias)。
    """
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1)) # 缩放参数 (Gain)
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0)) # 偏置参数 (Bias)
        u = tf.reduce_mean(x, axis=axis, keepdims=True) # 计算均值
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True) # 计算方差
        x = (x - u) * tf.rsqrt(s + epsilon) # 归一化
        x = x*g + b # 仿射变换
        return x

def split_states(x, n):
    """
    将 x 的最后一个维度重塑为 [n, x.shape[-1]/n]。
    通常用于将多头注意力的维度分离出来。
    """
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """
    将 x 的最后两个维度合并为一个维度。
    通常用于在多头注意力计算后合并结果。
    """
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    """
    一维卷积操作（实际上等同于全连接层/线性投影）。
    x: 输入张量
    scope: 变量作用域名称
    nf: 输出特征维度 (number of filters)
    w_init_stdev: 权重初始化的标准差
    """
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        # 将输入重塑为 2D 矩阵进行矩阵乘法，然后再重塑回原状
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """
    生成注意力掩码，用于因果注意力（Causal Attention）。
    确保位置 i 只能注意到位置 <= i 的信息（即下三角矩阵为 1）。
    nd: 目标序列长度
    ns: 源序列长度
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    """
    多头自注意力机制 (Multi-Head Self-Attention)。
    """
    assert x.shape.ndims == 3  # 输入形状应为 [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # 应为 [batch, 2, heads, sequence, features]，其中 2 代表 [k, v]

    def split_heads(x):
        # 将 [batch, sequence, features] 转换为 [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # split_heads 的逆操作
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # 掩盖注意力权重，实现因果性。
        # w 形状: [batch, heads, dst_sequence, src_sequence]
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b) # 将被掩盖的位置设为极小的负数
        return w

    def multihead_attn(q, k, v):
        # 执行多头注意力计算
        # q, k, v 形状: [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True) # 计算 Q * K^T
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype)) # 缩放点积 (Scaled Dot-Product)

        w = mask_attn_weights(w) # 应用掩码
        w = softmax(w) # 计算 Attention Score
        a = tf.matmul(w, v) # 计算加权和 Sum(Score * V)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3) # 投影生成 Q, K, V
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1) # 保存当前的 K, V 用于下一次推理的 past
        if past is not None:
            # 如果有过去的 K, V (用于增量推理)，则将其拼接
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a) # 合并多头
        a = conv1d(a, 'c_proj', n_state) # 输出投影
        return a, present


def mlp(x, scope, n_state, *, hparams):
    """
    多层感知机 (Feed Forward Network)。
    包含两层全连接层，中间使用 GELU 激活函数。
    """
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state)) # 第一层投影 + 激活
        h2 = conv1d(h, 'c_proj', nx) # 第二层投影
        return h2


def block(x, scope, *, past, hparams):
    """
    Transformer 的一个 Block（层）。
    包含 Attention 层和 MLP 层，以及 Residual Connection 和 Layer Norm。
    """
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        # 1. Self-Attention 部分
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a # 残差连接
        # 2. MLP 部分
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m # 残差连接
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    """计算 'past' 张量的形状 [batch, layers, 2(k,v), heads, sequence, features_per_head]"""
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """将给定值扩展维度并平铺"""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    """生成位置编码的索引"""
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=False):
    """
    定义整个 GPT-2 模型。
    hparams: 超参数
    X: 输入 Token ID 序列
    past: 过去的 K, V 缓存（用于加速生成）
    scope: 变量作用域
    reuse: 是否复用变量
    """
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        # 词嵌入 (WTE)
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        # 位置嵌入 (WPE)
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        
        # 将 Token 嵌入和位置嵌入相加
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer 堆叠层
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        # 保存所有层的 K, V 用于下次迭代
        results['present'] = tf.stack(presents, axis=1)
        
        # 最终的 Layer Norm
        h = norm(h, 'ln_f')

        # 语言模型 Head (计算 Logits)
        # 目标是预测下一个 token，这里复用了 wte 矩阵 (Weight Tying)
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
