import tensorflow as tf

import model

def top_k_logits(logits, k):
    """
    Top-k 采样策略。
    将概率最大的 k 个 token 之外的所有 token 的 logits 设为负无穷，
    从而在 softmax 后概率为 0。这确保了只从最可能的 k 个词中采样。
    """
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10, # 负无穷
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """
    Nucleus (Top-p) 采样策略。
    选择累积概率达到 p 的一组最小候选词集合。
    相比 top-k，这种方法可以动态调整候选词的数量。
    """
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        # 找到累积概率刚超过 p 的位置
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=1):
    """
    执行自回归生成（采样）过程。
    
    参数:
    hparams: 模型超参数
    length: 生成序列的长度
    start_token: 起始 token ID (通常是 <|endoftext|>)
    context: 已有的上下文 token IDs
    temperature: 温度参数，控制生成的随机性
    top_k: top-k 采样参数
    top_p: top-p 采样参数
    """
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        """单步推理：根据当前 tokens 和过去的缓存 (past) 计算 logits"""
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            # 1. 模型前向传播
            # prev: 上一个时间步生成的 token
            # past: 过去所有层的 KV 缓存
            next_outputs = step(hparams, prev, past=past)
            
            # 2. 获取最后一个 token 的 logits 并除以温度
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            
            # 3. 应用采样策略 (Top-k 和 Top-p)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            
            # 4. 从分布中采样下一个 token
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            
            # 5. 更新状态
            # 将新的 KV 拼接进 past，将新生成的 token 拼接到 output
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]

        # 初始化循环变量
        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        # 使用 tf.while_loop 进行高效的自回归生成
        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past, # 持续更新的 KV Cache
                prev, # 上一步生成的 Token
                output # 完整的生成序列
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False, # 推理阶段不需要反向传播
        )

        return tokens
