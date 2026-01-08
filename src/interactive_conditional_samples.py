#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    运行交互式条件生成模型。
    允许用户输入提示词 (Prompt)，模型基于提示词续写文本。
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    # 1. 加载编码器
    enc = encoder.get_encoder(model_name, models_dir)
    # 2. 加载超参数
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    # 3. 创建会话
    with tf.Session(graph=tf.Graph()) as sess:
        # 定义输入占位符，因为这次有 Prompt 输入
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        # 4. 构建采样计算图
        # 这里传入了 context 作为条件
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        # 5. 恢复权重
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        # 6. 进入交互循环
        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            
            # 将用户输入的文本编码为 Token IDs
            context_tokens = enc.encode(raw_text)
            
            generated = 0
            for _ in range(nsamples // batch_size):
                # 运行模型生成及其后续 token
                # 使用 feed_dict 将编码后的 Prompt 传入 context 占位符
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):] # 截取掉输入的 Prompt 部分，只保留生成部分
                
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i]) # 解码生成的文本
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

