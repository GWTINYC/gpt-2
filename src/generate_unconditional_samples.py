#!/usr/bin/env python3

import fire  # 用于将 Python 函数自动转换为命令行接口 (CLI) 的库
import json  # 用于处理 JSON 格式的数据（如加载配置文件）
import os    # 用于与操作系统交互（如路径处理、环境变量）
import numpy as np  # 数值计算库，用于处理数组和随机数
import tensorflow as tf  # 深度学习框架，用于构建和运行模型

import model, sample, encoder  # 导入项目内的模块：模型定义、采样逻辑、编码器

def sample_model(
    model_name='124M',     # 指定要使用的预训练模型名称（如 124M, 355M, 774M, 1558M）
    seed=None,             # 设置随机数种子，如果为了复现结果可以固定此值
    nsamples=0,            # 总共生成的样本数量。如果设为 0，则无限循环生成，直到手动停止
    batch_size=1,          # 批大小，即一次并行生成多少个样本（受显存/内存限制）
    length=None,           # 每个样本生成的文本长度（Token 数）。None 表示使用模型最大允许长度
    temperature=1,         # 采样温度。值越低分布越尖锐（更确定），值越高分布越平坦（更随机）
    top_k=0,               # Top-k 采样参数。0 表示不使用。若 >0，仅从概率最高的 k 个词中采样
    top_p=1,               # Nucleus (Top-p) 采样参数。1 表示不使用。否则仅从累积概率达到 p 的最小集合中采样
    models_dir='models',   # 模型文件所在的父目录
):
    """
    运行无条件文本生成模型。
    即不给定任何提示词（Prompt），让模型"凭空"（通常从 <|endoftext|> 开始）生成文本。
    """
    # 扩展路径中的用户目录符号（如 ~）和环境变量，确保路径正确
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    
    # 1. 加载编码器 (BPE Tokenizer)
    # get_encoder 会加载 encoder.json 和 vocab.bpe，用于将文本转换为 ID 以及将 ID 还原为文本
    enc = encoder.get_encoder(model_name, models_dir)
    
    # 2. 加载并设置模型超参数
    # 首先获取默认的超参数对象
    hparams = model.default_hparams()
    # 读取模型目录下的 hparams.json 文件，覆盖默认参数（如层数、头数等）
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    # 如果未指定生成长度，则默认为模型的最大上下文长度 (n_ctx)
    if length is None:
        length = hparams.n_ctx
    # 如果指定的长度超过了模型支持的最大长度，抛出错误
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    # 3. 创建 TensorFlow 会话 (Session)
    # 创建一个新的图 (Graph) 并在会话中运行。这是 TF 1.x 的标准做法
    with tf.Session(graph=tf.Graph()) as sess:
        # 设置 Numpy 的随机种子
        np.random.seed(seed)
        # 设置 TensorFlow 的随机种子
        tf.set_random_seed(seed)

        # 4. 构建采样计算图 (Computation Graph)
        # sample_sequence 函数会在图中定义模型的前向传播结构
        # start_token 设为 <|endoftext|>，表示从文本结束符开始（即无条件生成）
        output = sample.sample_sequence(
            hparams=hparams, 
            length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )[:, 1:] # 输出结果切片，去掉第一个 token（即 start_token <|endoftext|> 本身），只保留生成的内容

        # 5. 准备加载模型权重
        # 创建 Saver 对象，用于恢复模型变量
        saver = tf.train.Saver()
        # 查找模型目录中最新的检查点文件 (checkpoint)
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        # 将磁盘上的模型权重加载到当前的 Session 中
        saver.restore(sess, ckpt)

        # 6. 开始循环生成文本
        generated = 0 # 记录已生成的样本总数
        # 如果 nsamples 为 0，条件永远为 True (无限生成)；否则生成达到指定数量后停止
        while nsamples == 0 or generated < nsamples:
            # 运行计算图节点 'output'。这一步执行实际的模型推理计算。
            # out 的形状是 [batch_size, length]
            out = sess.run(output) 
            
            # 遍历当前批次生成的每一个样本
            for i in range(batch_size):
                generated += batch_size # 更新计数
                # 将生成的 Token ID 序列解码回人类可读的文本字符串
                text = enc.decode(out[i]) 
                # 打印分割线和样本编号
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                # 打印生成的文本内容
                print(text)

# 如果脚本作为主程序运行，使用 fire 库将 sample_model 函数暴露为命令行工具
if __name__ == '__main__':
    fire.Fire(sample_model)

