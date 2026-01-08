"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    GPT-2 的核心技巧之一：Byte-Level BPE。
    为了不直接处理 256 个字节（这会导致一些控制字符问题），也不使用庞大的 Unicode 词表（会导致 embedding 巨大），
    该函数构建了一个从 utf-8 字节到可打印 unicode 字符的双射映射。
    
    Returns:
        dict: 字节整数 -> Unicode 字符的映射
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    # 处理剩下的不可打印字符，映射到 256 之后的 unicode 字符上，保证不重复
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    返回单词中的所有相邻符号对。
    例如 word=("h", "e", "l", "l", "o") -> {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')}
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder # Token -> ID 映射
        self.decoder = {v:k for k,v in self.encoder.items()} # ID -> Token 映射
        self.errors = errors # 解码错误处理策略
        self.byte_encoder = bytes_to_unicode() # Byte -> Unicode 映射
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()} # Unicode -> Byte 映射
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges)))) # 合并规则及其优先级
        self.cache = {} # BPE 结果缓存

        # 正则表达式用于预分词，将文本拆分为单词、数字、标点等基础单元
        # 包含了对缩写（如 's, 't）的特殊处理
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """
        对单个 token 应用 BPE 合并规则。
        例如将 "hello" 逐步合并为 "he" "ll" "o" -> "hell" "o" -> "hello" (假设词表中有这些词)
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # 找到优先级最高的相邻对（rank 值越小优先级越高）
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        将文本转换为 Token ID 序列。
        过程: Text -> Pre-tokenization -> Byte Mapping -> BPE -> Token IDs
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 将 UTF-8 字节转换为可打印的 Unicode 字符
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 应用 BPE 合并，然后查找 Token ID
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        将 Token ID 序列还原为文本。
        过程: Token IDs -> Tokens -> Byte Mapping (Reverse) -> UTF-8 String
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(model_name, models_dir):
    """加载指定模型的 Encoder 和 BPE 词表"""
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
