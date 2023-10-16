import gc
import glob
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
from collections import Counter
from os.path import join as pjoin

import torch
from multiprocess import Pool

from others.logging import logger

import gluonnlp as nlp
# from kobert.utils import get_tokenizer 
# from kobert.utils import download as _download
# from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer

kobert_models = {
    'mxnet_kobert': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/mxnet/mxnet_kobert_45b6957552.params',
        'fname': 'mxnet_kobert_45b6957552.params',
        'chksum': '45b6957552'
    },
    'vocab': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/vocab/kobertvocab_f38b8a4d6d.json',
        'fname': 'kobertvocab_f38b8a4d6d.json',
        'chksum': 'f38b8a4d6d'
    }
}


def get_kobert_vocab(cachedir="./tmp/"):
    # Add BOS,EOS vocab
    vocab_info = kobert_models['vocab']                
    vocab_file = _download(
        vocab_info["url"], vocab_info["fname"], vocab_info["chksum"], cachedir=cachedir
    )

    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
        vocab_file, padding_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )

    return vocab_b_obj

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def greedy_selection(doc_sent_list, abstract, summary_size):
    """
    
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9가-힣▁ ]', '', s)

    max_rouge = 0.0
    abstract = _rouge_clean(' '.join(abstract)).split()    # join으로 string을 만든 뒤, 전처리 함수 후에 다시 split처리한다. 
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):               # 요약사이즈 만큼 반복문 default = 3
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):             # 전체 문장 수 만큼 가져오기
            if (i in selected):                 # i가 이미 selected에 있으면 건너뛰기
                continue
            c = selected + [i]                  # 없을 경우, [] + [0] 또는 [1,4,6] + [7]와 같이 리스트에 해당 idx를 추가하는 개념
            candidates_1 = [evaluated_1grams[idx] for idx in c]    # 해당 list에 들어가있는 idx를 빼서 각 1grams에서 뽑아 값을 리스트로 가짐
            candidates_1 = set.union(*map(set, candidates_1))      # 중복제거 및 합집합 표현
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']    # rouge 계산식에서 f1스코어를 가져옴
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']    # rouge 계산식에서 f1스코어를 가져옴
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:     # 현 rouge 스코어가 기존 rouge 스코어보다 높으면 갱신
                cur_max_rouge = rouge_score
                cur_id = i                      # 그리고 가장 rouge가 높아 갱신되었던 때의 문장 번호를 가져옴
        if (cur_id == -1):                      # 더이상 업데이트 사항이 없을 경우 현 selected 수집 내용을 반환
            return selected
        selected.append(cur_id)                 # cur_id 업데이트가 있었다면 (-1이 아니므로) selected에 추가
        max_rouge = cur_max_rouge               # 현재 rouge를 max로 설정

    return sorted(selected)                     # 가장 적합한 index 가져오기


class BertData:
    def __init__(self, args, vocab, tokenizer):
        self.args = args
        self.vocab = vocab
        self.tokenizer = tokenizer
        
        self.pad_idx = self.vocab["[PAD]"]
        self.cls_idx = self.vocab["[CLS]"]
        self.sep_idx = self.vocab["[SEP]"]
        self.mask_idx = self.vocab["[MASK]"]
        self.bos_idx = self.vocab["[BOS]"]
        self.eos_idx = self.vocab["[EOS]"]

    def preprocess(self, src, tgt, sent_labels, is_test=False):

        if ((not is_test) and len(src) == 0):   # test모드이고, src길이가 0일 경우 None 반환
            return None

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]  # src 길이가 지정한 문장 당 최소 토큰 수보다 클 경우, idxs에 indice list로 넣는다. 
        _sent_labels = [0] * len(src) # 문장 라벨 0으로 고정 
        for l in sent_labels:         # 기존 주요 문장 index는 1로 변경하여 라벨링 
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]               # 문장 뽑기 - 지정 최대 토큰수까지만 슬라이싱
        sent_labels = [_sent_labels[i] for i in idxs]                                   # 주요 index 라벨 생성
        src = src[:self.args.max_src_nsents]                                            # 문서 개수 정리 - 지정 최대 문장 수 까지만 다룸
        sent_labels = sent_labels[:self.args.max_src_nsents]                            # 최대 문장 수까지만 라벨 생성

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        # tokenization.py
        src_token_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in src]          # src 각 문장별로 토큰 넘버링 진행
        src_subtoken_idxs = [self.add_special_token(lines) for lines in src_token_ids]  # 토큰 넘버링 문장 앞뒤에 cls와 sep를 달아줌
        segments_ids = self.get_token_type_ids(src_subtoken_idxs)                       # segment 홀짝 기준 0, 1 번갈아 넣음

        src_subtoken_idxs = [x for sublist in src_subtoken_idxs for x in sublist]       # 1차원 리스트로 통합 -> flatten과 동일 [] 2, 142, 4563, 342, 3, 2,....3
        segments_ids = [x for sublist in segments_ids for x in sublist]                 # 위와 동일한데, 1,0으로 문장 구분으로 이뤄짐

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt)[:self.args.max_tgt_ntokens]   # 지정 최대 토큰수 만큼 짤라서 타겟 토큰 넘버링 진행
        tgt_subtoken_idxs = self.add_sentence_token(tgt_subtoken_idxs)                              # 문장 앞뒤로 bos, eos 토큰 설정
    
        if ((not is_test) and len(tgt_subtoken_idxs) < self.args.min_tgt_ntokens):
            return None

        cls_ids = self.get_cls_index(src_subtoken_idxs)                 # CLS 토큰의 index를 list로 가져온다.
        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src, tgt   # 전체 문장 토큰 flatten, 가장 적합한 문장 index, 정답 토큰 flat, 세그먼트 flat, cls idx, 문장, 정답

    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def add_sentence_token(self, token_ids):
        return [self.bos_idx] + token_ids + [self.eos_idx]
    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab.
        추가된 함수
        """
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        # if len(ids) > self.max_len:
        #     raise ValueError(
        #         "Token indices sequence length is longer than the specified maximum "
        #         " sequence length for this BERT model ({} > {}). Running this"
        #         " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
        #     )
        return ids

    def get_token_type_ids(self, src_token):
        seg = []
        for i, v in enumerate(src_token):
            if i % 2 == 0:
                seg.append([0] * len(v))
            else:
                seg.append([1] * len(v))
        return seg

    def get_cls_index(self, src_doc):
        cls_index = [index for index, value in enumerate(src_doc) if value == self.cls_idx]
        return cls_index

def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
        # print('datasets:', datasets)
        # print('args.raw_path:', args.raw_path)
    for corpus_type in datasets:
        # print(pjoin(args.raw_path, '*' + corpus_type + '.jsonl'))
        # jsonl 파일을 들고 있어야 하며, PATH가 존재해야 함
        path = glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.jsonl'))[0]
        with open(path, "r", encoding="utf-8") as f:
            jsonl = list(f)
        data = []
        for json_str in jsonl:
            data.append(json.loads(json_str))               # jsonl 데이터를 풀어서 list에 적재함
            
        os.makedirs(args.json_path, exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)
        for i in range(math.ceil(len(data) / 2000)):
            tmp_path = args.json_path + os.path.splitext(path.split('/')[-1])[0] + f'.{i}.jsonl'
            print(tmp_path)

            if os.path.exists(tmp_path):
                logger.info('%s exsists, pass.' % tmp_path)
                continue
            with open(tmp_path, 'w', encoding='utf-8') as f:
                for d in data[i * 2000 : (i + 1) * 2000]:
                    json.dump(d, f, ensure_ascii=False)
                    f.write('\n')

        vocab = get_kobert_vocab()
        tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
        
        a_lst = []
        for json_f in glob.glob(pjoin(args.json_path, '*' + corpus_type + '.*.jsonl')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, vocab, tokenizer, pjoin(args.save_path, real_name.replace('jsonl', 'bert.pt'))))

        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass
        pool.close()
        pool.join()

def _format_to_bert(batch):
    corpus_type, json_file, args, vocab, tokenizer, save_file = batch
    is_test = corpus_type == 'test'                 # test모드 여부 확인
    if (os.path.exists(save_file)):                 # file존재 여부 확인
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args, vocab, tokenizer)         # Bert 소환

    logger.info('Processing %s' % json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        jsonl = list(f)                             # Json file을 list로 가져옴
    jobs = []
    for json_str in jsonl:
        jobs.append(json.loads(json_str))           # Json을 풀어서 리스트에 넣음

    datasets = []
    for d in jobs:                                  # 각 json으로부터 데이터 하나씩 토크나이징 하여 source와 ttq에 넣음 
        source, tgt = [tokenizer(s) for s in d['article_original']], tokenizer(d['abstractive']) 
        #  source 예시: ['홍 대표 자 , 서울 을 바꾸 려면 영혼 이 맑 은 후보 로 단일 화 를 해야 해']
        # 
        if args.use_anno_labels:            # 어노테이션 라벨이 있는경우,
            sent_labels = d['extractive']   # 문장라벨 변수에 추출요약 index 가져옴
        else: # 없을 경우, greedy selection 실행 -> sent_labels = 가장 적합한 문장 indice
            sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)    # max_src_nsent default = 100
        if (args.lower):
            source = [' '.join(s).lower().split() for s in source]
            tgt = ' '.join(tgt).lower().split()
        b_data = bert.preprocess(source, tgt, sent_labels, is_test=is_test) # source: 전체 문서, tgt: 타겟 문장, sent_labels: 가장 적합한 문장 idx

        if (b_data is None):
            continue

        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data 
        # 전체 문장 토큰 flatten, 가장 적합한 문장 index, 정답 토큰 flat, 세그먼트 flat, cls idx, 전체문장, 정답문장
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()