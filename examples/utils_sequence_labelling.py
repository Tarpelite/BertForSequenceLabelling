# coding=utf-8
# Copyright 2019 Microsoft Research Team Authors

"""Load KeyPhrase dataset"""

from __future__ import absolute_import, division, print_function


import json
import logging
import math
import collections
from tqdm import *
import six
import re
from io import open
from nltk import word_tokenize
from nltk.stem import PorterStemmer


from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

stemmer = PorterStemmer()

def hit(str1, str2):
    str1_tokens = [stemmer.stem(x) for x in word_tokenize(str1)]
    str2_tokens = [stemmer.stem(x) for x in word_tokenize(str2)]
    if len(str1_tokens) != len(str2_tokens):
        return False
    else:
        for str1_token, str2_token in zip(str1_tokens, str2_tokens):
            if str1_token != str2_token:
                return False
        return True

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class RankExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class RankFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class RankPreProcessor(object):

    def __init__(self):
        pass

    def get_train_examples(self, in_examples, keyphrase_list):
        assert len(in_examples) == len(keyphrase_list)
        i = 1
        examples = []
        for exp, keys in zip(tqdm(in_examples), tqdm(keyphrase_list)):
            guid = "train-%s" % (i)
            true_keywords = exp.keywords
            doc_tokens = exp.doc_tokens
            text_a = " ".join(doc_tokens)
            for pred in keys:
                text_b = pred
                label = 0
                for target in true_keywords:
                    if hit(pred, target):
                        label = 1
                        break
                examples.append(
                    RankExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                )
            i += 1
        return examples



def convert_examples_to_rank_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, show_info=True):
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing  rank example %d of %d" % (ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label
        if ex_index < 5 and show_info:
            logger.info("*** Rank Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        
        features.append(
                RankFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features
                        


class LabellingExample(object):

    def __init__(self,
                label_id,
                doc_tokens,
                keywords,
                label_seq=None,
                pos_tag_seq=None,
                weights_seq=None):
        self._id = label_id
        self.doc_tokens = doc_tokens
        self.keywords = keywords
        self.label_seq = label_seq
        self.pos_tag_seq = pos_tag_seq
        self.weights_seq = weights_seq
    
    def __str__(self):
        return self.__repr__
    
    def __repr__(self):
        s = ""
        s += "_id: %s " % (self._id)
        s += "doc_tokens:[%s]" % (" ".join(self.doc_tokens))
        if self.label_seq:
            s += "label_seq: [%s]" % (" ".join([str(x) for x in self.label_seq]))
        if self.pos_tag_seq:
            s += "pos_tag_seq: [%s]" % (" ".join([str(x) for x in self.pos_tag_seq]))
        if self.weights_seq:
            s += "weights_seq: [%s]" % (" ".join([str(x) for x in self.weights_seq]))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                  unique_id,
                  example_index,
                  tokens,
                  token_to_orig_map,
                  input_ids,
                  input_mask,
                  segment_ids,
                  label_seq=None,
                  pos_tag_seq=None,
                  weights_seq=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_seq = label_seq
        self.pos_tag_seq = pos_tag_seq
        self.weights_seq = weights_seq
    
def read_label_examples(input_file):
    '''
    File format: 
    One line inculdes:
    doc \t  keywords(;) \t  label_seq \t pos_tag_seq \t weights_seq \n  
    '''
    examples = []
    with open(input_file, "r", encoding='utf-8') as reader:
        for label_id, line in enumerate(reader):
            sentences = line.split('\t')
            doc_tokens = word_tokenize(sentences[0].lower())
            label_seq = [int(item) for item in sentences[2].strip().split(' ')]
            keywords = list(sentences[1].split(';'))
            assert len(doc_tokens) == len(label_seq)
            if len(sentences) == 4:
                pos_tag_seq = [int(item) for item in sentences[3].strip().split(' ')]
                example = LabellingExample(
                    label_id=label_id,
                    doc_tokens=doc_tokens,
                    keywords=keywords,
                    label_seq=label_seq,
                    pos_tag_seq=pos_tag_seq
                )
            elif len(sentences) == 5:
                pos_tag_seq = [int(item) for item in sentences[3].strip().split(' ')]
                weights_seq = [float(item) for item in sentences[4].strip().split(' ')]
                example = LabellingExample(
                    label_id=label_id,
                    doc_tokens=doc_tokens,
                    keywords=keywords,
                    label_seq=label_seq,
                    pos_tag_seq=pos_tag_seq,
                    weights_seq=weights_seq
                )


            else:
                example = LabellingExample(
                    label_id=label_id,
                    doc_tokens=doc_tokens,
                    keywords=keywords,
                    label_seq=label_seq
                )
            examples.append(example)
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride,
                                max_query_length):
    
    unique_id = 1000000000
    features = []

    for (example_index, example) in enumerate(tqdm(examples)):
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -2 accounts for [CLS], [SEP]
        max_tokens_for_doc = max_seq_length - 2

        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        label_seq = []
        pos_tag_seq = []
        weights_seq=[]

        tokens.append("[CLS]")
        segment_ids.append(0)
        label_seq.append(0)
        pos_tag_seq.append(0)
        weights_seq.append(0)


        
        for i in range(min(len(all_doc_tokens), max_tokens_for_doc)):
            split_token_index = i
            orig_index = tok_to_orig_index[split_token_index]
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            if example.label_seq[orig_index] == 1:
                label_seq.append(1)
            else:
                label_seq.append(0)
            
            if example.pos_tag_seq[orig_index] > 0:
                pos_tag_seq.append(example.pos_tag_seq[orig_index])
            else:
                pos_tag_seq.append(0)
            
            weights_seq.append(example.weights_seq[orig_index])

            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(0)
        
        tokens.append("[SEP]")
        segment_ids.append(0)
        label_seq.append(0)
        pos_tag_seq.append(0)
        weights_seq.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_seq.append(0)
            pos_tag_seq.append(0)
            weights_seq.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_seq) == max_seq_length
        assert len(pos_tag_seq) == max_seq_length
        assert len(weights_seq) == max_seq_length

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("tokens: %s" % " ".join(
                [x for x in tokens]))
            logger.info("token_to_orig_map: %s" % " ".join(
                ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
            logger.info("token_is_max_context: %s" % " ".join([
                "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
            ]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                "tok_to_orig_index: %s" % " ".join([str(x) for x in tok_to_orig_index]))
            logger.info(
                "label seq: %s" % " ".join([str(x) for x in label_seq]))
            logger.info(
                "pos tag seq: %s" % " ".join([str(x) for x in pos_tag_seq]))
            logger.info(
                "weights seq: %s" % " ".join([str(x) for x in weights_seq])
            )

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_seq=label_seq,
                pos_tag_seq=pos_tag_seq,
                weights_seq=weights_seq))
        unique_id += 1

    return features



