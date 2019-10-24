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

from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

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
            s += "weights_seq: [%s]" % (" ".join([str(x) for x in self.weights_seq])


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
    examples = []
    with open(input_file, "r", encoding='utf-8') as reader:
        for label_id, line in enumerate(reader):
            sentences = line.split('\t')
            doc_tokens = word_tokenize(sentences[0])
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
        tokens.append("[CLS]")
        segment_ids.append(0)
        label_seq.append(0)
        pos_tag_seq.append(0)


        
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

            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(0)
        
        tokens.append("[SEP]")
        segment_ids.append(0)
        label_seq.append(0)
        pos_tag_seq.append(0)

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

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_seq) == max_seq_length
        assert len(pos_tag_seq) == max_seq_length

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
                pos_tag_seq=pos_tag_seq))
        unique_id += 1

    return features



