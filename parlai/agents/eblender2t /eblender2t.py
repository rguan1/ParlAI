#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension

See https://arxiv.org/abs/1910.13461.

The BART agent can be instantiated as simply `-m bart`,
however it is recommended to specify `--init-model zoo:bart/bart_large/model`
or `-mf zoo:bart/bart_large/model` to ensure correct dictionaries are saved.
"""
import os
import torch
from typing import Optional, Dict, Any

from parlai.agents.bart.convert_fairseq_to_parlai import ConversionScript
from parlai.agents.bart.modules import BartModel
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.agents import compare_init_model_opts
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_agent import History
from parlai.utils.typing import TShared
from parlai.utils.io import PathManager
from parlai.zoo.bart.build import download, CONVERSION_ARGS, BART_ARGS

from parlai.core.params import ParlaiParser
from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
from parlai.core.metrics import SumMetric, AverageMetric, FairseqBleuMetric
from parlai.utils.fp16 import FP16SafeCrossEntropy
import parlai.utils.fsdp as fsdp_utils
from parlai.utils.torch import (
    neginf,
    total_parameters,
    trainable_parameters,
    PipelineHelper,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# import psutil


class Eblender2tAgent(TransformerGeneratorAgent):
    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__(opt, shared)
        # empathy_path = '/home/rg4312/thesis/evaluators/finetuned_output_models/empathy/fully_trained_roberta'
        # self.empathy_model = RobertaWrapper(empathy_path)
        # print(psutil.cpu_percent())
        # print(psutil.virtual_memory())  # physical memory usage
        # print('memory used:', psutil.virtual_memory()[2])

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.

        Easily overridable for customized loss functions.

        If return_output is True, the full output from the call to self.model()
        is also returned, via a (loss, model_output) pair.
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, *_ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        loss = self.criterion(score_view, batch.label_vec.view(-1))
        loss = loss.view(scores.shape[:-1]).sum(dim=1)


        # clamp_min = 1e-6 if self.opt['fp16'] else 1e-20
        maxlen = self.label_truncate or 256
        with torch.no_grad():
            beam_pred_scores, rest = self._generate(batch, self.beam_size, maxlen)

        generations = [g[1:] for (g, s, _) in beam_pred_scores]
        pred_toks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True)
        text = [self._v2t(p) for p in pred_toks]

        empathy_path = '/home/rg4312/thesis/evaluators/finetuned_output_models/empathy/fully_trained_roberta'
        self.empathy_model = RobertaWrapper(empathy_path)
        empathy_predictions = torch.FloatTensor(self.empathy_model.predict(text))
        target_empathy = torch.ones(empathy_predictions.size()) * 5

        mseloss = nn.MSELoss()
        empathy_loss = mseloss(empathy_predictions, target_empathy)

        empathy_loss /= 1 # reduce magnitude of the loss.

        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)
        correct = ((batch.label_vec == preds) * notnull).sum(dim=-1)
        



        # # cross entropy loss
        # self.record_local_metric('loss', AverageMetric.many(additional_loss, target_tokens))
        # perplexity
        self.record_local_metric('ppl', PPLMetric.many(loss, target_tokens))
        # token-wise accuracy
        self.record_local_metric(
            'token_acc', AverageMetric.many(correct, target_tokens)
        )
        # utterance-wise exact match
        self.record_local_metric(
            'token_em', AverageMetric.many(correct == target_tokens)
        )
        # actually do backwards loss
        
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token

        loss /= 2 #reduce tok loss magnitude

        #ryan change
        loss += empathy_loss

        num_toks = len(target_tokens.tolist())
        
        fake_loss = torch.FloatTensor([loss] * num_toks)
        fake_tar_toks = torch.FloatTensor([1] * num_toks)


        # cross entropy loss
        self.record_local_metric('loss', AverageMetric.many(fake_loss, fake_tar_toks))

        # print("~~~~~~~~~~~")
        # print(loss)
        if return_output:
            return (loss, model_output)
        else:
            return loss


class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())


class RobertaWrapper():
    def __init__(self, turn_level_path):
        self.tokenizer = RobertaTokenizer.from_pretrained(turn_level_path)
        self.model = RobertaForSequenceClassification.from_pretrained(turn_level_path)
        self.model.eval()
    
    def predict_single(self, phrase):
        inputs = self.tokenizer(phrase, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        
        return outputs[1]
    
    
    def predict(self, phrases):
        inputs = self.tokenizer(phrases, return_tensors="pt", padding=True)
        labels = torch.tensor([len(phrases)]).unsqueeze(0)  # Batch size of the input list
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        
        # unpack from tensors        
        return list(map(lambda x: x[0].item(), outputs[1]))

