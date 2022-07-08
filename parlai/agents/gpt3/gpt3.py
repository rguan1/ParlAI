#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""

"""

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.message import Message
import openai  
import os

import re
import json
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import nltk
import argparse
import os
from nltk import tokenize
import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast




start_sequence = "\nAI:"
restart_sequence = "\nHuman:"
MAX_COMPLETION_LENGTH = 150 


def helper_split_words(Text, numberOfWords=1):
    if (numberOfWords > 1):
        text = Text.lstrip()
        pattern = '(?:\S+\s*){1,'+str(numberOfWords-1)+'}\S+(?!=\s*)'
        x =re.findall(pattern,text)
    elif (numberOfWords == 1):
        x = Text.split()
    else: 
        x = None
    return x



def split_on_sentences_word_count(text, word_count = 100):
    text = re.sub(r'(?<=[.,?!])(?=[^\s])', r' ', text)
    # text = re.sub(r'\.(?=[^ \W\d])', '. ', text)
    sent_splits = tokenize.sent_tokenize(text)
    ## now we should add sent until we are about to exceed 100 lines
    
    #go through each string and split if word count exceeds word_count
    
    word_count_chunks = []
    for s in sent_splits:
        num_words = len(s.split())
        if num_words > word_count:
            #then sentence already exceeds limit
            #split it
            word_count_chunks += helper_split_words(s, word_count)
        else:
            word_count_chunks.append(s)
                
    return_lines = []
    hold = ''

    for s in word_count_chunks:
        if len(hold.split()) + len(s.split()) > word_count:
            # then reset
            return_lines.append(hold)
            hold = ''
        else:
            hold += s
            hold += ' '
            
    if len(hold) > 0:
        return_lines.append(hold)
                
    # check that we split text properly < 128
    
    for l in return_lines:
        if len(l.split()) > word_count:
            print(word_count)
            print("LINE detected with a proble!!!")
            print(l)
            raise Exception("bad line!")
            
    return return_lines


def create_prime_within_gpt3_token_limit(instructions, article, turns):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    GPT3_TOKEN_LIMIT = 2049

    instr_tokens = tokenizer(instructions)['input_ids']
    turns_tokens = tokenizer(turns)['input_ids']

    tokens_left = GPT3_TOKEN_LIMIT - len(instr_tokens) - len(turns_tokens) - MAX_COMPLETION_LENGTH - 10

    article_formatted = re.sub(r'(?<=[.,?!])(?=[^\s])', r' ', article)
    sent_splits = tokenize.sent_tokenize(article_formatted)

    for i in range(len(sent_splits) - 1, -1, -1):
        subset_article = " ".join(sent_splits[0:i])
        article_cut = tokenizer(subset_article)['input_ids']
        if len(article_cut) < tokens_left:
            # we can use this subset of the article
            return prompt_compose(instructions, subset_article, turns)
    
    raise Exception("Failed to create proper prime")

    

def query_completion_api(prompt, engine='text-davinci-001'):
    response = openai.Completion.create(
        engine=engine, 
        prompt=prompt, 
        temperature=0.9, 
        max_tokens=MAX_COMPLETION_LENGTH, 
        top_p=1, 
        frequency_penalty=1, 
        presence_penalty=0.6, 
        stop=["###"]
    )
    
    return response

def prompt_compose(instructions, article_info, seed_turns):
    return f'{instructions}\nArticle: {article_info}\n{seed_turns}'

class FakeHistory:
    def __init__(self, gpt3):
        self.gpt3 = gpt3
    
    def add_reply(self, text):
        self.gpt3.turns += f"{start_sequence} {text}"

    def get_history_str(self):
        return f"{self.gpt3.article_info} {self.gpt3.turns}"



class Gpt3Agent(Agent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        group = parser.add_argument_group('GPT3 Arguments')
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'Gpt3Agent'
        self.turns = ''
        self.history = FakeHistory(self)
        self.article_info = ""
        
        # openai.api_key = os.getenv("GPT3_KEY")  
        openai.api_key = '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'



    def reset(self):
        """
        Reset the agent, clearing its observation.

        Many subclasses implement additional reset logic.
        """
        # self.observation = None
        
        self.turns = ''
        self.article_info = ''

    def observe(self, observation):
        """
        Receive an observation/action dict. Store it into
        """
        self.observation = observation

        self.turns += f"{restart_sequence} {observation['text']}"

        return observation

    def act(self):
        """
        Generate response to last seen observation.

        Replies with a message from using the Alice bot.

        :returns: message dict with reply
        """

        obs = self.observation
        
        instr = "An empathetic AI will discuss a news article. The AI should be be empathetic, kind, and emotional. The AI should have read the following news article below."

        self.turns += f"{start_sequence} "


        p = create_prime_within_gpt3_token_limit(instr, self.article_info, self.turns)

        resp = query_completion_api(p)
        resp_txt = resp.choices[0].text 

        resp_txt = resp_txt.strip()

        self.turns += f"{resp_txt}"
        
        return Message(
            {'id': self.getID(), 'text': resp_txt, 'episode_done': False}
        )
