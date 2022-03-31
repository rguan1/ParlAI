#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Simple agent which always outputs the given fixed response.

Good for debugging purposes or as a baseline, e.g. if always predicting a given class.
"""

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.message import Message
import openai  
import os 


def query_completion_api(prompt, engine='text-davinci-001'):
    response = openai.Completion.create(
        engine=engine, 
        prompt=prompt, 
        temperature=0.9, 
        max_tokens=150, 
        top_p=1, 
        frequency_penalty=1, 
        presence_penalty=0.6, 
        stop=["###"]
    )
    
    return response

def prompt_compose(context, instructions, seed_turns):
    return f'{context}\n ###\n{instructions}\n{seed_turns}'


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
        self.sturns = ''
        
        openai.api_key = os.getenv("GPT3_KEY")  

    # def act(self):
    #     return Message(
    #         {'id': self.getID(), 'text': self.fixed_response, 'episode_done': False}
    #     )

    def act(self):
        """
        Generate response to last seen observation.

        Replies with a message from using the Alice bot.

        :returns: message dict with reply
        """

        obs = self.observation
        
        context = "This is conversation with an AI assistant"
        instr = "Chat with the other person"
        # sturns = "" if obs is None else obs 

        if obs is None:
            self.sturns += f"Human: \nAI: "
        else:
            self.sturns += f"Human: {obs['text']}\nAI: "

        p = prompt_compose(context, instr, self.sturns)
        resp = query_completion_api(p)
        resp_txt = resp.choices[0].text 

        self.sturns += f"{resp_txt}\n"
        

        reply = {}
        reply['id'] = self.getID()
        reply['text'] = resp_txt

        return reply