#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Basic script which allows local human keyboard input to talk to a trained model.

## Examples

```shell
parlai NextUtterance --model-file "zoo:tutorial_transformer_generator/model"
```

When prompted, enter something like: `Bob is Blue.\\nWhat is Bob?`

Input is often model or task specific. Some tasks will automatically format
The input with context for the task, e.g. `-t convai2` will automatically add
personas.
""" 
from xmlrpc.client import Boolean
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging

#ryan changes
from parlai.scripts.ryan_custom_scripts.agent_history_helpers import add_history, print_history


import random

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, 'next_utterance will insert chat history into chatbot and then ask chatbot for next response'
        )
        parser.add_argument(
            '--utterances_path',
            type=str,
            required=True,
            help='Path to txt file containing previous utterances that will be loaded into history. \
                  Each line of txt will be considered a separate utterance.',
        )
        parser.add_argument(
            '--is_bot_speaking_first',
            type=bool,
            default=False,
            help='Path to txt file containing previous utterances that will be loaded into history. \
                  Each line of txt will be considered a separate utterance.',
        )
    return parser


def next_utterance(opt):
    print(opt['is_bot_speaking_first'])
    if isinstance(opt, ParlaiParser):
        logging.error('next_utterance should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)
    agent.opt.log()

    prev_utterances = []
    with open(opt["utterances_path"]) as utter_file:
        for line in utter_file:
            prev_utterances.append(line.rstrip())


    add_history(agent, prev_utterances, opt['is_bot_speaking_first'])

    # Show some example dialogs:
    print_history(agent)
    print(agent.act()['text'])

class NextUtterance(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return next_utterance(self.opt)


if __name__ == '__main__':
    random.seed(42)
    NextUtterance.main()
