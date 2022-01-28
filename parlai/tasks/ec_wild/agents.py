#!/usr/bin/env python3

import os
from typing import Any, List
import numpy as np
import copy

from parlai.utils.io import PathManager
from parlai.core.teachers import ParlAIDialogTeacher

class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)

        # get datafile
        opt['parlaidialogteacher_datafile'] = _path(opt, '')

        super().__init__(opt, shared)

def _path(opt, filtered):
    # set up path to data (specific to each dataset)

    # YOUR_PATH = "/scratch/as11919/empathic-conversations-chatbot/data/empathic_conversations_"  # <--------------
    
    YOUR_PATH =  os.path.join(opt['datapath'], 'ec_wild','ec_wild_')

    dt = opt['datatype'].split(':')[0]
    final_path = YOUR_PATH + dt + '.txt'
    return (final_path)