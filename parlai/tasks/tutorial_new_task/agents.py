from parlai.core.teachers import ParlAIDialogTeacher
import copy
import os

class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)

        # get datafile
        opt['parlaidialogteacher_datafile'] = _path(opt, '')

        super().__init__(opt, shared)
        
def _path(opt, filtered):
    # build the data if it does not exist
    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    # return os.path.join(opt['datapath'], 'conversation_only', dt + '.json')
    return os.path.join(opt['datapath'], 'conversation_only','conversation_only.json')
    