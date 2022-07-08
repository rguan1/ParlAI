from typing import List
from xmlrpc.client import Boolean
from parlai.core.agents import Agent


def add_history(agent : Agent, chat_list : List, is_first: Boolean):
    ### Chat list consists of [p1, p2, convo_starter]
    p1 = chat_list[0]
    p2 = chat_list[1]
    convo_starter = chat_list[2]

   # special logic for specific gpt3 priming method
    if "Gpt3Agent" in agent.id:
        #then it is a gpt3Agent model and we just send the article in
        if is_first:
            agent.article_info = " ".join(p1)
        else:
            agent.article_info = " ".join(p2)
    else:
        if is_first:
            #first person seeding
            for index, text in enumerate(p1):
                agent.history.add_reply(text)
        else:
            for index, text in enumerate(p2):
                agent.observe({'text' : text, 'episode_done': False})


    for index, text in enumerate(convo_starter):
        if index % 2 == 0:
            agent.history.add_reply(text)            
        else:
            agent.observe({'text' : text, 'episode_done': False})



def print_history(agent: Agent):
    print("~~~~~~~~~~~~History Start~~~~~~~~~~~~~~")
    print(agent.history.get_history_str())
    print("~~~~~~~~~~~~History End~~~~~~~~~~~~~~")
