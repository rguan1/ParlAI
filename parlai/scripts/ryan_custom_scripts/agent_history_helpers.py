from typing import List
from xmlrpc.client import Boolean
from parlai.core.agents import Agent


def add_history(agent : Agent, chat_list : List, is_first: Boolean):
    for index, text in enumerate(chat_list):
        if index % 2 == is_first:
            agent.observe({'text' : text, 'episode_done': False})
        else:
            agent.history.add_reply(text)

def print_history(agent: Agent):
    print("~~~~~~~~~~~~History Start~~~~~~~~~~~~~~")
    print(agent.history.get_history_str())
    print("~~~~~~~~~~~~History End~~~~~~~~~~~~~~")
