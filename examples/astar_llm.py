
# -*- coding: utf-8 -*-

def run(astarLLM):

    while astarLLM.cur_node != astarLLM.target_node:

        # Get the next action from the LLM agent
        tell = astarLLM.comm.ask(astarLLM.pre_node, astarLLM.cur_node)
        actions = astarLLM.extract_node(tell)

        # Update the current node based on the action taken
        astarLLM.opt_node_select(actions)

    return astarLLM.open_queue, astarLLM.close_queue