import config
import re
import copy
import openai

class communication:

    """
    This class is to communicate with the LLM (Language Model) for pathfinding tasks.   

    It initializes the model with the environment description, start and target nodes, and margin size.
    It also provides methods to initialize messages and ask the model for optimal actions based on the current and previous nodes. 

    """

    def __init__(self, key, model_name, env_description, max_margin, start_node, target_node):
        # 完整保留原始初始化逻辑
        self.openai_key = key  # 实际应通过config配置
        self.model_name = model_name
        self.env_discription = env_description
        self.start_node = start_node
        self.target_node = target_node
        self.max_margin = max_margin
        
        
    def message_initialize(self):

        """
        Initialize the map message for the LLM.
        :return: None
        """

        messages = [{'role': 'system', 'content': 'Glad to see you!'}]

        # user tell for grid world information
        grid_world_inf = "We have a " + str(self.max_margin) + "*" + str(self.max_margin) + " grid map in which there are totally " + str(int(self.max_margin * self.max_margin)) + " grids inside. In this map, we use [i, j] to represent the grid of ith row and jth column. Additionally, there is a binary number which is ‘1’ or ‘0’ in each grid to represent the obstacle conditions that ‘1’ means the grid is free whilst ‘0’ means the grid has been occupied by an obstacle and the agent cannot move to it. Our objective is to make an agent at the starting node to bypass obstacles and reach the designated target node. Considering the token limitation problem, please correctly and precisely remember the information above and do not return to much contexts for not only this but also the following prompt. Here is the detail information below. "
        print("User: ", grid_world_inf)
        messages.append({'role': 'user', 'content': grid_world_inf})

        chat_for_task = openai.ChatCompletion.create(model=self.model_name, messages=messages)
        reply_for_task = chat_for_task.choices[0].message.content
        print(f"ChatGPT: {reply_for_task}")
        messages.append({'role': 'assistant', 'content': reply_for_task})

        # tell the start_node and target_node:
        start_and_target = "Start node: " + str(self.start_node) + "; Target node: " + str(self.target_node) + " which is at the agent's lowerleft direction. "
        print("User: ", start_and_target)
        messages.append({'role': 'user', 'content': start_and_target}, )
        chat_for_task = openai.ChatCompletion.create(model=self.model_name, messages=messages)
        reply_for_task = chat_for_task.choices[0].message.content
        print(f"ChatGPT: {reply_for_task}")
        messages.append({'role': 'assistant', 'content': reply_for_task})

        # user tell the obstacle information.
        obstacles_inf = self.env_discription
        print("User: ", obstacles_inf)
        messages.append({'role': 'user', 'content': obstacles_inf}, )
        chat_for_task = openai.ChatCompletion.create(model=self.model_name, messages=messages)
        reply_for_task = chat_for_task.choices[0].message.content
        print(f"ChatGPT: {reply_for_task}")
        messages.append({'role': 'assistant', 'content': reply_for_task})

        # user tell the agent action space and distance standard
        action_inf = "Agent action space (set current coordinate of the agent is [i, j]): Action 0: move up ([i, j] -> [i-1, j]); Action 1: move down ([i, j] -> [i+1, j]); Action 2: move right ([i, j] -> [i, j+1]); Action 3: move left ([i, j] -> [i, j-1]); Action 4: move upper right [i, j] -> [i-1, j+1] Action 5: move upper left ([i, j] -> [i-1, j-1]); Action 6: move lower right ([i, j] -> [i+1, j+1]); Action 7: move lower left ([i, j] -> [i+1, j-1]). Notice: 1. In this map, row index i increases from upward direction to downward direction, whilst column index j increases from leftward direction to righward direction. 2. When the agent is located in a grid adjacent to an obstacle, some actions in the action space will not be achieved. Please dynamically adjust the agent's actions at runtime. Distance standard: Euclidean distance. Considering the token limitation problem, please correctly and precisely remember the information above and do not return to much contexts for not only this but also the following prompt."
        print("User: ", action_inf)
        messages.append({'role': 'user', 'content': action_inf}, )
        chat_for_task = openai.ChatCompletion.create(model=self.model_name, messages=messages)
        reply_for_task = chat_for_task.choices[0].message.content
        print(f"ChatGPT: {reply_for_task}")
        messages.append({'role': 'assistant', 'content': reply_for_task})


        # user tell the objective
        obj_inf = "Our objective is to move the agent step by step with the actions above bypassing the obstacles from the start node to the target node. Considering the token limitation problem, please correctly and precisely remember the information above and do not return to much contexts for not only this but also the following prompt."
        print("User: ", obj_inf)
        messages.append({'role': 'user', 'content': obj_inf}, )
        chat_for_task = openai.ChatCompletion.create(model=self.model_name, messages=messages)
        reply_for_task = chat_for_task.choices[0].message.content
        print(f"ChatGPT: {reply_for_task}")
        messages.append({'role': 'assistant', 'content': reply_for_task})

        self.ini_massages = copy.deepcopy(messages)


    def ask(self, pre_node, cur_node):

        """
        Ask the LLM to give the action space based on the current node and the previous node.
        :param pre_node: The previous node of the agent.
        :param cur_node: The current node of the agent.
        :return: The recommended action space of the agent.
        """

        massages = copy.deepcopy(self.ini_massages)

        ask_opt_actions_inf = "Now, the agent has moved from " + str(pre_node) + " to " + str(cur_node) +" which is the current position. Considering the current node the agent stays at above, based on the information of target nodes obstacles, action space and our objective, please for only the next step return me a subset of action space in which there are several optimal actions satisfied with the factors below: 1. Take care the obstacle regions I told you. Please always be careful do not move to them. 2. Take care the action space which includes at most 8 actions allowed, especially for some cells adjacent to the obstacles. 3. All of the optimal actions should serve for the purpose ‘achieve and reach the target node’. The agent must move in the correct direction while avoiding obstacles. Please prudentially deal with the causality relationship among obstacles and the correct direction and dinamically adjust and improve the solution. After selecting actions, please put the corresponding action numbers into a list and come out it. Considering the token limitation problem, please correctly and precisely remember the information above and do not return to much contexts for not only this but also the following prompt."
        print("User: ", ask_opt_actions_inf)
        massages.append({'role': 'user', 'content': ask_opt_actions_inf}, )
        chat_for_task = openai.ChatCompletion.create(model=self.model_name, messages=massages)
        reply_for_task = chat_for_task.choices[0].message.content
        print(f"ChatGPT: {reply_for_task}")
        self.messages.append({'role': 'assistant', 'content': reply_for_task})

        ask_opt_actions_inf = "Please come out ONLY the action number list ITSELF again with the format 'opt_actions: [1, 2, 3]' and WITHOUT ANY OTHER CONTENTS WITH LIST FORMAT. This is exclusively for the convenience for extracting the action number list. So please do not output redundant information with the list format. "
        print("User: ", ask_opt_actions_inf)
        massages.append({'role': 'user', 'content': ask_opt_actions_inf}, )
        chat_for_task = openai.ChatCompletion.create(model=self.model_name, messages=massages)
        reply_for_task = chat_for_task.choices[0].message.content
        print(f"ChatGPT: {reply_for_task}")
        massages.append({'role': 'assistant', 'content': reply_for_task})

        return reply_for_task


def extract_node(reply):
    # 完整正则匹配逻辑
    context = reply
    pattern = r'\[(\d+),\s*(\d+)\]'
    # ...完整实现...

    try:
        lists = re.findall(pattern, context)
        search_nodes = [eval(lst) for lst in lists]
    except IndexError:
        print('Empty list. It is caused by error output of LLM.')
    finally:
        if len(search_nodes) == 0:
            return search_nodes
        else:
            return search_nodes[-1]
        