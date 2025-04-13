import re



def extract_node(reply):
    """
    This method is to extract the nodes of the search space in each step molocation. Commonly, there are multi-numbers
    of coordinate tuples in the context. Therefore, it is fairly different from extract_node() function.
    """
    pattern = r'\[(\d+),\s*(\d+)\]'

    try:
      lists = re.findall(pattern, reply)
      search_nodes = [eval(lst) for lst in lists]
    except IndexError:
      print('Empty list. It is caused by error output of LLM.')
    finally:
      if len(search_nodes) == 0:
        return search_nodes
      else:
        return search_nodes[-1]

def opt_node_select(opt_actions, max_margin, dis_map, start_node, pre_node, cur_node, par_nodes, open_queue, close_queue, action_space):

    """
    param:
    opt_actions: the action list return by function extract_node()
    """

    if sum(i>7 for i in opt_actions) != 0 or sum(i<0 for i in opt_actions) != 0 or \
    len(opt_actions) != len(set(opt_actions)) or len(opt_actions) == 0:   # error results
        return
    else:
        for i in range(len(opt_actions)):                           # collect neighbour nodes into open.queue
            act = opt_actions[i]
            sup_x = cur_node[0] + action_space[act][0]
            sup_y = cur_node[1] + action_space[act][1]
            sup_node = [sup_x, sup_y]
            if not(0 <= sup_x < max_margin) or not(0 <= sup_y < max_margin):        # move out of range
                continue
            elif abs(dis_map[sup_x][sup_y]) == 1024:           # move to obstacles
                continue
            elif str(sup_node) in list(close_queue.keys()):      # duplicated planing (meet the same node inside)
                continue
            elif sup_node in list(open_queue.values()):          # when duplicated searching, select the node with minimum G-cost as the parent node.

                par_key = list(open_queue.keys())[list(open_queue.values()).index(sup_node)]
                par = par_nodes[par_key]
                g_par = abs(dis_map[start_node[0]][start_node[1]] - dis_map[par[0]][par[1]])
                g_cur = abs(dis_map[start_node[0]][start_node[1]] - dis_map[cur_node[0]][cur_node[1]])
                if g_cur < g_par:
                    sup_key = list(open_queue.keys())[list(open_queue.values()).index(sup_node)]
                    par_nodes[sup_key] = cur_node

            else: # add new nodes
                end = 0 if len(open_queue) == 0 else list(open_queue.keys())[-1] + 1
                open_queue[end] = sup_node
                par_nodes[end] = cur_node

        # select optimal node based on f-cost

        opt_key = min(open_queue, key = lambda i: abs(dis_map[start_node[0]][start_node[1]] - \
                                                            dis_map[open_queue[i][0]][open_queue[i][1]]) + \
                    abs(dis_map[open_queue[i][0]][open_queue[i][1]]))

        print(f'opt_key: {opt_key}')

        to_close = open_queue.pop(opt_key)
        to_close_par = par_nodes.pop(opt_key)

        # update parent node

        close_queue[str(to_close)] = to_close_par

        # update current node of agent
        pre_node = to_close_par
        cur_node = to_close
