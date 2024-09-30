from config import *
import networkx as nx
import graphviz
from tqdm import tqdm
import time



def time_converter(ptime: str) -> float:
    '''
        convert time string to seconds

    Args:
        ptime (str): time string in format of "HH:MM:SS"(SS is a float number)

    Returns:
        float: seconds
    '''
    if len(ptime.split(':')) == 3:
        h, m, s = ptime.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        return ptime


def Dict_tupleSet(res: dict, forward=True) -> set:
    '''
        convert a dictionary to a set of tuples, which represents the edges.

    Args:
        res (dict): a dictionary containing the edges
        forward (bool, optional): whether the edges are forward or backward. Defaults to True.

    Returns:
        set: a set of tuples representing the edges
    '''
    new_res = []
    for key in res:
        for value in res[key]:
            if forward:
                new_res.append((key, value))
            else:
                new_res.append((value, key))
    return set(new_res)


def resDict_tupleList(res1: dict, res2: dict) -> list:
    '''
        convert two dictionaries to a list of tuples, which represents the cross edges.
        Note that the multiple edges between two nodes are counted only once, i.e.,
        if there are multiple edges from A to B, only (A, B) is stored in the result list.

    Args:
        res1 (dict): a dictionary containing the forward edges
        res2 (dict): a dictionary containing the backward edges

    Returns:
        list: a list of tuples representing the cross edges
    '''
    new_res1 = []
    for key in res1:
        for value in res1[key]:
            new_res1.append((key, value))
    new_res2 = []
    for key in res2:
        for value in res2[key]:
            new_res2.append((value, key))
    
    new1 = set(new_res1)
    # print(f"Number of forward edges(multiple edges between two nodes are counted only once): {len(new_res1)}")
    new2 = set(new_res2)
    # print(f"Number of backward edges(multiple edges between two nodes are counted only once): {len(new_res2)}")
    res = new1 & new2
    # print(f"Number of cross edges(multiple edges between two nodes are counted only once): {len(res)}")

    return list(res)


def replace_colons_with_semicolons(event_list: list) -> list:
    for i in range(len(event_list)):
        if ':' in event_list[i]['entity1']:
            event_list[i]['entity1'] = event_list[i]['entity1'].replace(':', ';')
        if ':' in event_list[i]['entity2']:
            event_list[i]['entity2'] = event_list[i]['entity2'].replace(':', ';')
    return event_list


def get_shape(string: str) -> str:
    if '/' in string:
        return 'ellipse'
    elif '->' in string and '.' in string:
        return 'parallelogram'
    else:
        return 'square'


#########################################################################################
#########################################################################################


# EXPORT
class Data:
    def __init__(self, from_dot=False, preprocess_funcs = []) -> None:
        '''
            Initialize the data object.

        Args:
            from_dot (bool, optional): whether the data is from dot file. Defaults to False. If from_dot is True,
            the dot file in config.py will be read to get the data. Otherwise, the processed data in config.py will be read.

            preprocess_funcs (list, optional): a list of functions to preprocess the data. Defaults to [].

            Functions in preprocess_funcs should take a list of events as input and return a list of events as output.
            For example:
            >>> def preprocess_func_example(event_list):
            >>>     new_event_list = []
            >>>     for event in event_list:
            >>>         if event['event_type'] == 'connect':
            >>>             new_event_list.append(event)
            >>>     return new_event_list
            >>> preprocess_funcs = [preprocess_func_example]

            By default, the preprocess_funcs is an empty list, which means no preprocessing is done.

            By default, we only consider the start time of the event as the timestamp. If you want to consider the end time,
            you can modify the code in the Data class and config.py.(Timestamp2 is the end time)
        '''
        self.events = []
        self.from_dot = from_dot
        if self.from_dot:
            self.__from_dot()
        else:
            with open(processed_dir, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.strip() == '':
                    continue
                event_list = line.strip().split(partition)
                self.events.append({
                            "entity1": event_list[ENTITY1],
                            "entity2": event_list[ENTITY2],
                            "event_type": event_list[EVENT_TYPE],
                            "timestamp1": time_converter(event_list[TIMESTAMP1]),
                            # "timestamp2": time_converter(event_list[TIMESTAMP2]), 考虑结束时间的话，直接取消这里的注释，代码的其它部分也要作相应改动
                })
        self.__preprocess(preprocess_funcs)
        self.events.sort(key=lambda x: x['timestamp1'])
    

    def __preprocess(self, func = []) -> None:
        new_events = []

        for i in range(len(self.events)-1):
            if self.events[i]['entity1'] != self.events[i+1]['entity1'] or \
                self.events[i]['entity2'] != self.events[i+1]['entity2'] or \
                self.events[i]['event_type'] != self.events[i+1]['event_type']:

                new_events.append(self.events[i]) if '<NA>' not in self.events[i]['entity1'] and \
                    '<NA>' not in self.events[i]['entity2'] else None

        new_events.append(self.events[-1]) if '<NA>' not in self.events[-1]['entity1'] and \
            '<NA>' not in self.events[-1]['entity2'] else None

        if func:
            for f in func:
                new_events = f(new_events)
        
        self.events = new_events
    

    def __from_dot(self) -> None:
        self.G = nx.drawing.nx_pydot.read_dot(dot_dir)

        node_labels = nx.get_node_attributes(self.G, 'label')

        for u, v, key in self.G.edges(keys=True):
            self.events.append({
                "entity1": node_labels[u][1:-1],
                "entity2": node_labels[v][1:-1],
                "event_type": self.G.get_edge_data(u, v, key)['label'][1:-1].split(' ')[0],
                "timestamp1": time_converter(self.G.get_edge_data(u, v, key)['label'][1:-1].split(' ')[1]),
                "color": self.G.get_edge_data(u, v, key)['color'],
                # "timestamp2": self.G.get_edge_data(u, v, key)['timestamp'],
            })
    


class Graph(Data):
    def __init__(self, key_nodes: list = [], from_dot=False, preprocess_funcs=[]) -> None:
        '''
            Initialize the graph object.

        Args:
            key_nodes (list, optional): a list of key nodes. Defaults to None.

            from_dot (bool, optional): whether the data is from dot file. Defaults to False. If from_dot is True,
            the dot file in config.py will be read to get the data. Otherwise, the processed data in config.py will be read.

            preprocess_funcs (list, optional): a list of functions to preprocess the data. Defaults to [].
        '''
        super().__init__(from_dot=from_dot, preprocess_funcs=preprocess_funcs)
        self.G = nx.MultiDiGraph()
        self.key_nodes = key_nodes
        self.from_dot = from_dot
        if from_dot:
            for event in self.events:
                self.G.add_edge(
                    event["entity1"],
                    event["entity2"],
                    event_type=event["event_type"],
                    timestamp=event["timestamp1"],
                    color=event["color"],
                    # label = str(event["timestamp1"]) + " " + event["event_type"]
                )
        else:
            for event in self.events:
                self.G.add_edge(
                    event["entity1"],
                    event["entity2"],
                    event_type=event["event_type"],
                    timestamp=event["timestamp1"],
                    # label = str(event["timestamp1"]) + " " + event["event_type"]
                )
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
    

    def successors(self, node: str) -> list:
        return list(self.G.successors(node))
    
    
    def predecessors(self, node: str) -> list:
        return list(self.G.predecessors(node))
    

    def neighbors(self, node: str) -> list:
        return list(set(self.successors(node) + self.predecessors(node)))
    
    
    def edges_between(self, node1: str, node2: str) -> list:
        '''
            Get the edges between two nodes.

        Args:
            node1 (str): the first node
            node2 (str): the second node

        Returns:
            list: a list of tuples representing the information of the edges between node1 and node2.
            Each tuple contains three elements: (u, v, attr), where u and v are the endpoints of the edge,
            and attr is a dictionary containing the attributes of the edge.
        '''
        tmp = self.G.edges(node1, keys=True)
        edges = []
        for u, v, key in tmp:
            if v == node2:
                edges.append((u, v, key))

        res = []
        for u, v, key in edges:
            res.append((u, v, self.G.get_edge_data(u, v, key)))
        return res


    def __forwardBFS(self, start_node: str) -> list:
        visited = set()
        res = {}
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self.successors(node))
                res[node] = self.successors(node)
        return Dict_tupleSet(res, forward=True)
    
    
    def __backwardBFS(self, start_node: str) -> list:
        visited = set()
        res = {}
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self.predecessors(node))
                res[node] = self.predecessors(node)
        return Dict_tupleSet(res, forward=False)
    

    def __forward_backward_crossEdges(self, node1, node2) -> list:
        res = self.__forwardBFS(node1) & self.__backwardBFS(node2)
        return list(res)
    

    def draw_graph_from_edges(self, edge_set: set, save_path: str, key_colored=False) -> None:
        '''
            Draw the graph from a set of edges.

        Args:
            edge_set (set): a set of tuples representing the edges.
            save_path (str, optional): the path to save the graph. Defaults to None.
            key_colored (bool, optional): whether the key nodes are colored. Defaults to False.
        '''
        ## 待修改
        tmp = graphviz.Digraph()
        tmp.graph_attr['rankdir'] = 'LR'

        edge_list = []
        for u, v in edge_set:
            edge_list.extend(self.edges_between(u, v))
        edge_list.sort(key=lambda x: x[2]['timestamp'])

        if key_colored:
            for u, v in edge_set:
                if u in self.key_nodes:
                    tmp.node(u, color='red', shape=get_shape(u))
                else:
                    tmp.node(u, shape=get_shape(u))
                if v in self.key_nodes:
                    tmp.node(v, color='red', shape=get_shape(v))
                else:
                    tmp.node(v, shape=get_shape(v))

            for u, v, attr in edge_list:
                if u in self.key_nodes or v in self.key_nodes:
                    tmp.edge(
                        u,
                        v,
                        label=str(attr['timestamp']) + " " + attr['event_type'],
                        color='red',
                    )
                else:
                    tmp.edge(
                        u,
                        v,
                        label=str(attr['timestamp']) + " " + attr['event_type'],
                    )
        else:
            for u, v in edge_set:
                tmp.node(u, shape=get_shape(u))
                tmp.node(v, shape=get_shape(v))
            for u, v, attr in edge_list:
                tmp.edge(
                    u,
                    v,
                    label=str(attr['timestamp']) + " " + attr['event_type'],
                )
        tmp.render(save_path, view=False, format='pdf', cleanup=True)

    
    def draw_graph(self, save_path: str) -> None:
        '''
            Draw the graph.

        Args:
            save_path (str): the path to save the graph.
        '''
        tmp = graphviz.Digraph()
        tmp.graph_attr['rankdir'] = 'LR'
        for n in self.G.nodes():
            tmp.node(n, shape=get_shape(n))
        if self.from_dot:
            for u, v, key in self.G.edges(keys=True):
                tmp.edge(
                    u,
                    v,
                    label=str(self.G.get_edge_data(u, v, key)['timestamp']) + " " + self.G.get_edge_data(u, v, key)['event_type'],
                    color=self.G.get_edge_data(u, v, key)['color'],
                )
        else:
            for u, v, key in self.G.edges(keys=True):
                tmp.edge(
                    u,
                    v,
                    label=str(self.G.get_edge_data(u, v, key)['timestamp']) + " " + self.G.get_edge_data(u, v, key)['event_type'],
                )
        tmp.render(save_path, view=False, format='pdf', cleanup=True)
    

    def __get_paths(self, start_node: str, end_node: str, edges: set) -> list:
        ## 现在的路径是只经过每个结点一次，这样的话，删去的成分过多；
        ## 后续可以把这个算法改成经过每条边一次；看之后的数据集测试效果怎么样

        G = nx.DiGraph()
        for u, v in edges:
            G.add_edge(u, v)
        def find_paths(graph, start, end, path=[]):
            path = path + [start]
            if start == end:
                return [path]
            if start not in graph:
                return []
            paths = []
            for node in graph.neighbors(start):
                if node not in path:
                    newpaths = find_paths(graph, node, end, path)
                    for newpath in newpaths:
                        paths.append(newpath)
            return paths
        return find_paths(G, start_node, end_node)
    

    def reduced_crossEdges1(self):
        '''
            Method1: Remove the paths (which are not connected to the middle key nodes) from cross edges of first and last key nodes.

        Returns:
            set: a set of tuples representing the reduced cross edges.
        '''
        paths = self.__get_paths(
            self.key_nodes[0],
            self.key_nodes[-1],
            self.__forward_backward_crossEdges(self.key_nodes[0], self.key_nodes[-1])
        )
        result = []
        for path in tqdm(paths, desc='Reducing paths'):
            flag = True
            for n in self.key_nodes[1:-1]:
                if n not in path:
                    flag = False
                    break
            if flag:
                result.append(path)
        res = set()
        for r in result:
            for i in range(len(r)-1):
                res.add((r[i], r[i+1]))
        return res
    

    def reduced_crossEdges2(self):
        '''
            Method2: Get the cross edges between two adjacent key nodes, and then get the union of all cross edges.

        Returns:
            set: a set of tuples representing the reduced cross edges.
        '''
        res = set()
        for i in range(len(self.key_nodes)-1):
            res.update(self.__forward_backward_crossEdges(self.key_nodes[i], self.key_nodes[i+1]))
        return res
    

    def reduced_crossEdges3(self):
        # New method(to be implemented)
        pass
    

    def number_of_edges_and_nodes(self, edges: set):
        '''
            Calculate the number of edges and nodes in a set of edges.

        Args:
            edges (set): a set of tuples representing the edges.

        Returns:
            tuple: a tuple containing two integers, the first one is the number of edges, and the second one is the number of nodes.
        '''
        numEdge = len(edges)
        node_set = set()
        for u, v in edges:
            node_set.add(u)
            node_set.add(v)
        numNode = len(node_set)
        return numEdge, numNode
    

    def edgeSet_to_logs(self, edges: set, save_path: str) -> None:
        '''
            Save the edges to a log file.

        Args:
            edges (set): a set of tuples representing the edges.
            save_path (str): the path to save the log file.
        '''
        edge_list = []
        lines = []
        for u, v in edges:
            edge_list.extend(self.edges_between(u, v))
        edge_list.sort(key=lambda x: x[2]['timestamp'])
        for u, v, attr in edge_list:
            lines.append(f"{u}{partition}{v}{partition}{attr['event_type']}{partition}{attr['timestamp']}")
        with open (save_path, 'w') as f:
            f.write('\n'.join(lines))
    

    def delete_node(self, edge_set: set, node: str) -> set:
        '''
            Delete a node from a set of edges.

        Args:
            edge_set (set): a set of tuples representing the edges.
            node (str): the node to be deleted.

        Returns:
            set: a set of tuples representing the edges after deleting the node.
        '''
        new_edge_set = set()
        for u, v in edge_set:
            if u != node and v != node:
                new_edge_set.add((u, v))
        return new_edge_set

    


if __name__ == '__main__':

    start_time = time.time()
###########################################################
#################  code for testing  ######################
###########################################################
# 代码从这里开始，方便计算运行时间

    key_nodes = ['13187git', '13190git-remote-http', '13205chmod', '13209hello'] # key_nodes按照时间顺序传入
    g = Graph(from_dot=False, key_nodes=key_nodes, preprocess_funcs=[replace_colons_with_semicolons])
    print(f"Number of edges: {g.num_edges}, Number of nodes: {g.num_nodes}")
    edges = g.reduced_crossEdges2()
    # edges = g.delete_node(edges, '1153alsa-sink-ALC25')
    edgeNum, nodeNum = g.number_of_edges_and_nodes(edges)
    print(f"Number of reduced edges: {edgeNum}, Number of reduced nodes: {nodeNum}")
    # g.draw_graph_from_edges(edges, save_path='case51', key_colored=True)# 关键结点以及与其连接的边标成红色

    
    
    
    
###########################################################
###########################################################
    end_time = time.time()
    print("Time taken:", end_time - start_time)

