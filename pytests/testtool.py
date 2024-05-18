import torchKQI
import logging
import math


class DiGraph():
    def __init__(self) -> None:
        '''
        self.__pred = {node: [pred, ...]}
        self.__succ = {node: [succ, ...]}
        decay: Percentage of decay per decade, from 0 to 1.
        '''
        self.__pred = {}
        self.__succ = {}
        self.__UPDATE_FLAG = False
        self.__volume = {}
        self.__graph_volume = 0

    def add_node(self, v, pred: list):
        '''
        Add node like BA model. This function will override the old node.
        '''
        if self.__pred.__contains__(v):
            raise Exception('Add repeated node!')

        self.__pred[v] = [u for u in set(pred)]
        for u in self.__pred[v]:
            if not self.__succ.__contains__(u):
                self.__succ[u] = [v]
            else:
                self.__succ[u].append(v)
        self.__UPDATE_FLAG = True

    def nodes(self):
        for v in self.__pred.keys():
            yield v

    def number_of_nodes(self):
        return self.__pred.__len__()

    def number_of_edges(self):
        return sum(map(lambda k: len(k), self.__pred.values()))

    def exist_node(self, v):
        return v in self.__pred

    def successors(self, u):
        if not self.__succ.__contains__(u):
            return []
        for v in self.__succ[u]:
            yield v

    def predecessors(self, v):
        if not self.__pred.__contains__(v):
            return []
        for u in self.__pred[v]:
            yield u

    def in_degree(self, v):
        return len(self.__pred[v])

    def out_degree(self, u):
        if u not in self.__succ:
            return 0
        return len(self.__succ[u])

    def __clear_cache(self):
        self.__volume = {}
        self.__graph_volume = 0
        self.__partition_tree()

    def topological_sort(self):
        try:
            if not self.__UPDATE_FLAG:
                return self.sorted_list
        except AttributeError:
            pass

        indegree_map = {v: self.in_degree(v) for v in self.nodes() if self.in_degree(v) > 0}
        # These nodes have zero indegree and ready to be returned.
        zero_indegree = [v for v in self.nodes() if self.in_degree(v) == 0]

        self.sorted_list = []
        while zero_indegree:
            node = zero_indegree.pop()
            for child in self.successors(node):
                indegree_map[child] -= 1
                if indegree_map[child] == 0:
                    zero_indegree.append(child)
                    del indegree_map[child]

            self.sorted_list.append(node)

        if indegree_map.keys():
            print(indegree_map.keys())
            raise Exception("Graph contains a cycle or graph changed during iteration")

        return self.sorted_list

    def __partition_tree(self):
        """
        Calculate volume.
        """
        for node in reversed(self.topological_sort()):
            self.__volume[node] = self.out_degree(node) + sum(map(lambda k: self.__volume[k] / self.in_degree(k), self.successors(node)))

        self.__graph_volume = sum([1 + self.__volume[v] for v in self.nodes() if self.in_degree(v) == 0])

    def kqi(self, v: int) -> float:
        """
        Calculate KQI.
        """
        if self.__UPDATE_FLAG:
            self.__clear_cache()
            self.__UPDATE_FLAG = False

        if self.__volume[v] == 0:
            return 0

        in_deg = self.in_degree(v)
        if in_deg == 0:
            return -self.__volume[v] / self.__graph_volume * math.log2(self.__volume[v] / self.__graph_volume)

        return sum(map(lambda k: -self.__volume[v] / in_deg / self.__graph_volume * math.log2(self.__volume[v] / in_deg / self.__volume[k]), self.predecessors(v)))

    def volume(self, v: int) -> float:
        return self.__volume[v]

    def graph_volume(self) -> float:
        return self.__graph_volume

    def print_kqi(self, names: list):
        """
        Print KQI of the graph.
        :param names: A list of names of nodes.
        :return:
        """
        kqi = 0
        for name in names:
            kqi += sum(map(lambda k: self.kqi(k) if name in k else 0, self.nodes()))
            print(f'{name}: KQI={kqi}, node={len([k for k in self.nodes() if name in k])}, volume={sum([self.volume(k) for k in self.nodes() if name in k])}')


def testKQI(model, x):
    logging.debug('============================ KQI (torch) ============================')
    kqi_torch = torchKQI.KQI(model, x)

    logging.debug('============================ KQI (graph) ============================')
    G = DiGraph()
    for v, pred, _, _, _ in torchKQI.Graph(model, x):
        G.add_node(v, pred)
    kqi_graph = sum([G.kqi(k) for k in G.nodes()])

    logging.debug('============================ KQI (check) ============================')
    for v, _, funcname, kqi, vol in torchKQI.Graph(model, x):
        assert math.isclose(G.volume(v), vol, rel_tol=1e-4), f'[{funcname}({v})]: Volume (graph) = {G.volume(v)}, Volume (torch) = {vol}'
        assert math.isclose(G.kqi(v), kqi, rel_tol=1e-4), f'[{funcname}({v})]: KQI (graph) = {G.kqi(v)}, KQI (torch) = {kqi}'

    assert math.isclose(kqi_graph, kqi_torch, rel_tol=1e-4), f'KQI (torch) = {kqi_torch}, KQI (graph) = {kqi_graph}'
