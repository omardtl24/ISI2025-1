import networkx as nx # type: ignore
import matplotlib.pyplot as plt
import re

def parse_input(input_str):
    '''
    Parses the input string to extract the costed edges and heuristics.
    The input format is expected to be:
    node (h = \\d): (pairs , cost)
    where pairs are in the format (node, cost).
    Example:
    A (h = 2): (B, 1), (C, 2)
    B (h = 1): (D, 2), (E, 3)
    C (h = 0): (D, 1), (E, 2)
    D (h = 0): (F, 1)
    E (h = 0): (F, 2)
    F (h = 0): ()

    Arguments:
        input_str: str -- The input string containing the graph information.
    
    Returns:
        costed_edges: dict -- A dictionary of edges with their costs {(a,b): cost}.
        
        heuristics: dict -- A dictionary of nodes with their heuristics {node: heuristic}.
    '''
    lines = input_str.strip().split('\n')
    costed_edges = {}
    heuristics = {}

    for line in lines:
        line = line.strip()
        line = line.split(':')
        assert len(line) == 2, "Invalid input format. Expected 'node (h = \\d): (pairs , cost)'."
        #Heuristics
        node = line[0].strip()
        found = re.search(r'(.+)\(h\s*=\s*(\d+)\)', node)
        assert found, "Invalid input format. Expected 'node (h = \\d)'."
        node , h = found.groups()
        heuristics[node] = int(h)
        #Edges
        edges = line[1].strip()
        found = re.findall(r'\(\s*(\w+)\s*,\s*(\d+)\s*\)', edges)
        for edge in found:
            a, cost = edge
            cost = int(cost)
            costed_edges[(node, a)] = cost
    return costed_edges, heuristics

class Graph:
    def __init__(self,costed_edges,heuristics):
        '''
        costed_edges: dict of edges with their costs {(a,b): cost}
        heuristics: dict of nodes with their heuristics {node: heuristic}
        '''
        self.costed_edges = costed_edges
        self.heuristics = heuristics

        self.adj_list = {i:set() for i in self.heuristics}

        for a,b in costed_edges.keys():
            self.adj_list[a].add(b)

    def defineSortCriteria(self,option = None):
        flag = False
        if option == 'ascending label':
            for node, adj in self.adj_list.items():
                l = list(adj)
                l.sort()
                self.adj_list[node] = l
            flag = True
        elif option == 'descending label':
            for node, adj in self.adj_list.items():
                l = list(adj)
                l.sort(reverse = True)
                self.adj_list[node] = l
            flag = True
        elif option is None:
            for node, adj in self.adj_list.items():
                l = list(adj)
                self.adj_list[node] = l
            flag = True
        return flag

    def contains(self,node):
        return node in self.adj_list.keys()

    def h(self,node):
        if node not in self.heuristics.keys(): return None
        return self.heuristics[node]

    def cost(self,a,b):
        if (a,b) not in self.costed_edges.keys(): return None
        return  self.costed_edges[(a,b)]

    def adj(self,node):
        if node not in self.adj_list.keys(): return None
        return self.adj_list[node]

    def draw(self, path = None, title=None):
        G = nx.DiGraph()
        for a, b in self.costed_edges.keys():
            weight = self.cost(a, b)
            label = weight
            G.add_edge(a, b, label=label)

        pos = nx.circular_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        if path:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            path_edges_labels = {(a,b): self.cost(a,b) for a,b in path_edges}

            cost = sum(self.cost(a, b) for a, b in path_edges)

            nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=700, node_color='lightgreen')
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=path_edges_labels, font_color='red')
            arrow = '\\rightarrow '
            if title: title+= f"\nPath: $ {arrow.join(map(str, path))} $\nTotal Cost: {cost}"
            else: title = f"Path: ${arrow.join(map(str, path))}$\nTotal Cost: {cost}"

        if title:
            plt.title(title)
        plt.show()

    def print_structure(self):
        nodes = sorted(self.adj_list.keys())
        max_name_len = max(len(str(n)) for n in nodes)
        max_h_len = max(len(str(self.h(n))) for n in nodes)
        for node in nodes:
            neighbours = sorted(self.adj(node))
            strings = [f'({n},{self.cost(node,n)})' for n in neighbours]
            s = ', '.join(strings)
            node_str = str(node).ljust(max_name_len)
            h_str = str(self.h(node)).rjust(max_h_len)
            print(f'{node_str}(h = {h_str})  :  {s}')
