import networkx as nx # type: ignore
import matplotlib.pyplot as plt

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
