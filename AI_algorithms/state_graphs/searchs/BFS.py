from .verbose import get_structure
from ..graph import Graph

def BFSearch(G: Graph,start,end, criteria = None, verbose = False, reached = True):
    """
    Breadth First Search algorithm to find a path from start to end node in a graph.
    The graph must be defined with a sort criteria.

    Parameters:
        G (Graph): The graph to search in.
        start (str): The starting node.
        end (str): The ending node.
        criteria (str): The criteria to sort the graph. Default is None.
        verbose (bool): If True, prints the search process. Default is False.
        reached (bool): If True, only new nodes are added to the queue. Default is True.

    Returns:
        list: A list of nodes representing the path from start to end node.
        None if no path is found.
    """

    assert G.contains(start) and G.contains(end), "Start or end node not in graph"
    assert G.defineSortCriteria(option = criteria) , "Sort criteria not defined"

    queue = [(start , [start])]
    out = []
    reached_ = set([start])
    while len(queue) > 0:
        if verbose:
            fridge = get_structure(queue)
            print(f'Out: {fridge[0]}  -  Fridge: {fridge}')
        cur,path = queue.pop(0)
        out.append(cur)
        if cur == end:
            if verbose: print(f'\nOut order: {" ".join(out)}')
            return path
        for node in G.adj(cur):
            if reached:
                if node not in reached_:
                    queue.append((node , path + [node]))
                    reached.add(node)
            else:
                queue.append((node , path + [node]))
    return None