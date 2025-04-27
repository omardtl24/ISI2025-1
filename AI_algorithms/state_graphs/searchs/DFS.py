from .verbose import get_structure
from ..graph import Graph

def DFSearch(G: Graph,start,end, criteria = None, verbose = False):
    """
    Depth First Search algorithm to find a path from start to end node in a graph.
    The graph must be defined with a sort criteria.

    Parameters:
        G (Graph): The graph to search in.
        start (str): The starting node.
        end (str): The ending node.
        criteria (str): The criteria to sort the graph. Default is None.
        verbose (bool): If True, prints the search process. Default is False.

    Returns:
        list: A list of nodes representing the path from start to end node.
        None if no path is found.
    """
    assert G.contains(start) and G.contains(end) , "Start or end node not in graph"
    assert G.defineSortCriteria(option = criteria) , "Sort criteria not defined"

    stack = [(start , [start])]
    out = []
    reached = set([start])
    while len(stack) > 0:
        if verbose:
            fridge = get_structure(stack)
            print(f'Out: {fridge[-1]}  -  Fridge: {fridge}')
        cur,path = stack.pop()
        out.append(cur)
        if cur == end:
            if verbose: print(f'\nOut order: {" ".join(out)}')
            return path
        for node in G.adj(cur):
            if node not in reached:
              stack.append((node , path + [node]))
              reached.add(node)
    if verbose: print(f'\nOut order: {" ".join(out)}')
    return None