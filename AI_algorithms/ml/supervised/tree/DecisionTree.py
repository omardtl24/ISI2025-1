import numpy as np # type: ignore
import pandas as pd # type: ignore
from ..base import SupervisedModel
from graphviz import Digraph, Source # type: ignore

def mode(y):
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]

def entropy(y):
    """
    
    Calculate the entropy of a label array.
    
    Parameters:
    y : np.ndarray
        Label array of shape (n_samples,).
        
    Returns:
    entropy : float
        The entropy of the label array.
        
    """
    if len(y) == 0:
        return 0
    if y.dtype == str:
        y = pd.Categorical(y).codes
    _ , counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding a small constant to avoid log(0)

def information_gain(X, y):
    """
    Calculate the information gain for each feature in the dataset and the number of samples per.

    Parameters:
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Label array of shape (n_samples,).

    Returns:
    gains : np.ndarray
        Information gain for each feature.

    """
    currEntropy = entropy(y)
    feat_entropies = np.zeros(X.shape[1])
    for feature in range(X.shape[1]):
        feature_values = np.unique(X[:, feature])
        for value in feature_values:
            subset_indices = X[:, feature] == value
            subset_y = y[subset_indices]
            feat_entropies[feature] += (len(subset_y) / len(y)) * entropy(subset_y)
    gains = currEntropy - feat_entropies
    return gains

def choose_split(X, y):
    gains = information_gain(X.values, y)
    best_feature = np.argmax(gains)
    return X.columns[best_feature], best_feature

class DecisionTree(SupervisedModel):
    def __init__(self, max_depth=None):
        super().__init__(name = "Decision Tree")
        self.max_depth = max_depth
        self.tree = None

    class Node:
        def __init__(self):
            self.children = {}

        def add_child(self, value, node):
            self.children[value] = node

    class NonLeafNode(Node):
        def __init__(self, feature, index, num_elements, entropy):
            super().__init__()
            self.feature = feature
            self.index = index
            self.entropy = entropy
            self.num_elements = num_elements

    class LeafNode(Node):
        def __init__(self, class_label):
            self.class_label = class_label
    
    def build_tree(self, X, y, default_class, strategy = choose_split, depth=0):
        if len(X)==0:
            return DecisionTree.LeafNode(default_class)
        if len(np.unique(y)) == 1:
            return DecisionTree.LeafNode(y[0])
        if len(X.columns) == 0 or (self.max_depth is not None and depth == self.max_depth):
            return DecisionTree.LeafNode(mode(y))
        feat, i = strategy(X, y)
        root = DecisionTree.NonLeafNode(feat, i, len(y), entropy(y))
        feature = X[feat]
        for value in feature.unique():
            subset_indices = feature == value
            subset_X = X[subset_indices].drop(columns=[feat])
            subset_y = y[subset_indices]
            child_node = self.build_tree(subset_X, subset_y, mode(y), strategy=strategy, depth=depth + 1)
            root.add_child(value, child_node)
        return root
    
    def custom_fit(self, X, y, strategy):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        self.tree = self.build_tree(X, y, mode(y), strategy=strategy, depth=0)

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.tree = self.build_tree(X, y, mode(y), depth=0)
    
    def predict(self, X):
        def traverse_tree(node, sample):
            if isinstance(node, DecisionTree.LeafNode):
                return node.class_label
            feature_value = sample[node.feature] if isinstance(sample, pd.Series) else sample[node.index]
            if feature_value in node.children:
                return traverse_tree(node.children[feature_value], sample)
            else:
                return None
            
        if isinstance(X, pd.DataFrame):
            return X.apply(lambda row: traverse_tree(self.tree, row), axis=1).values
        else:
            return np.array([traverse_tree(self.tree, row) for row in X])
    
    def plot(self):
        """
        Plot the decision tree inline in a Jupyter Notebook.
        """
        dot = Digraph()
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

        def add_nodes_edges(node, parent=None, edge_label=''):
            node_id = str(id(node))  # unique identifier

            if isinstance(node, DecisionTree.LeafNode):
                label = f'Class {node.class_label}'
                dot.node(node_id, label, fillcolor='lightgreen')
            elif isinstance(node, DecisionTree.NonLeafNode):
                label = f'Feature {node.feature}\nEntropy: {node.entropy:.3f}\nNum Samples: {node.num_elements}'
                dot.node(node_id, label, fillcolor='lightblue')
                for value, child in node.children.items():
                    add_nodes_edges(child, node_id, f'{value}')

            if parent is not None:
                dot.edge(parent, node_id, label=edge_label)

        # Build the graph from the root node
        add_nodes_edges(self.tree)
        return dot