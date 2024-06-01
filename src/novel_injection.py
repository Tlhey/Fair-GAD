import torch
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
num_nodes = G.number_of_nodes()

node_features = torch.randn(num_nodes, 10)
node_classes = torch.randint(0, 2, (num_nodes,))

def path_outlier_injection(G, node_features, candidate_set):
    new_node_features = node_features.clone()
    for i in candidate_set:
        lengths = nx.single_source_shortest_path_length(G, i)
        farthest_node = max(lengths, key=lengths.get)
        new_node_features[i] = node_features[farthest_node]
    return new_node_features


def dice_n_outlier_injection(G, node_classes, candidate_set, r=0.1):
    new_G = G.copy()
    for i in candidate_set:
        V_r1 = [k for k in G.neighbors(i) if node_classes[k] == node_classes[i]]
        num_edges_to_remove = int(len(V_r1) * r)
        edges_to_remove = np.random.choice(V_r1, num_edges_to_remove, replace=False)
        new_G.remove_edges_from([(i, j) for j in edges_to_remove])

        V_r2 = [k for k in range(num_nodes) if k not in V_r1 and node_classes[k] != node_classes[i]]
        num_edges_to_add = num_edges_to_remove
        edges_to_add = np.random.choice(V_r2, num_edges_to_add, replace=False)
        new_G.add_edges_from([(i, j) for j in edges_to_add])
    
    return new_G

candidate_set = range(5)
new_node_features = path_outlier_injection(G, node_features, candidate_set)
print("New Node Features (Path Outlier Injection):", new_node_features)
new_G = dice_n_outlier_injection(G, node_classes, candidate_set, r=0.1)
print("New Graph (DICE-n Outlier Injection):", new_G.edges())
