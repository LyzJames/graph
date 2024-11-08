import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import networkx as nx
from tqdm import tqdm

splits = {'train': 'split/train-00000-of-00001.parquet', 'validation': 'split/validation-00000-of-00001.parquet', 'test': 'split/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])
df_validation = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["validation"])
df_test= pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["test"])

G = nx.DiGraph()

print(df_train.head())
word_count = {}

def Graph(df, graph, word_count):
    for text in tqdm(df['text'], desc="Processing texts", unit="text"):
        words = text.split()
        
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        
        for i in range(len(words) - 1):
            source = words[i]
            target = words[i + 1]
            

            if source not in graph.nodes:
                graph.add_node(source)
                graph.nodes[source]['out_weight'] = 0
                graph.nodes[source]['in_weight'] = 0
            

            if target not in graph.nodes:
                graph.add_node(target)
                graph.nodes[target]['out_weight'] = 0
                graph.nodes[target]['in_weight'] = 0
            

            if graph.has_edge(source, target):
                graph[source][target]['weight'] += 1
            else:
                graph.add_edge(source, target, weight=1)


            graph.nodes[source]['out_weight'] += 1
            

            graph.nodes[target]['in_weight'] += 1


    for word, count in word_count.items():
        graph.nodes[word]['count'] = count

Graph(df_train,G,word_count)

def remove_low_out_degree_nodes(graph, threshold):
    
    low_out_degree_nodes = [node for node in graph.nodes if graph.out_degree(node) < threshold]
    
    
    graph.remove_nodes_from(low_out_degree_nodes)
    return graph
threshold = 1




#remove_low_out_degree_nodes(G, threshold)

max_node = max(G.nodes(data=True), key=lambda x: x[1].get('count'))  


edges = G.edges(data=True)
max_edge_weight = max(edges, key=lambda x: x[2].get('weight'))
min_edge_weight = min(edges, key=lambda x: x[2].get('weight'))
'''
# print result
print("Nodes:",  len(G.nodes(data=True)))
print("Edges:", len(G.edges(data=True)))
print(f"max node value: {max_node[0]} (count: {max_node[1].get('count', 'N/A')})")
print(f"max edge value: {max_edge_weight[0]} -> {max_edge_weight[1]} (weight: {max_edge_weight[2]['weight']})")
print(f"min edge value: {min_edge_weight[0]} -> {min_edge_weight[1]} (weight: {min_edge_weight[2]['weight']})")
'''


def get_edge_weight(graph, node1, node2):
    if graph.has_edge(node1, node2):
        return graph[node1][node2]['weight']  
    else:
        return 0

def get_node_degree(graph, node):
    return (graph.in_degree(node), graph.out_degree(node)) 

def rebuild_word(text, graph, threshold):
    text = text.split(" ")
    new_words = []
    format_words = ''
    i = 0
    j = 0
    merged_word = ''
    while i < len(text) - 1:
        node1 = text[i]
        node2 = text[i + 1]
        
        edge_weight = get_edge_weight(graph, node1, node2)
        
        #relative_rate = min(edge_weight/get_node_degree(graph, node1)[1],edge_weight/get_node_degree(graph, node2)[0])
        if edge_weight != 0 and min(edge_weight/graph.nodes[node1].get('out_weight'),edge_weight/graph.nodes[node2].get('in_weight')) >= threshold:
            #print(edge_weight,graph.nodes[node1].get('out_weight'),graph.nodes[node2].get('in_weight'))
            merged_word += f"{node1} {node2} "
            
            i += 2  
        else:
            
            merged_word+= f"{node1}"
            new_words.append(f'({str(merged_word)})')
            merged_word = ''
            i += 1
    
    
    if i == len(text) - 1:
        new_words.append(f'({text[-1]})')
    
    if new_words:

        for j in range(len(new_words) - 1):
            format_words += str(new_words[j]).strip(' ').lstrip(' ') + ' '
        
        
        format_words += str(new_words[-1])
        format_words = format_words.strip(' ').lstrip(' ')
    return format_words

df_train['text'] = df_train['text'].apply(lambda x: rebuild_word(x, G, threshold=0.1))

df_validation['text'] = df_validation['text'].apply(lambda x: rebuild_word(x, G, threshold=0.2))
df_test['text'] = df_test['text'].apply(lambda x: rebuild_word(x, G, threshold=0.2))
print(df_train.tail())

