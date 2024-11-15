import pandas as pd
import networkx as nx
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

data_unsplit = pd.read_parquet("hf://datasets/dair-ai/emotion/unsplit/train-00000-of-00001.parquet")
df_list = []
db = data_unsplit.copy()

for num in [3000,1000,6000]:
  # Split the data based on label values
  sadness_data = db[db['label'] == 0].iloc[:num]
  joy_data = db[db['label'] == 1].iloc[:num]
  love_data = db[db['label'] == 2].iloc[:num]
  anger_data = db[db['label'] == 3].iloc[:num]
  fear_data = db[db['label'] == 4].iloc[:num]
  surprise_data = db[db['label'] == 5].iloc[:num]

  # Combine the data into a single DataFrame
  df = pd.concat([sadness_data, joy_data, love_data, anger_data, fear_data, surprise_data])

  # Remove the sampled data from the original dataset
  db = db.drop(df.index)

  df_list.append(df)

df_test = df_list[0].reset_index(drop=True)
df_validation = df_list[1].reset_index(drop=True)
df_train = df_list[2].reset_index(drop=True)

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
        
        if len(words) == 1:
            graph.add_node(words[0])
            graph.nodes[words[0]]['out_weight'] = 0
            graph.nodes[words[0]]['in_weight'] = 0
          
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

Graph(pd.concat([df_train, df_test, df_validation]),G,word_count)

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
            if  merged_word == '':
                merged_word += f"{node1}{node2}"
            
                #i += 1
            else:
                merged_word += f"{node2}" 
                #i += 1
        else:
            if merged_word != '':
                new_words.append(f'({str(merged_word)})')
                #pass
            else:
                merged_word+= f"{node1}"
                new_words.append(f'({str(merged_word)})')
            merged_word = ""
        i += 1

    if i == len(text) - 1:
        new_words.append(f'({text[-1]})')
    
    return new_words

df_train['text'] = df_train['text'].apply(lambda x: rebuild_word(x, G, 0.05))
df_validation['text'] = df_validation['text'].apply(lambda x: rebuild_word(x, G, 0.05))
df_test['text'] = df_test['text'].apply(lambda x: rebuild_word(x, G, 0.05))

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# 定义函数：移除文本中的停用词
def remove_stopwords(text):
    # 将文本拆分为单词
    words = text
    # 过滤掉停用词
    filtered_words = [word for word in words if word[1:-1].lower() not in stop_words]
    
    return " ".join(filtered_words)

df_train['text'] = df_train['text'].apply(remove_stopwords)
df_validation['text'] = df_validation['text'].apply(remove_stopwords)
df_test['text'] = df_test['text'].apply(remove_stopwords)


print(df_train.head())
