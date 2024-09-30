# Readme
## I. Introduction
这是Linux sysdig的日志解析脚本，将基本的工具都集成到了一起。
- config.py 在使用之前，请先在config.py中填好相关的参数
- utils.py 包含了一些常用的函数

## II. Basic Usage

使用前，导入utils.py文件：
```python
from utils import *
```

### Step 1. 修改config.py文件

```python
partition = " "# 每条日志的信息之间以空格分隔

ENTITY1 = 0# 不同成分的位置
ENTITY2 = 2
EVENT_TYPE = 1
TIMESTAMP1 = 3
TIMESTAMP2 = 5

raw_dir = "/path/to/sysdig/logs/" # 最原始的日志所在目录，目前进尧的代码还未合并进来，所以这一项暂时无用

processed_dir = "/path/to/processed/logs/" # raw_dir用进尧的代码处理之后的日志路径，也就是待分析的日志

dot_dir = "/path/to/dot/files/" # 如果要从一个dot图开始分析，这里填入待分析的dot文件所在目录
```

### Step 2. 实例化Graph对象，传入关键结点，以及对日志的一些预处理函数

```python
key_nodes = ['13187git', '13190git-remote-http', '13205chmod', '13209hello'] # key_nodes按照时间顺序传入

g = Graph(from_dot=False, key_nodes=key_nodes, preprocess_funcs=[replace_colons_with_semicolons])

print(f"Number of edges: {g.num_edges}, Number of nodes: {g.num_nodes}")
```

### Step 3. Reduce size of logs (这里用方法二)

```python
edges = g.reduced_crossEdges2()

# 当然，这里还可以根据step 4的结果作进一步的筛选，比如删除某个结点：
# edges = g.delete_node(edges, '1153alsa-sink-ALC25')

edgeNum, nodeNum = g.number_of_edges_and_nodes(edges)

print(f"Number of edges: {edgeNum}, Number of nodes: {nodeNum}")
```

### Step 4. 画出缩减后的图

```python
g.draw_graph_from_edges(edges, save_path='', key_colored=True)
```
注意，draw_graph_from_edges中调用的render函数的clean_up参数默认为True，会删除.dot文件，如果需要.dot文件，请将clean_up设置为False。

## III. Detailed reference

### 1. class Data(self, from_dot=False, preprocess_funcs=[])

- from_dot: 是否从dot文件开始分析，默认为False
- preprocess_funcs: 日志预处理函数列表，默认为[]

### 2. class Graph(self, from_dot=False, key_nodes=[], preprocess_funcs=[])

- from_dot: 是否从dot文件开始分析，默认为False
- key_nodes: 关键结点列表，按照时间顺序传入
- preprocess_funcs: 日志预处理函数列表，默认为[]
- self.G: 从dot文件或日志文件生成的networkx图
- self.key_nodes: 关键结点列表
- self.num_nodes: 图中节点的数量
- self.num_edges: 图中边的数量

### 3. Graph.successors(self, node: str)
获取node的后继结点，返回列表

### 4. Graph.predecessors(self, node: str)
获取node的前驱结点，返回列表

### 5. Graph.neighbors(self, node: str)
获取node的邻居结点，返回列表

### 6. Graph.edges_between(self, node1: str, node2: str)
获取node1和node2之间的所有边，返回一个元组列表
[(node1, node2, attr1: dict), (node1, node2, attr2: dict),...]

### 7. Graph.draw_graph_from_edges(self, edge_set: set, save_path: str, key_colored=False)
- edge_set: 元组列表，[(node1, node2), (node1, node3),...]
- save_path: 保存路径
- key_colored: 是否着重标注关键结点，默认为False

### 8. Graph.draw_graph(self, save_path: str)
- save_path: 保存路径

画出processed_dir对应日志的图

### 9. Graph.reduced_crossEdges1(self)
这个是方法一：首先，取第一个结点A和最后一个结点B的前后向遍历交集，并计算出所有从A到B的路径。再从这些路径中，筛选经过中间那些关键结点的路径，返回edge_set。存在的问题

- 时间、空间复杂度超高：其中的路径搜索算法用的是递归实现
- 最后得到的图非常简洁，会多删除成分

需要改进：
- 重新定义路径(path)：现在的路径定义为每个结点经过一次；之后可以定义为每条边经过一次。
- 改进路径搜索算法：比如，用BFS算法搜索路径，或者用Dijkstra算法搜索路径。

### 10. Graph.reduced_crossEdges2(self)
这个是方法二：对于给定的按照实现先后的关键结点列表，两两取前后向遍历的交集，最后合并。时间和空间复杂度相对较低。\
比如：对于key_nodes = ['A', 'B', 'C', 'D']。首先，取A和B的交集，再取B和C的交集，最后取C和D的交集，合并。

### 11. Graph.reduced_crossEdges3(self)
如有新的方法，请补充

### 12. Graph.number_of_edges_and_nodes(self, edge_set: set)
- edge_set: 元组列表，[(node1, node2), (node1, node3),...]

计算edge_set中边的数量和节点的数量

### 13. Graph.edgeSet_to_logs(self, edge: set, save_path: str)
- edge: 元组列表，[(node1, node2), (node1, node3),...]
- save_path: 保存路径

将edge_set转换为日志文件

### 14. Graph.delete_node(self, edge_set: set, node: str)
- edge_set: 元组列表，[(node1, node2), (node1, node3),...]
- node: 要删除的结点

删除结点node，返回新的edge_set