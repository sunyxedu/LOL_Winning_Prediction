from collections import Counter
import pandas as pd # 数据处理
import numpy as np # 数学运算
import math
from sklearn.model_selection import train_test_split, cross_validate # 划分数据集函数
from sklearn.metrics import accuracy_score # 准确率函数
RANDOM_SEED = 2020 # 固定随机种子
csv_data = './data/high_diamond_ranked_10min.csv' # 数据路径
data_df = pd.read_csv(csv_data, sep=',') # 读入csv文件为pandas的DataFrame
data_df = data_df.drop(columns='gameId') # 舍去对局标号列
print(data_df.iloc[0]) 
# 输出第一行数据
data_df.describe() 
# 每列特征的简单统计信息
drop_features = ['blueGoldDiff', 'redGoldDiff', 
                 'blueExperienceDiff', 'redExperienceDiff', 
                 'blueCSPerMin', 'redCSPerMin', 
                 'blueGoldPerMin', 'redGoldPerMin'] # 需要舍去的特征列
df = data_df.drop(columns=drop_features) # 舍去特征列
info_names = [c[3:] for c in df.columns if c.startswith('red')] # 取出要作差值的特征名字（除去red前缀）
for info in info_names: # 对于每个特征名字
    df['br' + info] = df['blue' + info] - df['red' + info] # 构造一个新的特征，由蓝色特征减去红色特征，前缀为br
# 其中FirstBlood为首次击杀最多有一只队伍能获得，brFirstBlood=1为蓝，0为没有产生，-1为红
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood']) # 原有的FirstBlood可删除
print(df)
discrete_df = df.copy() # 先复制一份数据
for c in df.columns[1:]: # 遍历每一列特征，跳过标签列
    '''
    请离散化每一列特征，即discrete_df[c] = ...
    
    提示：
    对于有些特征本身取值就很少，可以跳过即 if ... : continue
    对于其他特征，可以使用等区间离散化、等密度离散化或一些其他离散化方法
    可参考使用pandas.cut或qcut
    '''
    if len(df[c].unique()) <= 5:
        continue
    else:
        discrete_df[c] = pd.cut(df[c], bins=5, labels=False)
    # discrete_df[c] = (df[c] * 10 // max(abs(df[c])))
    
print(discrete_df)
all_y = discrete_df['blueWins'].values # 所有标签数据
feature_names = discrete_df.columns[1:] # 所有特征的名称
all_x = discrete_df[feature_names].values # 所有原始特征值，pandas的DataFrame.values取出为numpy的array矩阵

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=RANDOM_SEED)
all_y.shape, all_x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape # 输出数据行列信息
# 定义决策树类

class DecisionTree(object):
    class Node:
        def __init__(self, left, right, feature, number, value):
            self.feature = feature
            self.left = left
            self.right = right
            self.number = number
            self.value = value
    def __init__(self, classes, features, max_depth=10, min_samples_split=10, impurity_t='entropy'):
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.root = None # 定义根节点，未训练时为空
        '''
        传入一些可能用到的模型参数，也可能不会用到
        classes表示模型分类总共有几类
        features是每个特征的名字，也方便查询总的共特征数
        max_depth表示构建决策树时的最大深度
        min_samples_split表示构建决策树分裂节点时，如果到达该节点的样本数小于该值则不再分裂
        impurity_t表示计算混杂度（不纯度）的计算方式，例如entropy或gini
        '''  
   
    '''
    请实现决策树算法，使得fit函数和predict函数可以正常调用，跑通之后的测试代码，
    要求之后测试代码输出的准确率大于0.6。
    
    提示：
    可以定义额外一些函数，例如
    impurity()用来计算混杂度
    gain()调用impurity用来计算信息增益
    expand_node()训练时递归函数分裂节点，考虑不同情况
        1. 无需分裂 或 达到分裂阈值
        2. 调用gain()找到最佳分裂特征，递归调用expand_node
        3. 找不到有用的分裂特征
        fit函数调用该函数返回根节点
    traverse_node()预测时遍历节点，考虑不同情况
        1. 已经到达叶节点，则返回分类结果
        2. 该特征取值在训练集中未出现过
        3. 依据特征取值进入相应子节点，递归调用traverse_node
    当然也可以有其他实现方式。

    '''
    def impurity(self, labels, title):
        if title == "gini":
            # print("in")
            # print("QWQ1")
            total_size = len(labels)
            gini = 0.0
            # print(len(self.classes))
            for item in self.classes:
                frequency = list(labels).count(item) / total_size
                gini += frequency ** 2
            # print("QWQ2")
            return 1.0 - gini
        elif title == "entropy":
            entropy = 0.0
            total_size = len(labels)
            for item in self.classes:
                frequency = list(labels).count(item) / total_size
                if frequency == 0:
                    continue
                else:
                    entropy -= frequency * math.log2(frequency)
            return entropy

    def gain(self, left, right, impurity_now):
        total = len(left) + len(right)
        return float(impurity_now - float(self.impurity(left, self.impurity_t) * len(left) / total) - float(self.impurity(right, self.impurity_t) * len(right) / total))
    '''
    def count_max(self, label):
        max_value = 0
        max_number = 0
        label_count = {}
        for item in label:
            if item.count not in label_count:
                label_count[item] = 1
            else:
                label_count[item] += 1
        for item, number in label_count.items():
            if count > max_number:
                max_number = count
                max_value = item
        return max_number, max_value
    '''
    def expand_node(self, feature, label, depth):
        # print(2333)
        if len(np.unique(label)) == 1:
            return self.Node(None, None, None, None, label[0])
        # print(23333)
        if depth >= self.max_depth or len(feature) < self.min_samples_split:
            return self.Node(None, None, None, None, np.bincount(label).argmax())
        # print(233333)
        max_value = 0
        max_number = 0
        left_x = None
        left_y = None
        right_x = None
        right_y = None
        number = 0
        impurity_now = self.impurity(label, self.impurity_t)
        # print("qwq")
        # print(feature.shape[1], len(feature))
        for item in range(feature.shape[1]):
            # print(item)
            sorted_index = np.argsort(feature[:, item])
            for index in range(1, feature.shape[0]):
                # print(index)
                if feature[sorted_index[index - 1], item] == feature[sorted_index[index], item]:
                    continue
                # print("IN")
                LLL = sorted_index[:index]
                RRR = sorted_index[index:]
                # print("OUT1")
                value = self.gain(label[LLL], label[RRR], impurity_now)
                # print("OUT2")
                if value > max_value:
                    max_value = value
                    max_number = item
                    number = (feature[sorted_index[index-1], item] + feature[sorted_index[index], item]) / 2
                    left_x = feature[LLL]
                    left_y = label[LLL]
                    right_x = feature[RRR]
                    right_y = label[RRR]
                # print("OUT3")

        if max_value == 0:
            return self.Node(None, None, None, None, np.bincount(label).argmax())
        '''
        Left = []
        Right = []
        for item in range(len(feature)):
            if feature[item, max_number] <= number:
                Left.append(feature[item, max_number])
            else:
                Right.append(feature[item, max_number])
        LL = np.array(Left)
        RR = np.array(Right)
        '''
        # print(len(total_left))
        left = self.expand_node(left_x, left_y, depth + 1)
        right = self.expand_node(right_x, right_y, depth + 1)
        return self.Node(left, right, max_number, number, None)

    def traverse_node(self, node, ft):
        if node.value == None:
            # print(node.feature)
            val = ft[node.feature]
            if val <= node.number:
                return self.traverse_node(node.left, ft)
            else:
                return self.traverse_node(node.right, ft)
        else:
            return node.value

    def fit(self, feature, label):
        assert len(self.features) == len(feature[0]) # 输入数据的特征数目应该和模型定义时的特征数目相同
        '''
        训练模型
        feature为二维numpy（n*m）数组，每行表示一个样本，有m个特征
        label为一维numpy（n）数组，表示每个样本的分类标签
        
        提示：一种可能的实现方式为
        self.root = self.expand_node(feature, label, depth=1) # 从根节点开始分裂，模型记录根节点
        '''
        self.root = self.expand_node(feature, label, depth=1)
    
    def predict(self, feature):
        assert len(feature.shape) == 1 or len(feature.shape) == 2 # 只能是1维或2维
        '''
        预测
        输入feature可以是一个一维numpy数组也可以是一个二维numpy数组
        如果是一维numpy（m）数组则是一个样本，包含m个特征，返回一个类别值
        如果是二维numpy（n*m）数组则表示n个样本，每个样本包含m个特征，返回一个numpy一维数组
        
        提示：一种可能的实现方式为
        if len(feature.shape) == 1: # 如果是一个样本
            return self.traverse_node(self.root, feature) # 从根节点开始路由
        return np.array([self.traverse_node(self.root, f) for f in feature]) # 如果是很多个样本
        '''
        if len(feature.shape) == 1:
            return self.traverse_node(self.root, feature)
        return np.array([self.traverse_node(self.root, f) for f in feature])
    
    def evaluate(self, node, features, labels):
        predictions = [self.traverse_node(node, f) for f in features]
        return sum(p != l for p, l in zip(predictions, labels))
    
    def prune_node(self, node, features, labels):
        if node.value is not None:
            return self.evaluate(node, features, labels)
        left_error_before = self.evaluate(node.left, features[features[:, node.feature] <= node.number], labels[features[:, node.feature] <= node.number])
        right_error_before = self.evaluate(node.right, features[features[:, node.feature] > node.number], labels[features[:, node.feature] > node.number])
        left_error_after = self.prune_node(node.left, features[features[:, node.feature] <= node.number], labels[features[:, node.feature] <= node.number])
        right_error_after = self.prune_node(node.right, features[features[:, node.feature] > node.number], labels[features[:, node.feature] > node.number])
        node_error = len(labels) - np.bincount(labels).max()
        total_error_before = left_error_before + right_error_before
        total_error_after = left_error_after + right_error_after
        if node_error <= total_error_after:
            node.left = None
            node.right = None
            node.feature = None
            node.number = None
            node.value = np.bincount(labels).argmax()
            return node_error
        return total_error_before
        
    def post_prune(self, features, labels):
        self.prune_node(self.root, features, labels)

# 定义决策树模型，传入算法参数
DT = DecisionTree(classes=[0,1], features=feature_names, max_depth=6, min_samples_split=10, impurity_t='entropy')
# print("IN")
DT.fit(x_train, y_train) # 在训练集上训练
DT.post_prune(x_valid, y_valid)
p_test = DT.predict(x_test) # 在测试集上预测，获得预测值
print(p_test) # 输出预测值
test_acc = accuracy_score(p_test, y_test) # 将测试预测值与测试集标签对比获得准确率
print('accuracy: {:.4f}'.format(test_acc)) # 输出准确率