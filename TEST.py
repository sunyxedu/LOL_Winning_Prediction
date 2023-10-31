from collections import Counter
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
RANDOM_SEED = 2020
csv_data = './data/high_diamond_ranked_10min.csv'
data_df = pd.read_csv(csv_data, sep=',')
data_df = data_df.drop(columns='gameId')
print(data_df.iloc[0])
data_df.describe()
drop_features = ['blueGoldDiff', 'redGoldDiff', 
                 'blueExperienceDiff', 'redExperienceDiff', 
                 'blueCSPerMin', 'redCSPerMin', 
                 'blueGoldPerMin', 'redGoldPerMin']
df = data_df.drop(columns=drop_features)
info_names = [c[3:] for c in df.columns if c.startswith('red')]
for info in info_names:
    df['br' + info] = df['blue' + info] - df['red' + info]
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood'])
discrete_df = df.copy()
# print(discrete_df)
for c in df.columns[1:]:
    if len(df[c].unique()) <= 10:
        continue
    else:
        discrete_df[c] = pd.qcut(df[c], 4, labels=False)
all_y = discrete_df['blueWins'].values
feature_names = discrete_df.columns[1:]
all_x = discrete_df[feature_names].values

x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
all_y.shape, all_x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape # 输出数据行列信息

class DecisionTree(object):
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(self, classes, features, max_depth=10, min_samples_split=10, impurity_t='gini'):
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity_t = impurity_t
        self.root = None

    def impurity(self, labels):
        m = len(labels)
        return 1.0 - sum([(list(labels).count(c) / m) ** 2 for c in self.classes])

    def gain(self, left_labels, right_labels, current_impurity):
        p = float(len(left_labels)) / (len(left_labels) + len(right_labels))
        return current_impurity - p * self.impurity(left_labels) - (1 - p) * self.impurity(right_labels)

    def expand_node(self, feature, label, depth):
        unique_labels = np.unique(label)
        if len(unique_labels) == 1:
            return self.Node(value=unique_labels[0])

        if depth >= self.max_depth or len(feature) < self.min_samples_split:
            return self.Node(value=max(list(label), key=list(label).count))

        n_features = feature.shape[1]
        best_gain = 0
        best_feature_index = None
        best_threshold = None
        best_left = None
        best_right = None

        current_impurity = self.impurity(label)
        for feature_index in range(n_features):
            thresholds, classes = zip(*sorted(zip(feature[:, feature_index], label)))
            for i in range(1, len(feature)):
                if thresholds[i] == thresholds[i - 1]:
                    continue
                left_labels, right_labels = classes[:i], classes[i:]
                current_gain = self.gain(left_labels, right_labels, current_impurity)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature_index = feature_index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
                    best_left = np.array(left_labels)
                    best_right = np.array(right_labels)

        if best_gain == 0:
            return self.Node(value=max(list(label), key=list(label).count))
        left_node = self.expand_node(feature[feature[:, best_feature_index] <= best_threshold], best_left, depth + 1)
        right_node = self.expand_node(feature[feature[:, best_feature_index] > best_threshold], best_right, depth + 1)
        return self.Node(best_feature_index, best_threshold, left_node, right_node)

    def traverse_node(self, node, sample):
        if node.value is not None:
            return node.value
        feature_value = sample[node.feature]
        if feature_value <= node.threshold:
            return self.traverse_node(node.left, sample)
        return self.traverse_node(node.right, sample)

    def fit(self, feature, label):
        self.root = self.expand_node(feature, label, 1)

    def predict(self, feature):
        if len(feature.shape) == 1:
            return self.traverse_node(self.root, feature)
        return np.array([self.traverse_node(self.root, f) for f in feature])

DT = DecisionTree(classes=[0,1], features=feature_names, max_depth=5, min_samples_split=10, impurity_t='gini')
DT.fit(x_train, y_train)
print("QWQ")
p_test = DT.predict(x_test)
print(p_test)
test_acc = accuracy_score(p_test, y_test)
print('accuracy: {:.4f}'.format(test_acc))