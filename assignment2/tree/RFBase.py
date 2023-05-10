"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import var_red, riro_loss
import math
np.random.seed(42)

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    x= Y.value_counts(normalize=True)
    return -1*(x*np.log2(x)).sum()

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    x= Y.value_counts(normalize=True)
    return (x*(1-x)).sum()

def riro_loss(Y: pd.Series, attr: pd.Series) -> float:
    loss = 0
    for val in attr.unique():
        vals = Y[attr==val]
        mean = vals.mean()
        diff = vals-mean
        loss += (diff**2).sum()
    return -1*loss

def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    set_sizes=pd.crosstab(attr,Y).sum(axis=1)
    df_norm=pd.crosstab(attr,Y,normalize='index')

    df_entropy=-1*(df_norm*np.log2(df_norm)).fillna(0)
    entropies = df_entropy.sum(axis=1)

    return entropy(Y) - (entropies*(set_sizes/len(Y))).sum(axis=0)

class Node(object):
    def __init__(self, value,depth):
        self.value = value
        self.edge = None
        self.children = []
        self.depth = depth  

    def add_child(self, obj):
        self.children.append(obj)

@dataclass
class RFBase():
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int = np.inf # The maximum depth the tree can grow to
    _root_node : Node = Node(value=None,depth = 0)
    
    def __id3(self, examples, target_attribute, attributes, depth):


        node=Node(value=None, depth = depth)

        if(depth == self.max_depth):
            node.value = "Class "+str(target_attribute.mode()[0])
            return node

        if(len(target_attribute.unique()) == 1):
            node.value = "Class "+str(target_attribute.iloc[0])
            return node

        if(len(attributes)==0):
            node.value = "Class "+str(target_attribute.mode()[0])
            return node

        gain_list=[]
        for column in examples.columns:
            col=examples[column]
            if( self.criterion == 'information_gain') :
                inf_gain = information_gain(target_attribute,col)
            else:
                inf_gain = 0
                for val in col.unique():
                    targ_sub = target_attribute[col==val]
                    inf_gain += gini_index(targ_sub)*(len(targ_sub)/len(col))
            gain_list.append(inf_gain)

        node.value =  "(" + str(examples.columns[np.argmax(gain_list)]) + ")"
        col_name = examples.columns[np.argmax(gain_list)]

        for val in examples[col_name].unique():
            example_val = examples.loc[examples[col_name] == val].drop([col_name], axis=1).copy()
            target_attribute_val = target_attribute.loc[example_val.index].copy()
            attributes_val = np.delete(attributes, np.argwhere( attributes == col_name ))

            child_node = self.__id3(example_val, target_attribute_val, attributes_val, depth+1)
            child_node.edge = str(val) + ":"
            node.add_child(child_node)

        return node

    def __diro(self, examples, target_values, attributes,depth):
        node=Node(value=None,depth=depth)


        if(depth == self.max_depth):
            node.value = "Value "+str(target_values.mean())
            return node
        
        if(len(target_values.unique()) == 1):
            node.value = "Value "+str(target_values.iloc[0])
            return node

        if(len(attributes)==0):
            node.value = "Value "+str(target_values.mean())
            return node
        
        variance_gain_list=[]
        for column in examples.columns:
            col=examples[column]
            var_gain=var_red(target_values,col)
            variance_gain_list.append(var_gain)
        
        node.value =  "(" + str(examples.columns[np.argmax(variance_gain_list)]) + ")"
        col_name = examples.columns[np.argmax(variance_gain_list)]

        for val in examples[col_name].unique():
            example_val = examples.loc[examples[col_name] == val].drop([col_name], axis=1).copy()
            target_set_val = target_values.loc[example_val.index].copy()
            attributes_val = np.delete(attributes, np.argwhere( attributes == col_name ))

            child_node = self.__diro(example_val, target_set_val, attributes_val,depth+1)
            child_node.edge = str(val) + ":"
            node.add_child(child_node)
        
        return node

    def __rido(self, examples, target_attribute, attributes,depth):

        node=Node(value=None,depth=depth)
        examples_before_drop=examples.copy()
        target_attribute_bef_drop=target_attribute.copy()
        to_drop=2
        if(to_drop>len(examples.columns)):
            to_drop=0
            
        columns_all=examples.columns
        chosen_elements=np.random.choice(np.arange(len(columns_all)),to_drop)
        columns_to_drop=columns_all[chosen_elements]

        examples=examples.copy().drop(columns_to_drop,axis=1)
        attributes=examples.columns
        
        if(depth == self.max_depth):
            node.value = "Class "+str(target_attribute.mode()[0])
            return node
        
        if(len(target_attribute.unique()) == 1):
            node.value = "Class "+str(target_attribute.iloc[0])
            return node

        if(len(attributes)==0):
            node.value = "Class "+str(target_attribute.mode()[0])
            return node

        gain_list=[]#stores best information gain corresponging to a single column
        gain_split=[]#stres best split corresponging to a single column
        
        for column in examples.columns:
            example_copy=examples.copy()
            target_attribute_copy=target_attribute.copy()

            example_copy['values_target']=target_attribute_copy
            example_copy=example_copy.sort_values(by=[column])
            sorted_atrribute=example_copy['values_target']
            
            split_best = 0
            max_gain_split= -np.inf

            for i in range(len(sorted_atrribute)-1):
                split=(example_copy.loc[example_copy.index[i]][column]+example_copy.loc[example_copy.index[i+1]][column])/2
                col=example_copy[column]
                col=np.where(col<split, 0, 1)
                
                if( self.criterion == 'information_gain'):
                    inf_gain_split=information_gain(sorted_atrribute, col)
                    if(inf_gain_split>max_gain_split):
                        max_gain_split=inf_gain_split
                        split_best=split
                else:
                    inf_gain_split = 0
                    col_= pd.Series(col)
                    col_.index = sorted_atrribute.index
                    for val in col_.unique():
                        targ_sub = sorted_atrribute[col_==val]
                        inf_gain_split += gini_index(targ_sub)*(len(targ_sub)/len(col_))

                    if(inf_gain_split>max_gain_split):
                        max_gain_split=inf_gain_split
                        split_best=split
            
            gain_list.append(max_gain_split)
            gain_split.append(split_best)

        node.value =  "(" + str(examples.columns[np.argmax(gain_list)]) + ")"
        col_name = examples.columns[np.argmax(gain_list)]
        split= gain_split[np.argmax(gain_list)]
        
        example_val_0 = examples.loc[examples[col_name] <= split].copy()
        example_val_1=examples.loc[examples[col_name] > split].copy()

        target_attribute_val_0 = target_attribute.loc[example_val_0.index].copy()
        target_attribute_val_1 = target_attribute.loc[example_val_1.index].copy()
        
        child_node_0 = self.__rido(example_val_0, target_attribute_val_0, attributes,depth+1)
        child_node_1 = self.__rido(example_val_1, target_attribute_val_1, attributes,depth+1)

        child_node_0.edge = str(split) + "<:"
        node.add_child(child_node_0)

        child_node_1.edge = str(split) + ">:"
        node.add_child(child_node_1)

        return node

    def __riro(self, examples, target_values, attributes,depth):
        
        node=Node(value=None,depth=depth)

        to_drop=2
        if(to_drop>len(examples.columns)):
            to_drop=0

        columns_all=examples.columns
        chosen_elements=np.random.choice(np.arange(len(columns_all)),to_drop)
        columns_to_drop=columns_all[chosen_elements]

        examples=examples.copy().drop(columns_to_drop,axis=1)
        attributes=examples.columns

        if(depth == self.max_depth):
            node.value = "Value "+str(target_values.mean())
            return node

        if(len(target_values.unique()) == 1):
            node.value = "Value " + str(target_values.iloc[0])
            return node

        if(len(attributes)==0):
            node.value = "Value " + str(target_values.mean())
            return node
        
        gain_list=[]#stores best information gain corresponging to a single column
        gain_split=[]#stres best split corresponging to a single column
        
        for column in examples.columns:
            example_copy=examples.copy()
            target_value_copy=target_values.copy()

            example_copy['values_target']=target_value_copy
            example_copy=example_copy.sort_values(by=[column])
            sorted_atrribute=example_copy['values_target']

            split_best = 0
            max_split= -np.inf

            for i in range(len(sorted_atrribute)-1):
                split=(example_copy.loc[example_copy.index[i]][column]+example_copy.loc[example_copy.index[i+1]][column])/2
                col=example_copy[column]
                col=np.where(col<split, 0, 1)
                col=pd.Series(col)
                col.index=sorted_atrribute.index
                loss_split= riro_loss(sorted_atrribute,col)

                if(loss_split>max_split):
                    max_split=loss_split
                    split_best=split
            
            gain_list.append(max_split)
            gain_split.append(split_best)

        node.value =  "(" + str(examples.columns[np.argmax(gain_list)]) + ")"
        col_name = examples.columns[np.argmax(gain_list)]
        split= gain_split[np.argmax(gain_list)]
        
        ## splitting the dataset
        example_val_0 = examples.loc[examples[col_name] <= split].copy()
        example_val_1=examples.loc[examples[col_name] > split].copy()
        target_set_val_0 = target_values.loc[example_val_0.index].copy()
        target_set_val_1 = target_values.loc[example_val_1.index].copy()
 
        child_node_0 = self.__riro(example_val_0, target_set_val_0, attributes, depth+1)
        child_node_1 = self.__riro(example_val_1, target_set_val_1, attributes, depth+1)

        child_node_0.edge = str(split) + "<:"
        node.add_child(child_node_0)

        child_node_1.edge = str(split) + ">:"
        node.add_child(child_node_1)

        return node

         
    def __plot_helper(self, node, level=0):
        if(node.edge is None): #For root node
            ret = "  "*level + node.value + "\n"
        else:
            ret = "  "*level + node.edge + " " + node.value + "\n"

        for child in node.children:
            ret += self.__plot_helper(child, level+1)
        return ret

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        col = X.columns[0]
        #Discrete Input and Discrete Output
        if( str(X[col].dtype) == "category" and str(y.dtype) == "category" ):
            self._root_node = self.__id3(examples = X, target_attribute = y, attributes = X.columns.values, depth = 0)
            return self._root_node
        #Discrete Input real output
        elif( str(X[col].dtype) == "category" and str(y.dtype) != "category" ):
            self._root_node = self.__diro(examples = X, target_values = y, attributes = X.columns.values,depth=0)
            return self._root_node
        #Real Input real output
        elif( str(X[col].dtype) != "category" and str(y.dtype) == "category" ):
            self._root_node = self.__rido(examples = X, target_attribute = y, attributes = X.columns.values, depth=0)
            return self._root_node
        
        elif( str(X[col].dtype) != "category" and str(y.dtype) != "category" ):
            self._root_node = self.__riro(examples = X, target_values = y, attributes = X.columns.values,depth=0)
            return self._root_node
        

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        predictions=[]
        col = X.columns[0]
        
        if(str(X[col].dtype) == "category" ):
            for i in range(len(X)):
                x = X.iloc[i]
                traverse_node = self._root_node
        
                while(len(traverse_node.children) != 0):
                    childrens = traverse_node.children
                    for child in childrens:
                        if('Class' not in traverse_node.value): 
                            if( int(child.edge[:-1]) == x[int(traverse_node.value[1:-1])] ):
                                traverse_node = child
                                break

                        else:
                            if( int(child.edge[:-1]) == x[int(traverse_node.value[-2:])] ):
                                traverse_node = child  
                                break
                                
                predictions.append(traverse_node.value[6:])
        else:
            for i in range(len(X)):
                x = X.iloc[i]
                traverse_node = self._root_node
                
                while(len(traverse_node.children) != 0):
                    childrens = traverse_node.children
                    
                    left_child = childrens[0]
                    right_child = childrens[1]
                    split = float(left_child.edge[:-2])

                    if( x[int(traverse_node.value[1:-1])] <split):
                        traverse_node=left_child
                    else:
                        traverse_node=right_child
                                
                predictions.append(traverse_node.value[6:])


        return pd.Series(predictions).astype(np.float64)

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        print(self.__plot_helper(self._root_node))


# N = 30
# P = 5

# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randint(P, size = N), dtype="category")

# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randn(N))

# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randn(N))

# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randint(P, size = N), dtype="category")

# print(X)
# print(y)

# tree = DecisionTree(criterion='gini_index', max_depth=3) #Split based on Inf. Gain

# tree.fit(X, y)
# tree.plot()
# y_hat = tree.predict(X)