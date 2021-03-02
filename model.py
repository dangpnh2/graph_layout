from collections import defaultdict
import os
import sys
import argparse
import subprocess
import pdb
import time
import random
import _pickle as cPickle
import glob
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import copy
from tree import get_topic_idxs, get_child_to_parent_idxs, get_depth, get_ancestor_idxs, get_descendant_idxs
import itertools

def get_level_nodes(tree_depth):
    level_nodes = defaultdict(list)
    for level in range(3):
        for key, value in tree_depth.items():
            if value == level+1:
                level_nodes[level].append(key)
    level_nodes = dict(level_nodes)
    return level_nodes
def get_leaf_parents(tree_idxs, level_nodes, n_depth):
    leaf_parents = {x:[0] for x in level_nodes[n_depth-1]}

    for node in level_nodes[1]:
        for leaf, parent in leaf_parents.items():
          if leaf in tree_idxs[node]:
            parent.append(node)
    return  leaf_parents

def distance_1(C_xy, idx1, idx2):
    return tf.sqrt(tf.reduce_sum((C_xy[idx1] - C_xy[idx2])*(C_xy[idx1] - C_xy[idx2])))#tf.sqrt(tf.reduce_sum((C_xy[idx1] - C_xy[idx2])*(C_xy[idx1] - C_xy[idx2])))


def distance_matrix(topic_coords, level_nodes, leaf_parents, topics):
    graph_pair_dist = {}
    leaf_prev_parent = {e:max(i) for e,i in leaf_parents.items()}
    
    for leaf, parents in leaf_parents.items():
        prev_a = max(parents)
        d_ancestors = 2#distance_1(topic_coords, 0, prev_a)
        d_leaf_prev_a = 1#distance_1(topic_coords, leaf, prev_a)   

        graph_pair_dist[(0, leaf)] = d_ancestors + d_leaf_prev_a
        graph_pair_dist[(prev_a, leaf)] = d_leaf_prev_a
        graph_pair_dist[(parents[0], parents[1])] = d_ancestors
    

    for level, nodes in level_nodes.items():

            for combination in itertools.combinations(nodes, 2):
                v1 = min(combination)
                v2 = max(combination)
                if level==1:
                    graph_pair_dist[(v1, v2)] = graph_pair_dist[(0,v1)] + graph_pair_dist[(0, v2)]
                if level==2:
                    if leaf_prev_parent[v1] == leaf_prev_parent[v2]: #same parent
                        graph_pair_dist[(v1, v2)] = graph_pair_dist[(leaf_prev_parent[v1],v1)] + graph_pair_dist[(leaf_prev_parent[v2], v2)] #distance_1(topic_coords, leaf_prev_parent[v1], v1) + distance_1(topic_coords, leaf_prev_parent[v2], v2)
                    else: #different parent
                        graph_pair_dist[(v1, v2)] = graph_pair_dist[(0,v1)] + graph_pair_dist[(0, v2)]

    for i, j in itertools.product(level_nodes[1], level_nodes[2]):
        if (i,j) not in graph_pair_dist:
            graph_pair_dist[(i,j)] = graph_pair_dist[(0,i)] + graph_pair_dist[(0,j)]
                               
    return graph_pair_dist

def compute_topic_coord_reg_kk(topic_coords, max_d, level_nodes, tree_idxs, child_parents, topic_idxs):
    cost = 0.
    graph_pair_dist = distance_matrix(topic_coords, level_nodes, child_parents, topic_idxs)
    

    for (i,j) in itertools.combinations(topic_idxs, 2):
        temp = graph_pair_dist[(i,j)]
        d = distance_1(topic_coords, i, j)

        cost += (1./2.)*((d-temp)/temp)**2

    return cost
class KKLayout():
    def __init__(self, config):
        self.config = config

        self.t_variables = {}
        self.tree_idxs = config.tree_idxs
        self.topic_idxs = get_topic_idxs(self.tree_idxs)
        self.child_to_parent_idxs = get_child_to_parent_idxs(self.tree_idxs)
        self.tree_depth = get_depth(self.tree_idxs)
        self.n_depth = max(self.tree_depth.values())
        
        
        self.level_nodes = get_level_nodes(self.tree_depth)
        self.leaf_parents = get_leaf_parents(self.tree_idxs, self.level_nodes, self.n_depth)

        self.node_level = {}
        for level, nodes in self.level_nodes.items():
            for node in nodes:
                self.node_level[node] = level
        self.build()
        
    def build(self):

        # -------------- Build Model --------------
        tf.reset_default_graph()
        tf.set_random_seed(self.config.seed)
        self.t_variables['max_d'] = tf.placeholder(tf.float32)
        

        with tf.variable_scope('topic/enc', reuse=False):
            self.topic_bn = tf.layers.BatchNormalization()

            self.topic_coords = { i : self.topic_bn(tf.get_variable('topic_coords'+str(i), [1, 2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1))) for i in self.topic_idxs}#tf.random_normal_initializer(mean=0.0, stddev=0.01)

        # define losses
        self.global_step = tf.Variable(0, name='global_step',trainable=False)

        self.coord_reg3 = compute_topic_coord_reg_kk(self.topic_coords, self.t_variables['max_d'], self.level_nodes, self.tree_idxs, self.leaf_parents, self.topic_idxs)
        #reg4 = d_node_root(self.topic_coords, self.tree_idxs)
        self.loss = self.coord_reg3 #-reg4

        # define optimizer
        if self.config.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif self.config.opt == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(self.config.lr)

        self.grad_vars = optimizer.compute_gradients(self.loss)
        #self.clipped_grad_vars = [(tf.clip_by_value(grad, -self.config.grad_clip, self.config.grad_clip), var) for grad, var in self.grad_vars]
        self.opt = optimizer.apply_gradients(self.grad_vars, global_step=self.global_step)

    
    def get_feed_dict(self, max_d):
        feed_dict = {
                    self.t_variables['max_d']: max_d
        }
        return  feed_dict