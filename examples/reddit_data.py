# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import sys
import subprocess
import urllib
import zipfile

import networkx as nx
import numpy as np
import scipy.sparse as sp

from networkx.readwrite import json_graph
from euler.tools import json2dat

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <=
                         11), "networkx major version > 1.11, should be 1.11"


def load_data(prefix, normalize=True, load_walks=False):
  coo_adj = sp.load_npz(prefix + 'reddit_self_loop_graph.npz')
  G = nx.from_scipy_sparse_matrix(coo_adj)
  reddit_data = np.load(prefix + 'reddit_data.npz')
  feats = reddit_data['feature']

  id_map = list(map(int, reddit_data['node_ids']))
  class_map = list(map(int, reddit_data['label']))
  node_types = list(map(int, reddit_data['node_types']))

  ## Make sure the graph has edge train_removed annotations
  ## (some datasets might already have this..)
  print("Loaded data.. now preprocessing..")
  return G, feats, id_map, class_map, node_types


def node_type_id(node_types, node_id):
  return node_types[node_id] - 1

def one_hot(x, num_classes):
  return [0.] * x + [1.] + [0.] * (num_classes - 1 - x)

def convert_data(prefix):
  with_weight = False
  G, feats, id_map, class_map, node_types = load_data(prefix)

  meta = {
      "node_type_num": 3,
      "edge_type_num": 1,
      "node_uint64_feature_num": 0,
      "node_float_feature_num": 2,  # 0 label and 1 for feature
      "node_binary_feature_num": 0,
      "edge_uint64_feature_num": 0,
      "edge_float_feature_num": 0,
      "edge_binary_feature_num": 0
  }

  meta_out = open(prefix + '_meta.json', 'w')
  meta_out.write(json.dumps(meta))
  meta_out.close()

  out_val = open(prefix + '_val.id', 'w')
  out_train = open(prefix + '_train.id', 'w')
  out_test = open(prefix + '_test.id', 'w')
  out_vec = [out_train, out_val, out_test]
  out = open(prefix + '_data.json', 'w')
  for node in G.nodes():
    node = int(node)
    buf = {}
    buf["node_id"] = node
    buf["node_type"] = node_type_id(node_types, node)
    out_vec[node_type_id(node_types, node)].write(str(id_map[node]) + '\n')
    buf["node_weight"] = len(G[node]) if with_weight else 1
    buf["neighbor"] = {}
    for i in range(0, meta["edge_type_num"]):
      buf["neighbor"][str(i)] = {}
    for n in G[node]:
      buf["neighbor"]['0'][str(n)] = 1
    buf["uint64_feature"] = {}
    buf["float_feature"] = {}
    buf["float_feature"][0] = one_hot(class_map[node], 41)
    buf["float_feature"][1] = list(feats[node])
    buf["binary_feature"] = {}
    buf["edge"] = []
    for tar in G[node]:
      tar = int(tar)
      ebuf = {}
      ebuf["src_id"] = node
      ebuf["dst_id"] = tar
      ebuf["edge_type"] = 0
      ebuf["weight"] = 1
      ebuf["uint64_feature"] = {}
      ebuf["float_feature"] = {}
      ebuf["binary_feature"] = {}
      buf["edge"].append(ebuf)
    out.write(json.dumps(buf) + '\n')
  out.close()
  for i in out_vec:
    i.close()


if __name__ == '__main__':
  print('download reddit data..')
  url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit_self_loop.zip'
  urllib.urlretrieve(url, 'reddit.zip')
  with zipfile.ZipFile('reddit.zip') as reddit_zip:
    print('unzip data..')
    reddit_zip.extractall('reddit/')

  prefix = 'reddit/'
  convert_data(prefix)
  c = json2dat.Converter(prefix + '_meta.json', prefix + '_data.json',
                         prefix + '_data.dat')
  c.do()
