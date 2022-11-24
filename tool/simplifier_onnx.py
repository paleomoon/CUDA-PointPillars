# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import numpy as np
import onnx_graphsurgeon as gs
# 断开inputs和outputs，然后插入一层，重新连接起来
# 可以通过graph调用
@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    op_attrs["dense_shape"] = np.array([496,432])

    return self.layer(name="PPScatter_0", op="PPScatterPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

# 返回current_node的下loop_time个node，从1开始数
def loop_node(graph, current_node, loop_time=0):
  for i in range(loop_time):
    next_node = [node for node in graph.nodes if len(node.inputs) != 0 and len(current_node.outputs) != 0 and node.inputs[0] == current_node.outputs[0]][0] 
    current_node = next_node
  return next_node

def simplify_postprocess(onnx_model):
  print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)

  cls_preds = gs.Variable(name="cls_preds", dtype=np.float32, shape=(1, 248, 216, 18))
  box_preds = gs.Variable(name="box_preds", dtype=np.float32, shape=(1, 248, 216, 42))
  dir_cls_preds = gs.Variable(name="dir_cls_preds", dtype=np.float32, shape=(1, 248, 216, 12))

  tmap = graph.tensors()
  new_inputs = [tmap["voxels"], tmap["voxel_idxs"], tmap["voxel_num"]]
  new_outputs = [cls_preds, box_preds, dir_cls_preds]

  for inp in graph.inputs:
    if inp not in new_inputs:
      inp.outputs.clear() # 移除batch_size输入

  for out in graph.outputs:
    out.inputs.clear() # 断开所有输出节点

  first_ConvTranspose_node = [node for node in graph.nodes if node.op == "ConvTranspose"][0] # 原网络中第一个转置卷积
  concat_node = loop_node(graph, first_ConvTranspose_node, 3) # 找到concat node
  assert concat_node.op == "Concat"

  first_node_after_concat = [node for node in graph.nodes if len(node.inputs) != 0 and len(concat_node.outputs) != 0 and node.inputs[0] == concat_node.outputs[0]] # concat node下一层node，3个

  # 将输出接到transpose_node后面了，去掉了后面的一堆后处理op
  for i in range(3):
    transpose_node = loop_node(graph, first_node_after_concat[i], 1)
    assert transpose_node.op == "Transpose"
    transpose_node.outputs = [new_outputs[i]]

  graph.inputs = new_inputs
  graph.outputs = new_outputs
  graph.cleanup().toposort()
  
  return gs.export_onnx(graph)


def simplify_preprocess(onnx_model):
  print("Use onnx_graphsurgeon to modify onnx...")
  graph = gs.import_onnx(onnx_model)

  tmap = graph.tensors()
  MAX_VOXELS = tmap["voxels"].shape[0] # 10000

  # voxels: [V, P, C']
  # V is the maximum number of voxels per frame
  # P is the maximum number of points per voxel
  # C' is the number of channels(features) per point in voxels.
  input_new = gs.Variable(name="voxels", dtype=np.float32, shape=(MAX_VOXELS, 32, 10)) # 一帧点云样本的体素tensor，最多10000个体素，每个体素最多32个点，10通道(x,y,z,r,x_c,y_c,z_c,x_p,y_p,z_p)

  # voxel_idxs: [V, 4]
  # V is the maximum number of voxels per frame
  # 4 is just the length of indexs encoded as (frame_id, z, y, x).
  X = gs.Variable(name="voxel_idxs", dtype=np.int32, shape=(MAX_VOXELS, 4))

  # voxel_num: [1]
  # Gives valid voxels number for each frame
  Y = gs.Variable(name="voxel_num", dtype=np.int32, shape=(1,))

  first_node_after_pillarscatter = [node for node in graph.nodes if node.op == "Conv"][0] # 第一个卷积层Conv_335

  first_node_pillarvfe = [node for node in graph.nodes if node.op == "MatMul"][0] # 体素特征提取网络第一个节点MatMul_237

  next_node = current_node = first_node_pillarvfe
  for i in range(6):
    next_node = [node for node in graph.nodes if node.inputs[0] == current_node.outputs[0]][0]
    if i == 5:              # ReduceMax_242
      current_node.attrs['keepdims'] = [0]
      break
    current_node = next_node

  last_node_pillarvfe = current_node

  #merge some layers into one layer between inputs and outputs as below
  graph.inputs.append(Y)
  inputs = [last_node_pillarvfe.outputs[0], X, Y]
  outputs = [first_node_after_pillarscatter.inputs[0]]
  graph.replace_with_clip(inputs, outputs) # 至此ReduceMax_242和Conv_335之间被PPScatter_0代替

  # Remove the now-dangling subgraph.
  graph.cleanup().toposort()

  #just keep some layers between inputs and outputs as below
  graph.inputs = [first_node_pillarvfe.inputs[0] , X, Y] 
  graph.outputs = [tmap["cls_preds"], tmap["box_preds"], tmap["dir_cls_preds"]]

  graph.cleanup()

  #Rename the first tensor for the first layer 
  graph.inputs = [input_new, X, Y] # MatMul_237之前还有一大堆，用一个新输入voxels代替
  first_add = [node for node in graph.nodes if node.op == "MatMul"][0]
  first_add.inputs[0] = input_new

  graph.cleanup().toposort()

  return gs.export_onnx(graph)

if __name__ == '__main__':
    mode_file = "pointpillar-native-sim.onnx"
    simplify_preprocess(onnx.load(mode_file))
