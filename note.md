## torch.gather
```
torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
```
将input dim维度的索引改为index，根据index重新设置元素，返回新的tensor。

对于2维：
```
out[i][j] = input[index[i][j]][j]  # if dim == 0
out[i][j] = input[i][index[i][j]]  # if dim == 1
```

对于3维：
```
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

举例：
```
>>> t = torch.tensor([[1,2],[3,4]])
>>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
tensor([[ 1,  1],
        [ 4,  3]])
```
从某一轴上根据索引选元素值，这里轴为1，0维度上轴1索引[0, 0]对应元素位置为(0,0)(0,0)，即[1, 1]，1维度上轴1索引[1, 0]对应元素位置为(1,1)(1,0)，即[4, 3]。

## Tensor.scatter_

```
Tensor.scatter_(dim, index, src, reduce=None) → Tensor
```
将tensor dim维度的值用src的index处的值替换，是gather的逆向操作。

对于3维：
```
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

举例：
```
>>> x = torch.rand(2, 5)
>>> x
tensor([[ 0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
        [ 0.5735,  0.9006,  0.6797,  0.4152,  0.1732]])
>>> torch.zeros(3, 5).scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
tensor([[ 0.3992,  0.9006,  0.6797,  0.4850,  0.6004],
        [ 0.0000,  0.2908,  0.0000,  0.4152,  0.0000],
        [ 0.5735,  0.0000,  0.9044,  0.0000,  0.1732]])
```

https://zhuanlan.zhihu.com/p/536323578

使用pointnet提取点的特征后，会将pillars（C,P）映射成pseudo images（C,H,W）,也就是pillarScatter操作。

https://blog.csdn.net/QLeelq/article/details/118807574