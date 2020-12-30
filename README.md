# Caffe2PyTorch

This cookbook is a guide about how to transfer Caffe layers to PyTorch functions and networks.

## Guidance

- [Conv](#nnconv2d)
- [BN](#nnbatchnorm2d)
- [ReLU](#nnrelu)
- [Reshape](#torchreshape)
- [Transpose](#torchtranspose)
- [Matrix Mul](#torchbmm)
- [Softmax](#fsoftmax)
- [Crop](#crop)
- [Eltwise](#eltwise)
- [DeconV](#nnconvtranspose2d)
- [Concat](#concat)

---

###### nn.Conv2d
```python
nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride, bias)
```
```
layer {
  name: 
  type: "Convolution"
  bottom: 
  top: 
  convolution_param {
    num_output: 64
    bias_term: true
    pad: 3
    kernel_size: 7
    stride: 2
    weight_filler {
      type: "msra"
    }
  }
}
```

---

###### nn.BatchNorm2d 
```python
nn.BatchNorm2d(out_channels)
```
```
layer {
  name: 
  type: "BatchNorm"
  bottom: 
  top: 
  batch_norm_param {
    use_global_stats: false
  }
}
layer {
  name: 
  type: "Scale"
  bottom: 
  top: 
  scale_param {
    bias_term: true
  }
}
```

---

###### nn.ReLU 
```python
nn.ReLU(inplace=True)
```
```
layer {
  name: 
  type: "ReLU"
  bottom: 
  top: 
}
```

---

###### torch.reshape
```python
torch.reshape(x, (n, c, -1)) # caffe中dim=0表示维度不变
```
```
layer {
  name: 
  type: "Reshape"
  bottom: 
  top: 
  reshape_param {
    shape {
      dim: 0
      dim: 0
      dim: -1
    }
  }
}
```

---

###### torch.transpose
```python
torch.transpose(x, dim_0, dim_1)	# torch的transpose只能一次交换两个维度，多个维度需要多次交换
```
```
layer {
  name: 
  type: "TensorTranspose"
  bottom: 
  top: 
  tensor_transpose_param {
    order: 0
    order: 2
    order: 1
  }
}
```

---

###### torch.bmm
```python
torch.bmm(x, y)	# bmm针对batch做矩阵乘法，即(n, c, w, h)，如果是(c, w, h)则可以使用bm
```
```
layer {
  name: 
  type: "MatrixMultiplication"
  bottom: 
  bottom: 
  top: 
}
```

---

###### F.softmax
```python
F.softmax(x, dim)	# dim即为下面的axis
```
```
layer {
  name: 
  type: "Softmax"
  bottom: 
  top: 
  softmax_param {
    axis: 2
  }
}
```

---

###### Crop
```python
class Crop(nn.Module):
    '''
		@ axis		->		从axis开始裁减后面的维度
		@ offset	->		裁减的时候要偏离多少距离

		! 注意：在Caffe中，是以B的大小来裁减A，即 B.shape <= A.shape
	'''
    def __init__(self, axis, offset=0):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, indices.to(torch.int64))
        return x
```
```
layer {
  name: 
  type: "Crop"
  bottom: A
  bottom: B
  top: 
  crop_param {
    axis: 2
  }
}
```

---

###### Eltwise

```python
class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x =torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x
```

```
layer {
  name: 
  type: "Eltwise"
  bottom: 
  bottom: 
  top: 
}

layer {
  name: 
  type: "Eltwise"
  bottom: 
  bottom: 
  top: 
  eltwise_param {
    operation: PROD
  }
}
```

---

###### nn.ConvTranspose2d

```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
```
```
layer {
  name: 
  type: "Deconvolution"
  bottom: 
  top: 
  param {
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 0
    kernel_size: 4
    group: 128
    stride: 2
    weight_filler {
      type: "bilinear"
    }
  }
}
```

---

###### Concat
```python
class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)
```

```
layer {
  name: 
  type: "Concat"
  bottom: 
  bottom: 
  bottom: 
  top: 
  propagate_down: true
  propagate_down: true
  propagate_down: false
  concat_param {
    concat_dim: 1
  }
}
```
