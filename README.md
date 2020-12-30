# Caffe2PyTorch

This cookbook is a guide about how to transfer Caffe layers to PyTorch functions and networks.

## Guidance

- [Conv](#nnconv2d)
- [BN](#nnbatchnorm2d)
- [ReLU](#nnrelu)


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

