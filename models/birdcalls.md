#Models
Soundscape detection models.
In Att the number is the size of the kernel since it is fixed that we have 3 all of the same size.
In CNNRes is the number of kernels.

## CNNRes2GRU1FC1
```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
BirdcallClassification                   --
├─Classification: 1-1                    --
│    └─Extraction: 2-1                   --
│    │    └─Sequential: 3-1              896
│    │    └─AvgPool2d: 3-2               --
│    └─GRU: 2-2                          346,626,048
│    └─Linear: 2-3                       406,925
=================================================================
Total params: 347,033,869
Trainable params: 347,033,869
Non-trainable params: 0
=================================================================
BirdcallClassification(
  (model): Classification(
    (ext): Extraction(
      (res): Sequential(
        (CNNRes1): CNNRes(
          (conv1): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv4): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv5): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv6): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (CNN1): Conv2d(1, 2, kernel_size=(3, 3), stride=(2, 2))
        (CNNRes2): CNNRes(
          (conv1): Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          (conv2): Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          (conv3): Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          (conv4): Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          (conv5): Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          (conv6): Conv2d(2, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )
        (CNN2): Conv2d(2, 4, kernel_size=(5, 5), stride=(2, 2))
      )
      (pool): AvgPool2d(kernel_size=1, stride=1, padding=0)
    )
    (gru): GRU(112320, 512, bidirectional=True)
    (fc): Linear(in_features=1024, out_features=397, bias=True)
  )
)
```
## CNNAtt3CNNRes2GRU1FC1
```
```
## ResNet18
```
```
## ResNet50
```
```