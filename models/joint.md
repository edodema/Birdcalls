#Models
Soundscape detection models.
In Att the number is the size of the kernel since it is fixed that we have 3 all of the same size.
In CNNRes is the number of kernels.

## CNNAtt3CNNRes2GRU1FC1
```
==========================================================================================
Layer (type:depth-idx)                                            Param #
==========================================================================================
JointClassification                                               --
├─Classification: 1-1                                             --
│    └─Extraction: 2-1                                            --
│    │    └─CNNAtt: 3-1                                           459,284
│    │    └─CNNAtt: 3-2                                           459,284
│    │    └─Conv2d: 3-3                                           5
│    │    └─AvgPool2d: 3-4                                        --
│    │    └─Sequential: 3-5                                       896
│    │    └─AvgPool2d: 3-6                                        --
│    └─GRU: 2-2                                                   29,595,648
│    └─Linear: 2-3                                                407,950
==========================================================================================
Total params: 30,234,303
Trainable params: 30,234,303
Non-trainable params: 0
==========================================================================================
JointClassification(
  (model): Classification(
    (ext): Extraction(
      (cnnatt1): CNNAtt(
        (cnn_q): Sequential(
          (CNNQuery0): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (CNNQuery1): Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (CNNQuery2): Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (cnn_k): Sequential(
          (CNNKey0): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (CNNKey1): Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (CNNKey2): Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (cnn_v): Sequential(
          (CNNValue0): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (CNNValue1): Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (CNNValue2): Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (att1): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=313, out_features=313, bias=True)
        )
        (att2): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
      )
      (cnnatt2): CNNAtt(
        (cnn_q): Sequential(
          (CNNQuery0): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (CNNQuery1): Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (CNNQuery2): Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (cnn_k): Sequential(
          (CNNKey0): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (CNNKey1): Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (CNNKey2): Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (cnn_v): Sequential(
          (CNNValue0): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
          (CNNValue1): Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (CNNValue2): Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        )
        (att1): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=313, out_features=313, bias=True)
        )
        (att2): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
      )
      (cnn_att): Conv2d(4, 1, kernel_size=(1, 1), stride=(1, 1))
      (pool_att): AvgPool2d(kernel_size=1, stride=1, padding=0)
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
    (gru): GRU(9120, 512, bidirectional=True)
    (fc): Linear(in_features=1024, out_features=398, bias=True)
  )
)
```
## CNNRes2GRU1FC1
```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
JointClassification                      --
├─Classification: 1-1                    --
│    └─Extraction: 2-1                   --
│    │    └─Sequential: 3-1              896
│    │    └─AvgPool2d: 3-2               --
│    └─GRU: 2-2                          29,595,648
│    └─Linear: 2-3                       407,950
=================================================================
Total params: 30,004,494
Trainable params: 30,004,494
Non-trainable params: 0
=================================================================
JointClassification(
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
    (gru): GRU(9120, 512, bidirectional=True)
    (fc): Linear(in_features=1024, out_features=398, bias=True)
  )
)
```
## ResNet50
```
```
## ResNet18
```
```