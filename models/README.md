# Models

Here are shown some models (yet not all) that have been used nor tested,the the name of a model can broadly describe its architecture according to the following scheme:
- In CNNAtt*k* we always have 3 kernels of the same size *k*.
- In CNNRes*n* the size of the kernels is not fixed, we have *n* of them and in this case the name is not enough to describe the model.
- GRU*m* and FC*m* means that we have *m* GRUs or FCs layers stacked.

## CNNRes3GRU4FC2
```
=================================================================  
Layer (type:depth-idx)                   Param #  
=================================================================  
SoundscapeDetection                      --  
├─Detection: 1-1                         --  
│    └─Extraction: 2-1                   --  
│    │    └─Sequential: 3-1              1,568  
│    │    └─AvgPool2d: 3-2               --  
│    └─GRU: 2-2                          29,761,536  
│    └─Sequential: 2-3                   --  
│    │    └─Linear: 3-3                  524,800  
│    │    └─Linear: 3-4                  513  
├─BCEWithLogitsLoss: 1-2                 --  
├─Accuracy: 1-3                          --  
├─Accuracy: 1-4                          --  
├─Accuracy: 1-5                          --  
├─Precision: 1-6                         --  
├─Precision: 1-7                         --  
├─Precision: 1-8                         --  
├─Recall: 1-9                            --  
├─Recall: 1-10                           --  
├─Recall: 1-11                           --  
├─ConfusionMatrix: 1-12                  --  
=================================================================  
Total params: 30,288,417  
Trainable params: 30,288,417  
Non-trainable params: 0  
=================================================================  
SoundscapeDetection(  
  (model): Detection(  
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
          (conv1): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv2): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv3): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv4): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv5): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv6): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
        )  
        (CNN2): Conv2d(2, 4, kernel_size=(3, 3), stride=(2, 2))  
        (CNNRes3): CNNRes(  
          (conv1): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv2): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv3): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv4): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv5): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
          (conv6): Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
        )  
        (CNN3): Conv2d(4, 8, kernel_size=(3, 3), stride=(2, 2))  
      )  
      (pool): AvgPool2d(kernel_size=1, stride=1, padding=0)  
    )  
    (gru): GRU(4560, 512, num_layers=4, dropout=0.2, bidirectional=True)  
    (fc): Sequential(  
      (0): Linear(in_features=1024, out_features=512, bias=True)  
      (1): Linear(in_features=512, out_features=1, bias=True)  
    )  
  )  
  (loss): BCEWithLogitsLoss()  
  (train_accuracy): Accuracy()  
  (val_accuracy): Accuracy()  
  (test_accuracy): Accuracy()  
  (train_precision): Precision()  
  (val_precision): Precision()  
  (test_precision): Precision()  
  (train_recall): Recall()  
  (val_recall): Recall()  
  (test_recall): Recall()  
  (conf_mat): ConfusionMatrix()  
)
```
## CNNRes2GRU4FC2
```
=================================================================  
Layer (type:depth-idx)                   Param #  
================================================================= 
SoundscapeDetection                      --  
├─Detection: 1-1                         --  
│    └─Extraction: 2-1                   --  
│    │    └─Sequential: 3-1              384  
│    │    └─AvgPool2d: 3-2               --  
│    └─GRU: 2-2                          45,084,672  
│    └─Sequential: 2-3                   --  
│    │    └─Linear: 3-3                  524,800  
│    │    └─Linear: 3-4                  513  
├─BCEWithLogitsLoss: 1-2                 --  
├─Accuracy: 1-3                          --  
├─Accuracy: 1-4                          --  
├─Accuracy: 1-5                          --  
├─Precision: 1-6                         --  
├─Precision: 1-7                         --  
├─Precision: 1-8                         --  
├─Recall: 1-9                            --  
├─Recall: 1-10                           --  
├─Recall: 1-11                           --  
├─ConfusionMatrix: 1-12                  --  
=================================================================  
Total params: 45,610,369  
Trainable params: 45,610,369  
Non-trainable params: 0  
=================================================================  
SoundscapeDetection(  
  (model): Detection(  
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
          (conv1): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv3): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv4): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv5): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv6): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (CNN2): Conv2d(2, 4, kernel_size=(3, 3), stride=(2, 2))
      )
      (pool): AvgPool2d(kernel_size=1, stride=1, padding=0)
    )
    (gru): GRU(9548, 512, num_layers=4, dropout=0.2, bidirectional=True)
    (fc): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): Linear(in_features=512, out_features=1, bias=True)
    )
  )
  (loss): BCEWithLogitsLoss()
  (train_accuracy): Accuracy()
  (val_accuracy): Accuracy()
  (test_accuracy): Accuracy()
  (train_precision): Precision()
  (val_precision): Precision()
  (test_precision): Precision()
  (train_recall): Recall()
  (val_recall): Recall()
  (test_recall): Recall()
  (conf_mat): ConfusionMatrix()
)
```
## CNNRes2GRU1FC1
```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
SoundscapeDetection                      --
├─Detection: 1-1                         --
│    └─Extraction: 2-1                   --
│    │    └─Sequential: 3-1              896
│    │    └─AvgPool2d: 3-2               --
│    └─GRU: 2-2                          29,595,648
│    └─Linear: 2-3                       1,025
=================================================================
Total params: 29,597,569
Trainable params: 29,597,569
Non-trainable params: 0
=================================================================
SoundscapeDetection(
  (model): Detection(
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
    (fc): Linear(in_features=1024, out_features=1, bias=True)
  )
)
```
## CNNAtt3CNNRes2GRU1FC1
```
==========================================================================================
Layer (type:depth-idx)                                            Param #
==========================================================================================
SoundscapeDetection                                               --
├─Detection: 1-1                                                  --
│    └─Extraction: 2-1                                            --
│    │    └─CNNAtt: 3-1                                           459,284
│    │    └─CNNAtt: 3-2                                           459,284
│    │    └─Conv2d: 3-3                                           5
│    │    └─AvgPool2d: 3-4                                        --
│    │    └─Sequential: 3-5                                       896
│    │    └─AvgPool2d: 3-6                                        --
│    └─GRU: 2-2                                                   29,595,648
│    └─Linear: 2-3                                                1,025
==========================================================================================
Total params: 29,827,378
Trainable params: 29,827,378
Non-trainable params: 0
==========================================================================================
SoundscapeDetection(
  (model): Detection(
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
    (fc): Linear(in_features=1024, out_features=1, bias=True)
  )
)
```
## ResNet18
```
===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
SoundscapeDetection                                --
├─Detection: 1-1                                   --
│    └─Conv2d: 2-1                                 6
│    └─ResNet: 2-2                                 --
│    │    └─Conv2d: 3-1                            9,408
│    │    └─BatchNorm2d: 3-2                       128
│    │    └─ReLU: 3-3                              --
│    │    └─MaxPool2d: 3-4                         --
│    │    └─Sequential: 3-5                        147,968
│    │    └─Sequential: 3-6                        525,568
│    │    └─Sequential: 3-7                        2,099,712
│    │    └─Sequential: 3-8                        8,393,728
│    │    └─AdaptiveAvgPool2d: 3-9                 --
│    │    └─Linear: 3-10                           513,000
│    └─Linear: 2-3                                 1,001
===========================================================================
Total params: 11,690,519
Trainable params: 11,690,519
Non-trainable params: 0
===========================================================================
SoundscapeDetection(
  (model): Detection(
    (cnn): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
    (resnet): ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )
    (fc): Linear(in_features=1000, out_features=1, bias=True)
  )
)
```
## Resnet50
```
===========================================================================
Layer (type:depth-idx)                             Param #
===========================================================================
SoundscapeDetection                                --
├─Detection: 1-1                                   --
│    └─Conv2d: 2-1                                 6
│    └─ResNet: 2-2                                 --
│    │    └─Conv2d: 3-1                            9,408
│    │    └─BatchNorm2d: 3-2                       128
│    │    └─ReLU: 3-3                              --
│    │    └─MaxPool2d: 3-4                         --
│    │    └─Sequential: 3-5                        215,808
│    │    └─Sequential: 3-6                        1,219,584
│    │    └─Sequential: 3-7                        7,098,368
│    │    └─Sequential: 3-8                        14,964,736
│    │    └─AdaptiveAvgPool2d: 3-9                 --
│    │    └─Linear: 3-10                           2,049,000
│    └─Linear: 2-3                                 1,001
===========================================================================
Total params: 25,558,039
Trainable params: 25,558,039
Non-trainable params: 0
===========================================================================
SoundscapeDetection(
  (model): Detection(
    (cnn): Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1))
    (resnet): ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=2048, out_features=1000, bias=True)
    )
    (fc): Linear(in_features=1000, out_features=1, bias=True)
  )
)
```


