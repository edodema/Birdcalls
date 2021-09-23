# BirdcallDetection
BirdCLEF2021.

Birds' directories are not random, it is a species' code. 
You can just put it in sites as [eBird](https://ebird.org/species/scatan) and [TheCornellLab](https://www.allaboutbirds.org/guide/scatan) and see it yourself.
eBird seems to be better.

- [x] It seems that tensors are normalized between (-1,1)
- [ ] Add some background knowledge in the report as 
  [here](https://medium.com/analytics-vidhya/audio-data-processing-feature-extraction-science-concepts-behind-them-be97fbd587d8), 
  [here](https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504),
  [here](https://towardsdatascience.com/detecting-sounds-with-deep-learning-ed9a41909da0)
  [melscale](https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056),
  [mel spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53),
- [x] Try some preprocessing with different sample rate.
- [x] Training in different times.
- [x] Try resampling to a fixed size.
- [x] A sample rate of 32000 means that in 1 second we have 32000 frames.
- [ ]  A radix major implementation for ResNet (see paper) was too problematic due to computation
- [ ] See Resnet paper 3.2 plain network recommendations.
- [ ] During validation soundscape datasets have been halved, birdcalls probably thirded.
- [ ] Occam's razor.
- [ ] When there are imbalances accuracy is useless.
- [ ] In birdcalls batches are too small.
- [ ] Future works add what you could have done with more resources.
- [ ] Future works: thresholding and secondary labels.
- [ ] Specify that models are trained on the seed, our goal is not to have a good on-the-wild ai.
- [ ] Soundscapes and Joint are divided following 80-10-10, Birdcalls according to a 70-15-15 split

## Environment 
```
conda create -n Birdcalls
conda activate Birdcalls
pip install -r requirements.txt
```
```
./setup.sh -dmo
```