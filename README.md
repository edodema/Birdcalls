# BirdcallDetection
BirdCLEF2021.

Birds' directories are not random, it is a species' code. 
You can just put it in sites as [eBird](https://ebird.org/species/scatan) and [TheCornellLab](https://www.allaboutbirds.org/guide/scatan) and see it yourself.
eBird seems to be better.


- [ ] It seems that tensors are normalized between (-1,1)
- [ ] Add some background knowledge in the report as 
  [here](https://medium.com/analytics-vidhya/audio-data-processing-feature-extraction-science-concepts-behind-them-be97fbd587d8), 
  [here](https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504),
  [here](https://towardsdatascience.com/detecting-sounds-with-deep-learning-ed9a41909da0)
  [melscale](https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056),
  [mel spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53),
- [ ] Try some preprocessing with differend sample rate.
- [ ] Training in different times.
- [ ] Try resampling to a fixed size.
- [ ] Remember you just need to add ```self=...``` in hydra's call.
- [ ] Use **denoising autoencoders** to resolve the cocktail party problem, maybe use mean birdcalld as clean data.
- [ ] A sample rate of 32000 means that in 1 second we have 32000 frames.
- [ ] Organize methods in a clearer way as Pellacini did.
- [ ] Write a function that precomputes tensors, this way training will be easier.
- [ ]  A radix major implementation for ResNet (see paper) was too problematic due to computation
- [ ] See Resnet paper 3.2 plain network recommendations.
- [ ] During validation soundcalls datasets have been halved, birdcalls probablt thirded.
- [ ] Occam's razor.
- [ ] In env.template write the path as an explanation of the variable