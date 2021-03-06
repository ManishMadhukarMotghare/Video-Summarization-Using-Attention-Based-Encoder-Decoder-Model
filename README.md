# Video-Summarization-Using-Attention-Based-Encoder-Decoder-Model
In this model we uses Attention Based model to summarize the video. The output of the video will be 15% of the original video as summary. Also It's a Supervised Model.

## Model
Encoder Decoder based Model with Attention Mechanism

## Encoder 
Multi-Layered Bi-Directional LSTM Network

## Decoder
LSTM Network

## Key Shots
Generated by solving 0/1 knapsack problem with help of Dynamic Programming

## Motivation
As in today’s era we go through tons of video graphic content offline online like youtube, learning classes videos, news and sports etc . We don't have enough time to go through all lengthy time consuming videos. So there is a need for a video summarizer that is efficient and fast enough to meet our conditions. That can be done either in the form of keyshots or keyframes.

## Overview
This documentation addresses the problem of supervised video summarization by formulating it as a sequence-to-sequence learning problem, where the input is a sequence of original video frames, the output is a keyshot sequence. Our key idea is to learn a deep summarization network with an attention mechanism to mimic the way of selecting the keyshots of a human. To this end, we propose a novel video summarization framework named Attentive encoder-decoder networks for Video Summarization (AVS), in which the encoder uses a Bidirectional Long Short-Term Memory (BiLSTM) to encode the contextual information among the input video frames. As for the decoder, two attention-based LSTM networks are explored by using additive and multiplicative objective functions, respectively. Extensive experiments are conducted on three video summarization benchmark datasets.

## Earlier Development and Issue
In Earlier LSTM encoder decoder models we used to encode all the necessary info in one single vector. This didn't allow to give different weights to different frames,which gave the same temporal importance to all shots/frames. So the significance of temporal structure was lost. 
To overcome this issue Attention Based weights in encoder decoder framework was used.This framework was named Attentive encoder-decoder networks for Video Summarization (AVS).

In this framework a single layer of BiLSTM was employed to encoder decoder which gave importance to past and future frames/shots simultaneously. Accuracy was a major concern in our base attention model that we tried to resolve in our model.

![Single Layer encoder decoder network](https://user-images.githubusercontent.com/40494282/152011785-f67e1497-ea5d-4941-816c-c3ba4059f7f5.JPG)


## Our Approach
In our base model a single BiLSTM layer was used while training the encoder decoder but we altered that by increasing and  adding a multilayer BiLstm network. This increased the accuracy to a significant level that can be tallied from results and compared with previous models.

![Multi Layer Encoder Decoder Network](https://user-images.githubusercontent.com/40494282/152011875-fc399733-9ecd-4d8d-849c-3716cc5c083c.JPG)

## Key Modules
1.Encoder with Multi Layer Bidirectional LSTM Network : In our model different from base model 2 extra back to back BiLstm layers are connected other than base BiLstm network.
2. Decoder with Attention Mechanism : Decoder is a simple Lstm network whose output sequence is concatenated with an attention layer.
3. Key shots selection model : The key shots selection model aims at converting the frame-level importance scores into shot-level scores and generating summary with a length budget.
4. Summary : Summary is created by concatenating Each shot.Knapsacking according to required length of summary
5. Shots : Compute shot-level importance scores by taking an average of the frame importance scores within each shot.

## Dataset
We have used the same dataset as in our base model i.e Tvsum .Most of the videos in these datasets are 1 to 10 minutes in length. Specifically, TVSum contains 50 edited videos downloaded from YouTube in 10 categories, such as changing vehicle tires, getting vehicle unstuck, grooming an animal. The video  contents in both datasets are diverse and include both egocentric and third-person cameras.

## Results and Comparisions

### Improved Performance during Training Set and Test Set
![Improved Performance](https://user-images.githubusercontent.com/40494282/152011973-59e5e504-24e7-441b-a00c-292b128d0396.JPG)

![comparision with other models](https://user-images.githubusercontent.com/40494282/152012022-e04c8884-024b-425f-8b45-477b7a1a8747.JPG)

### Performance Comparision for Training Set
![Performance Comparision](https://user-images.githubusercontent.com/40494282/152012103-afb9beac-3466-4800-9fff-e1780c55cece.JPG)

### Performance Comparision for Test Set
![Performance Comparision for Test Set](https://user-images.githubusercontent.com/40494282/152012126-7557fabc-2166-472b-a4e4-5c20786ddb38.JPG)

## Performance Comparision of Metrics

### Model Improvements
![Model Improvement](https://user-images.githubusercontent.com/40494282/152012167-9b6a2f38-6a1d-4daf-8f49-5bf7680d18cb.JPG)

### Improved Performance during Training Set and Test Set
![Improved Performance for both Sets](https://user-images.githubusercontent.com/40494282/152012211-ffc5c391-d3fa-4fd5-931a-e8bce9435b88.JPG)


