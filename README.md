# Thesis

## 2025/01/15:
* Tried to recreate the Signal2Vec logic. Did k-Means -> SkipGram -> kNN
* <img src="https://github.com/user-attachments/assets/1f3c1216-c04e-4e08-ac8a-5cca9e4d1d1f" width="600">


## 2025/01/16
* Realized that I need to do it like this: k-Means -> kNN -> SkipGram, so my current implementation is not correct.
* I have extracted some embeddings with this "wrong implementation", to use in a pre-trained NN.
* Created a file 02, which attempts to use a pretrained model from embeddings created from file 01. Used ResNet 50, and got
  * 51.2% probability of being Control for all 54 samples - in 10 epochs training
  * 97.4% probability of being Dementia for all 54 samples - in 30 epochs training
 
 ## 2025/01/17
 * created another Code 01 file, in an attempt to make the logic of creating the embeddings correctly. Not finished and/or working yet

## 2025/01/19
* Tried some more models for the created embeddings, as it troubled me that I initially used a pretrained CNN (and CNN for embeddings from time-series sounds strange...). The model is still problematic, as it's not learning, and does the same for each epch.
* * <img src="https://github.com/user-attachments/assets/e9b59a2a-d37e-4ca4-b259-f2aa803f8516" width="600">

## 2025/01/20
* Also tried timeseries from Pitt, and embedding with the wrong code. Results still not good, so need to fix signal2vec code

## 2025/01/26
* 01 logic should be ok now? At least it gave the expected dimension in output, and logic is K-Means -> kNN - > Skipgram, and output shape is (number of rows) x (number of features we want)
* Something is wrong with the classifier, as it's giving unstable results, where accuracy is usually oscillating, and predicted values are really similar for different instances that could have different labels.
* * <img src="https://github.com/user-attachments/assets/9bce0df9843607022e45ac3ef46a9815c0eef8d5 width="600">
