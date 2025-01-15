# Thesis

## 2025/01/15:
* Tried to recreate the Signal2Vec logic. Did k-Means -> SkipGram -> kNN
* ![image](https://github.com/user-attachments/assets/1f3c1216-c04e-4e08-ac8a-5cca9e4d1d1f)

## 2025/01/16
* Realized that I need to do it like this: k-Means -> kNN -> SkipGram
* Created a file 02, which attempts to use a pretrained model from embeddings created from file 01 (used ResNet 50, but got 51.2% probability of being Control for all 54 samples - in 10 epochs training)
