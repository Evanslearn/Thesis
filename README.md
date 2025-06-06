# Thesis

## 2025/01/15:
* Tried to recreate the Signal2Vec logic. Did k-Means -> SkipGram -> kNN
* <img src="https://github.com/user-attachments/assets/1f3c1216-c04e-4e08-ac8a-5cca9e4d1d1f" width="900">


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
* * <img src="https://github.com/user-attachments/assets/e9b59a2a-d37e-4ca4-b259-f2aa803f8516" width="900">

## 2025/01/20
* Also tried timeseries from Pitt, and embedding with the wrong code. Results still not good, so need to fix signal2vec code

## 2025/01/26
* 01 logic should be ok now? At least it gave the expected dimension in output, and logic is K-Means -> kNN - > Skipgram, and output shape is (number of rows) x (number of features we want)
* Something is wrong with the classifier, as it's giving unstable results, where accuracy is usually oscillating, and predicted values are really similar for different instances that could have different labels.
* * <img src="https://github.com/user-attachments/assets/7cf908ab-2950-417c-aa9e-01b8a221f44f" width="900">

## 2025/01/27
* Tested window size 10, and a bit different learning rate. Used standardscaler, instead of max scaler.
* * <img src="https://github.com/user-attachments/assets/39d93eb3-9544-4d8d-a295-6f4f22fe3840" width="900">

## 2025/01/28-31
* Code cleanup + renames

## 2025/02/01-02
* Refactoring to remove data leakage. Sadly, this led to achieving significantly lower metric values, but is the correct way to go:
* <img src="https://github.com/user-attachments/assets/b25cb7bc-d441-4b3a-be32-1b1ad31e691e" width="900">
* Also tried 70-15-15 split instead of 70-10-20
* made an attempt to use stratification, but didn't make it yet. There are some considerations here, mostly about making sure we use the same instances between steps 2 and 3

## 2025/02/19
* Experimenting with parameters, trying to find a good train+val performance

## 2025/02/22
* Commited a fix, which implements correct pairing of remianing data with the labels, during step 1, due to removal of data depending on threshold

## 2025/02/26
* code cleanup + created another 03 file, to separate some models currently unused. still working on stratification and correct train-val-test split

## 2025/03/01 - 09
* Code cleanup + some more work on correct train-val-test split
## 2025/03/12 - 13
* code cleanup to make more readable and remove redundancy
## 2025/03/14-16
* still working on correct train-val-tet split + code cleanup
## 2025/03/25
* Did some changes to 02 file, as probably the logic there was wrong (was windowing across multiple rows, instead of windowing across each row)
## 2025/04/01
* Some changes/refactorings to the files. Now extracting raw timeseries from part1, so let's see how that goes
## 2025/04/06
* Commited changes/refactoring to all files. Files are now easier to use, and more generic/tunable
## 2025/04/21
* Might have found the issue with really bad performance, and it might have been scaling column-wise instead of row-wise, in Signal2Vec logic
* Commited some changes: Create results file for 01, scale row-wise and plots for clustering quality in 02, broke down utils into two parts (simple and plots)
## 2025/04/22
* 01 extract original sr, max freq, 02 plot most frequent tokens, 03 include f1 and confusion matrix
# 2025/04/23
* code cleanup for 01, 02, 03 03, 
    * Moved all write file logic to utils, added extra clustering evaluation metrics for kMeans (still optimize with silhouette)
    * logic for shuffling labels to challenge robustness of modal, plot for class labels distirbution per set, more clustering metrics in 02
# 2025/04/27
* Code cleanup + refactorings
    * small fixes here and there, rms clipping supported for all methods in 01, early stopping for SG, tokens_train in 02 from kMeans (used to be kNN also for train, while should have been only val and test)
    * Added cosine similarity calculation in 03
# 2025/05/03
* 01 improve MFCC logic to allow more components and also allow splitting an mp3 into chunks and doing mfcc in each chunk, 02 plots to check embedding quality also early stopping plot lines

# 2025/06/06
* Commit latest version that is cleaned up
