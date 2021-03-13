# Finding Chandler! Sarcasm-detection
### Problem Statement
Given a sentence , the proposed system has to classify whether it is "sarcastic" or not. \
Kaggle InClass competition : https://www.kaggle.com/c/finding-chandler-sarcasm

### Data
The training dataset consists of 39780 samples and the test set consists of 1975 samples. Each sample in the train dataset is labelled as 1 (sarcastic) or 0(not sarcastic). There are 21,292 postive and 18,488 negative samples. So, the classes are reasonably balanced.  

### Evaluation metric
The evaluation metric is the mean-F1 score \
F1 = 2 * precision * recall / (precision + recall) \
where \
precision = tp/(tp+fp) \
recall = tp/(tp+fn)

### Proposed system 
All of the sarcastic comments had  #sarcasm as a suffix. The model could pick this as a pattern to identify sarcastic comments. So, to prevent our model from overfitting, it is necessary to preprocess the dataset to remove such hastags. The text samples are then tokenized with Tokenizer (from keras.preprocessing.text module). The words are converted into 300 dimensional embeddings using GloVe (42B). These embedding are then fed into a Bi-directional LSTM layers with 200 units followed by a classifier consisting of 3 Dense layers as described in the table below. 

Model description :
| Layer | Activation | Dimension |
| :---: |:---:| :---:|
| BiLSTM | - | 200 |
| Dense | relu | 70 |
| Dense | leaky-relu | 20 |
| Dense | sigmoid | 1 |

The loss function used is "binary-crossentropy". The model is trained with adam optimizer with a batch size of 512. The model is trained for a total of 6 epochs.

This model ranked 6th in the competition with the mean F1 score of 0.90295. 
The model was trained on Colab. The code can be found [here]("./Chandler.ipynb")
