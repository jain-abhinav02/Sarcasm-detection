# Sarcasm-detection
### Problem Statement
Given a sentence , the proposed system has to classify whether it is "sarcastic" or not. \
Kaggle InClass competition : https://www.kaggle.com/c/finding-chandler-sarcasm

### Evaluation metric
The evaluation metric is the mean-F1 score \
F1 = 2 * precision * recall / (precision + recall) \
where \
precision = tp/(tp+fp) \
recall = tp/(tp+fn)

### Proposed system 
The text is tokenized with Tokenizer (from keras.preprocessing.text module). The words are converted into 300 dimensional embeddings using GloVe (42B). These embedding are then fed into a Bi-directional LSTM layers with 200 units followed by a classifier consisting of 3 Dense layers as described in the table below. 

Model description :
| Layer | Activation | Dimension |
| :---: |:---:| :---:|
| BiLSTM | - | 200 |
| Dense | relu | 70 |
| Dense | leaky-relu | 20 |
| Dense | sigmoid | 1 |

The loss function used is "binary-crossentropy". The model is trained with adam optimizer with a batch size of 512. The model is trained for a total of 6 epochs.
