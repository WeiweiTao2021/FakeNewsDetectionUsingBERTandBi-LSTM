# FakeNewsDetectionUsingBERTandBi-LSTM

## Fake News Detection ##

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset <br/>
Additional Datasets: 
1. https://github.com/phHartl/lrec_2022 <br />
2. https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php<br />
3. https://paperswithcode.com/dataset/realnews <br />
4. https://paperswithcode.com/dataset/fakenewsnet <br />

## Data preparation
1. https://github.com/MITsVision/Linkly/blob/master/code/data_combination.ipynb
2. https://github.com/MITsVision/Linkly/blob/master/code/combine_csv.ipynb
3. Downloading the twitter metadata: https://github.com/KaiDMML/FakeNewsNet/blob/master/README.md

### Baseline
1. File [most of the codes are included in this notebook]: https://github.com/MITsVision/Linkly/blob/master/code/baseline.ipynb

2. Unfinished Basline with glove: https://github.com/MITsVision/Linkly/blob/master/code/using_glove_emb.ipynb

### Bi-LSTM Model Implementation ###

Glove_biLSTM.ipynb conatins the code to train, test and load Bi-LSTM model. The notebook can be found at https://github.com/MITsVision/Linkly/blob/master/Bi-LSTM/Glove_biLSTM.ipynb and contains the following sections:

<p align="center">
  <img width="500" height="300" src="https://user-images.githubusercontent.com/31389737/206339554-f8f23d3e-be18-41b3-bc2f-92676a475d84.png">
</p>

1. Cleaning the data and removing stop words
2. Tokenizing words and loading GloVe vectors
3. Converting tokeinzed text to word embeddings
4. Training our model using FakeNewsNet dataset
5. Using tensorborad to see training charts
6. Saving the model
7. Loading the model and test on the testing dataset
8. Calculate `word_importance` for each word in testing dataset
9. Save `word_importance` to `test_result.json`

### BERT Model Implementation ###
The code to train and test BERT model is named as BERT_FreezeEmb_OneForwardLayer.ipynb. The code was developed using Google Colab.
The source code can be located at https://github.com/MITsVision/Linkly/blob/master/BERT/BERT_FreezeEmb_OneForwardLayer.ipynb
There are following tunable hyperparameters: 
1. The name of the pretrained model from Hugging face. On default is it bert-base-cased.
2. Train/Test splitting ratio: default 60% versus 40%
3. Test/Validating splitting ratio: default 50% versus 50%
4. Batch size: default 5
5. Max token length: the max allowed token length for BERT is 512.
6. Number of Epochs: default is 2. Two epochs is enough to yield a very high classificaiton accuracy for BERT.
7. Freeze embedding: whether to disable the finetune BERT word embeddings. The default is True.
7. Freeze all parameters: whether to disable the finetune all BERT parameters. The default is False.

The code contains following sections:
1. Hyperparameter setting
2. Load and preprocess the dataset
3. Train-test splitting
4. Generate the encoded dataset
5. Define BERT model
6. Model training
7. Prediction on FakeNewsNet validaiton set
8. Prediction on BuzzFeed dataset
9. Generate word importance for model inferece


### Goal ###
Build model for fake news detection and develop algorithm to perform model inferece.

### Report ###

Overleaf link for project final report: https://www.overleaf.com/8635424853ttjqtngwmktj


## References
1. https://onlinelibrary.wiley.com/doi/epdf/10.1002/spy2.9
2. https://www.researchgate.net/publication/320300831_Detection_of_Online_Fake_News_Using_N-Gram_Analysis_and_Machine_Learning_Techniques
3. https://arxiv.org/pdf/1811.00770.pdf
4. https://arxiv.org/pdf/1809.01286.pdf
5. https://ieeexplore.ieee.org/document/9074071
6. https://www.sciencedirect.com/science/article/pii/S2666307422000092

## Useful Articles
1. https://towardsdatascience.com/election-special-detect-fake-news-using-transformers-3e38d0c5b5c7


