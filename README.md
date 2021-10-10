# News-Category-Classification

This a Bidirectional LSTM model that uses fasttext crawl embeddings to classify whether given sentence is one of the four categories, (Politics, Technology, Entertainment, and Business).

The model requires fasttext crawl embeddings which can be downloaded here. (https://fasttext.cc/docs/en/english-vectors.html).

The input text sentences are first tokenized using Keras Tokenizer and then padded to 150. These tokenized vectors are then passed to Neural Net.

The model has 7 different layers:
1) Embedding
2) Spatial Dropout
3) Bidirectional LSTM
4) Global Average Pooling
5) Global Max Pooling
6) Dropout
7) Dense 

The model is trained on keras and is deployed using FAST API.

The dataset in this repo was taken from Machine Hack as part of a competition. (https://machinehack.com/hackathons/predict_the_news_category_hackathon/data)
