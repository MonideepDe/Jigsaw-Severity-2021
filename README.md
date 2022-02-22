# Intro

Kaggle competition - 2021

# Data
Three data files in `data/` directory
- comments_to_score.csv : Provided in competition
- validation_data.csv: Provided in competition
- toxic.csv - Created this based on data of previous toxicity competitions. This was created in a notebook on Kaggle

# Deep Learning Approaches

## 1. Roberta Base - Regression approach
Available under `src/score_approach` - this uses Roberta Base as backbone and predict toxicity score.

## 2. Roberta Base - Comparison approach
Available under `src/compare_approach` - this uses Roberta Base as backbone and predicts score for a pair (two texts) and uses MarginRankingLoss to train the model to correctly identify which of the two texts is more toxic than the other.
- Idea and most aspects of the approach taken from NB published by Debarshi Chanda on Kaggle for this competition.


# Downloads
## Embeddings - Did not use this approach in final solution
All embedding files should be placed in the "embeddings/" directory.
Embeddings of choice are available at: https://github.com/RaRe-Technologies/gensim-data

We have tested with `glove-wiki-gigaword-50` embedding available at: https://nlp.stanford.edu/projects/glove/

### Alternative - Use gensim library
You may use the gensim library to load the embeddings, as mentioned in the README file [here](https://github.com/RaRe-Technologies/gensim-data).

While you will get the updated API in the link mentioned above, at the time of writing this README, following is the way to load embeddings using gensim

#### Glove Embedding
```
import gensim.downloader as api

info = api.info()  # show info about available models/datasets
model = api.load("glove-twitter-25")  # download the model and return as object ready for use
model.most_similar("cat")

"""
output:

[(u'dog', 0.9590819478034973),
 (u'monkey', 0.9203578233718872),
 (u'bear', 0.9143137335777283),
 (u'pet', 0.9108031392097473),
 (u'girl', 0.8880630135536194),
 (u'horse', 0.8872727155685425),
 (u'kitty', 0.8870542049407959),
 (u'puppy', 0.886769711971283),
 (u'hot', 0.8865255117416382),
 (u'lady', 0.8845518827438354)]

"""
```

#### Word2Vec Embedding

```
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

corpus = api.load('text8')  # download the corpus and return it opened as an iterable
model = Word2Vec(corpus)  # train a model from the corpus
model.most_similar("car")

"""
output:

[(u'driver', 0.8273754119873047),
 (u'motorcycle', 0.769528865814209),
 (u'cars', 0.7356342077255249),
 (u'truck', 0.7331641912460327),
 (u'taxi', 0.718338131904602),
 (u'vehicle', 0.7177008390426636),
 (u'racing', 0.6697118878364563),
 (u'automobile', 0.6657308340072632),
 (u'passenger', 0.6377975344657898),
 (u'glider', 0.6374964714050293)]

"""
```