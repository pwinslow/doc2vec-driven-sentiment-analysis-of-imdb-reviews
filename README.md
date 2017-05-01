# Sentiment Analysis of IMDB Reviews using Doc2Vec

## Overview

This repo contains a collection of modules for training a logistic regression classifier to predict sentiment of IMDB reviews. The main purpose of this project was to familiarize myself with doc2vec, which is used to derive numerical feature vectors from the text of the reviews that codify semantic information.

## Prerequisites

The easiest way to work with the modules is to install the [Anaconda python distribution](https://anaconda.org/). To install all necessary dependencies, fork the repo to your local machine and create and source a conda environment by running
```
conda env create -f environment.yml
source activate doc2vec-imdb-sentiment-analysis
```

## Classifying Sentiment

To train the logistic regression classifier, run
```
python runner.py
```
This will create a directory called "models" which will contain a pickled version of the trained classifier as well as the trained doc2vec model.

## Run Testing Suite

To run the testing suite, run
```
python test_suite.py
```
