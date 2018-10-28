# RNN implementation for solving the Kaggle 'Jigsaw toxic comment classification challenge' competition

This repository contains my quick and dirty solution of the Kaggle Jigsaw competition. In this competition a set of labeled text messages was provided. The messages were categorized in 6 different categories which represented the degree of "toxicity" of the message. The goal is to assign the correct labels to each of the message only using the text as input for the algorithm.

## Motivation
I want to prove the power of the Recurrent Neural Networks (LSTM) in text-related tasks. Hypotheses:
- Text labeling can be performed by a many-to-one RNN
- We can achieve a high performance at a lower cost by inputting the data at a character level
- Zero padding can be used for fitting the sentences into a big matrix and it will not disturb the performance 

## Getting started
For running the algorithm just follow the next steps.
1) Clone the repository
2) Create a `settings.json` config file from the `settings_template.json` and fill in the required arguments
3) Create a folder in the root of the repository and name it `input`
4) Download the data from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and put it into the `input`folder
5) Run TensorBoard specifying the same logs path you specified in the `settings.json` file
6) From the root of the repository, open an `ipython` REPL and run `%run src/main.py`

## Disclaimer
This effort was not made with the aim of participating in the competition, but as a proof of concept to understand the frontiers of Neural Networks in this configuration. Don't expect breaking results compared to the results achieved by the winners of the competition

## License
This repository is licensed under MIT license. Copyright (c) 2018 Iván Vallés Pérez
