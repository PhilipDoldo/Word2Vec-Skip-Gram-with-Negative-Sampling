# Word2Vec-Skip-Gram-with-Negative-Sampling

This code naively implements Word2Vec word embeddings from scratch using a skip-gram model with negative sampling. Minibatch gradient descent was implemented to train the loss.

The **lib** folder contains **module.py** which contains all of the function definitions used in the script **train_embeddings_example.py**. This script learns two-dimensional word embeddings from a very small corpus of less than 200 words. This small corpus contains the phrase "tall buildings" in several places in order to see if the learned embeddings for "tall" and "buildings" end up close to each other in the embedding space. The plot created by this script indicates that some structure has been learned about the words and we indeed see the words "tall" and "buildings" are very close to each other. This was achieved without any particular effort being put into optimizing the training hyperparameters.  

## Notes

The data used for training the model is obtained from some written English text which is preprocessed into a list of words, in order as they appear in the written text (puncuation is naively removed and all letters are lowercased). 

TODO
