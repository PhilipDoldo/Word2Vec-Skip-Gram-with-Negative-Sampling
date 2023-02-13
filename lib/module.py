import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt


def process_text(s):
    """
    returns the input string with punctuation characters removed and all text
    is made lowercase
    
    Parameters:
    ----------
    s: str 
        This is the string of text that we want to remove all punctuation 
        characters from, these characters are obtained from string.punctuation
        and are the following: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    
    
    Notes:
    -----
    This function is intended to take a string of words (typically sentences,
    which may include punctuation) and return a string of lowercase words 
    separated by spaces without any punctuation. It is worth noting that this
    implementation is very naive as there could be words which contain 
    so-called punctuation characters that are actually not punctuation, such as
    hyphenated words or contractions. Additionally, there exist punctuation 
    characters in languages other than English which are not included in 
    string.punctuation. 
    """
    return s.translate(str.maketrans('','',string.punctuation)).lower()
    

def sigmoid(x):
    return (1 + np.exp(-x))**(-1)


def get_word_counts(words):
    """
    returns a dictionary where each key is a word and its value is the number
    of times that the word appeared in words
    
    Parameters:
    ----------
    words: list
        This is a list of strings where each string is a word
    """
    word_counts = {}
    for w in words:
        if w not in word_counts.keys():
            word_counts[w] = 1
        else:
            word_counts[w] += 1
    return word_counts


def get_context_words(center_word_index, words, window_size):
    """
    returns a list of strings where each string is a "context word" for the
    word at the center_word_index in the words list, i.e. a list of words
    within a window of window_size about the center word
    
    Parameters:
    ----------
    center_word_index: int
        The index of the current "center word" in the words list
    words: list
        This is a list of strings where each string is a word
    window_size: int
        The context ranges from center_word_index - window_index to 
        center_word_index + window_index
    """
    
    context_words = []
    
    for i in range(center_word_index - window_size, center_word_index + window_size + 1):
        if i >= 0 and i < len(words):
            context_words.append(words[i])
    return context_words


def get_context_word_indices(center_word_index, words, window_size):
    """
    returns a list of indices where each index corresponds to a "context word" 
    for the word at the center_word_index in the words list, i.e. a list of 
    words within a window of window_size about the center word
    
    Parameters:
    ----------
    center_word_index: int
        The index of the current "center word" in the words list
    words: list
        This is a list of strings where each string is a word
    window_size: int
        The context ranges from center_word_index - window_index to 
        center_word_index + window_index
        
        
    Notes:
    -----
    Since I plan on storing all of the context word information in a shuffled
    order at the start of every epoch of training (shuffled for SGD), I might
    as well store the indices rather than the words themselves because indices
    will take up less memory than strings, so this saves on storage. 
    
    EDIT: If I store the strings I no longer need the list of words. Storing the
    strings is simpler, so I will probably do that instead.
    """
    
    context_word_indices = []
    
    for i in range(center_word_index - window_size, center_word_index + window_size + 1):
        if i >= 0 and i < len(words):
            context_word_indices.append(i)
    return context_words


def get_unigram_probabilities(center_word_index, words, window_size, scale=0.75):
    """
    for a given center word index, returns a dictionary where each key is a 
    word and its value is the scaled unigram probability of that word
    
    Parameters:
    ----------
    center_word_index: int
        The index of the current "center word" in the words list
    words: list
        This is a list of strings where each string is a word
    window_size: int
        The context ranges from center_word_index - window_index to 
        center_word_index + window_index
    scale: float
        This is a parameter for scaling the word counts, 0.75 is supposedly a 
        common choice in practice and is made the default value for this reason
        
    
    Notes:
    -----
    Any given center word has a surrounding context of words (determined by the
    window size). When implementing negative sampling in the skip gram model, 
    we randomly choose some words that are not in the context of the current
    center word (and are not the center word itself). The random choices are
    based on scaled unigram probabilities of the remaining words. The scaled
    unigram probability of a word is number of occurences of that word in the
    list of words (raised to the power of the scale parameter) divided by the 
    sum of counts (each raised to the scale parameter) of each of the words in
    the words list (excluding the center word and the words in its context).
    Using a scale less than one makes rare words more probable to be chosen for
    a negative sample than without scaling. 
    """
    center_word = words[center_word_index]
    context_words = get_context_words(
        center_word_index=center_word_index,
        words=words,
        window_size=window_size
        )
    
    ignored_words = set(context_words + [center_word])
    
    word_counts = get_word_counts(words)
    
    scaled_word_counts = dict((w, word_counts[w]**scale) for w in word_counts.keys() if w not in ignored_words)
    total = sum(scaled_word_counts.values())
    
    return dict((w, scaled_word_counts[w]/total) for w in scaled_word_counts.keys())
    
    
def get_negative_sample(unigram_probabilities):
    """
    returns a word randomly according to its probability as stored in 
    unigram_probabilities
    
    Parameters:
    ----------
    unigram_probabilities: dict
        This is a dictionary where each key is a word and its value is its
        scaled unigram probability
    """
    random_value = np.random.uniform(0, 1)
    total = 0
    for w, p in unigram_probabilities.items():
        total += p
        if random_value <= total:
            return w
    assert False, "Probabilities don't sum to 1?"
        
    
def get_negative_samples(unigram_probabilities, number_of_samples):
    """
    returns a list containing number_of_samples words
    
    Parameters:
    ----------
    unigram_probabilities: dict
        This is a dictionary where each key is a word and its value is its
        scaled unigram probability
    number_of_samples: int
        The number of words to sample
    
    
    Notes:
    -----
    The main idea is that we randomly sample some words that are not within the
    context of the current center word and use these as negative samples when
    training a binary classifier for predicting whether or not a given word is
    within the context of a given center word.
    """
    negative_samples = set()
    while len(negative_samples) < number_of_samples:
        w = get_negative_sample(unigram_probabilities)
        negative_samples.add(w)
    
    return list(negative_samples)


def initialize_embeddings(words, embedding_dimension, mean=0, std=1):
    """
    returns two pandas dataframes containing randomly initialized values 
    corresponding to all of the center embeddings and context embeddings
    
    Parameters:
    ----------
    words: list
        This is a list of strings where each string is a word
    embedding_dimension: int
        The dimension of the word embeddings
    mean: float
        The mean of the normal distribution used to initialize the embeddings
    std: float
        The standard deviation of the normal distribution used to initialize
        the embeddings
    """
    center_embeddings = np.random.normal(loc=mean, scale=std, size=(len(words), embedding_dimension))
    center_embeddings = pd.DataFrame(data=center_embeddings, index=words)

    context_embeddings = np.random.normal(loc=mean, scale=std, size=(len(words), embedding_dimension))
    context_embeddings = pd.DataFrame(data=context_embeddings, index=words)
    
    return center_embeddings, context_embeddings


def initialize_training_data(words, window_size, k=2, scale=0.75):
    """
    returns a list of dictionaries, each of which contains a center word, one
    of its context words, and k negative sample words 
    
    Parameters:
    ----------
    words: list
        This is a list of strings where each string is a word
    window_size: int
        The context ranges from center_word_index - window_index to 
        center_word_index + window_index
    k: int
        The number of negative sample words per context word
    scale: float
        This is a parameter for scaling the word counts, 0.75 is supposedly a 
        common choice in practice and is made the default value for this reason
        
        
    Notes:
    -----
    The main idea is to shuffle this list of dictionaries at the start of every
    epoch of training in order to randomize the order in which gradients are
    computed. Each dictionary in the list is a training example which contains
    a center word, one of its context words, and k negative word samples (i.e.
    words that are not within the context window of the center word).
    """
    training_data = []
    for center_word_index, center_word in enumerate(words):
        
        context_words = get_context_words(center_word_index, words, window_size)
        unigram_probabilities = get_unigram_probabilities(center_word_index, words, window_size, scale=scale)
        
        negative_samples = get_negative_samples(unigram_probabilities, number_of_samples=k*len(context_words))
        
        for i, context_word in enumerate(context_words):
        
            training_point = {'center' : center_word, 'context' : context_word, 'negative' : negative_samples[k*i:k*i+k]}
            training_data.append(training_point)
        
    return training_data


def unpack_training_point(training_point):
    """
    given a training point, returns the center word (string), context word
    (string), and list of negative words (list of strings)
    
    Parameters:
    ----------
    training_point: dict
        Contains a key "center" corresponding to the center word, a key
        "context" corresponding to a single context word, and a key "negative"
        corresponding to a list of words which are negative context samples
        (i.e. samples of words not within the context of the center word)
    
    
    Notes:
    -----
    The only reason I'm writing this function is because I want to reduce
    repeated code and the keys of the dictionary are specific strings, so it
    is more manageable to only use them in this function if I ever decided to
    change them for some reason.
    """
    
    center_word = training_point['center']
    context_word = training_point['context']
    negative_words = training_point['negative']

    return center_word, context_word, negative_words


def get_gradients(training_point, center_embeddings, context_embeddings):
    """
    for a given training point, returns the center word gradient (stored in a 
    dictionary where the key is the center word and the value is the gradient
    which is represented as a pandas DataFrame) and all of the context word 
    gradients (which are all stored in their own context dictionary, analogous
    to the center word dictionary)
    
    Parameters:
    ----------
    training_point: dict
        Contains a key "center" corresponding to the center word, a key
        "context" corresponding to a single context word, and a key "negative"
        corresponding to a list of words which are negative context samples
        (i.e. samples of words not within the context of the center word)
    center_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as center words.
    context_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as context words.
    
    
    Notes:
    -----
    The reason I choose to store the gradients in dictionaries is so that I can
    easily keep track of which words correspond to which gradients as well as
    keep track of whether the embedding corresponds to a center word or a
    context word. 
    """
    
    center_word, context_word, negative_words = unpack_training_point(training_point)
    
    center_gradient = {}
    context_gradients = {}
    
    # Center Word Gradient
    center_grad = sigmoid(np.dot( context_embeddings.loc[context_word], center_embeddings.loc[center_word] ) - 1)*context_embeddings.loc[context_word]
    for negative_word in negative_words:
        center_grad += sigmoid(np.dot( context_embeddings.loc[negative_word], center_embeddings.loc[center_word] ))*context_embeddings.loc[negative_word]
        
    center_gradient[center_word] = center_grad
    
    # Positive Context Gradient
    context_gradients[context_word] = (sigmoid(np.dot( context_embeddings.loc[context_word], center_embeddings.loc[center_word] )) - 1)*center_embeddings.loc[center_word]
    
    # Negative Context Gradients
    for negative_word in negative_words:
        context_gradients[negative_word] = sigmoid(np.dot( context_embeddings.loc[negative_word], center_embeddings.loc[center_word] ))*center_embeddings.loc[center_word]
    
    return center_gradient, context_gradients


def get_batch_gradients(training_points, center_embeddings, context_embeddings):
    """
    for a batch of training points, returns two dictionaries, each of which
    contains the sum of all of the appropriate gradients corresponding to each
    word in the batch of training points
    
    Parameters:
    ----------
    training_points: list
        This is a list of dictionaries where each dictionary contains a key 
        "center" corresponding to the center word, a key "context" 
        corresponding to a single context word, and a key "negative"
        corresponding to a list of words which are negative context samples
        (i.e. samples of words not within the context of the center word)
    center_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as center words.
    context_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as context words.
    
    
    Notes:
    -----
    The main idea is that if we are training with minibatch gradient descent,
    then this function will compute all of the gradients for that minibatch
    and add them together. It is assumed that training_points is a list of
    training points corresponding to an entire minibatch.
    """
    
    center_updates = {}
    context_updates = {}
    
    for training_point in training_points:
        center_gradient, context_gradients = get_gradients(training_point, center_embeddings, context_embeddings)

        for center_word in center_gradient.keys():
            if center_word not in center_updates.keys():
                center_updates[center_word] = center_gradient[center_word]
            else:
                center_updates[center_word] += center_gradient[center_word]
        
        for context_word in context_gradients.keys():
            if context_word not in context_updates.keys():
                context_updates[context_word] = context_gradients[context_word]
            else:
                context_updates[context_word] += context_gradients[context_word]
                
    return center_updates, context_updates


def get_training_point_loss(training_point, center_embeddings, context_embeddings):
    """
    for a given training point, returns the negative log probability associated
    with it as a float
    
    Parameters:
    ----------
    training_point: dict
        Contains a key "center" corresponding to the center word, a key
        "context" corresponding to a single context word, and a key "negative"
        corresponding to a list of words which are negative context samples
        (i.e. samples of words not within the context of the center word)
    center_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as center words.
    context_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as context words.
    
    
    Notes:
    -----
    The negative log is applied to the probability that the context word is in
    fact a true context word multiplied by the probabilities that each of the
    negative samples are in fact not true context words (a product is used due
    to naive independence assumptions) where the probability of a context word,
    whose context embedding is denoted by c, being a true context word of a 
    center word, whose center embedding is denoted by w, is modeled as 
    sigmoid(w^T c) (and thus the "context" words corresponding to negative 
    samples will contribute in the form 1 - sigmoid(w^T c) which can be written
    as sigmoid(-w^T c)).
    """
    
    center_word, context_word, negative_words = unpack_training_point(training_point)
    
    # initialize loss as -log prob for the true context word
    loss = -np.log( sigmoid( np.dot(center_embeddings.loc[center_word], context_embeddings.loc[context_word]) ) )
    
    # now add on to the loss for every negatuve word
    for negative_word in negative_words:
        loss += -np.log( sigmoid( - np.dot(center_embeddings.loc[center_word], context_embeddings.loc[negative_word]) ) )
    
    return loss


def get_loss(training_data, center_embeddings, context_embeddings):
    """
    for a list of training points, returns the sum of all of the negative log
    probabilities associated with each individual training point divided by
    the total number of training points (i.e. returns the average loss across
    all of the training points) as a float
    
    Parameters:
    ----------
    training_data: list
        This is a list of dictionaries where each dictionary contains a key 
        "center" corresponding to the center word, a key "context" 
        corresponding to a single context word, and a key "negative"
        corresponding to a list of words which are negative context samples
        (i.e. samples of words not within the context of the center word)
    center_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as center words.
    context_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as context words.
        
    
    Notes:
    -----
    The main idea is that this function is returning the overall loss when
    taking every single training point into account (because training_data is
    a list containing every training point and we compute the loss for each of
    these individual training points and then add them up).
    """
    loss = 0
    for training_point in training_data:
        loss += get_training_point_loss(training_point, center_embeddings, context_embeddings)
    return loss/len(training_data)


def update_embeddings(training_data, center_embeddings, context_embeddings, batch_size=1, learning_rate=.01, verbose=False):
    """
    returns the center and context embeddings after a single epoch of training
    
    Parameters:
    ----------
    training_data: list
        This is a list of dictionaries where each dictionary contains a key 
        "center" corresponding to the center word, a key "context" 
        corresponding to a single context word, and a key "negative"
        corresponding to a list of words which are negative context samples
        (i.e. samples of words not within the context of the center word)
    center_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as center words.
    context_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as context words.
    batch_size: int
        The number of training points to include in a single minibatch when
        computing the gradients in gradient descent. If set to 1, this reduces
        to pure SGD.
    learning_rate: float
        Scaling hyperparameter that determines how far in the direction of the
        computed gradients we should update our embedding parameters.
    verbose: bool
        If true, will print how many training points have been processed by
        gradient descent so far out of the total number after each minibatch
        is processed.
        
    
    Notes:
    -----
    The main idea is that training_data contains every single training point
    and this function trains the embedding parameters for an entire epoch of 
    minibatch gradient descent.
    """
    
    for i in range( int(np.ceil(len(training_data)/batch_size)) ):
        training_points = training_data[batch_size*i:batch_size*i + batch_size]
        
        center_updates, context_updates = get_batch_gradients(training_points, center_embeddings, context_embeddings)
        
        for center_word in center_updates.keys():
            
            center_embeddings.loc[center_word] -= learning_rate * center_updates[center_word]
        
        for context_word in context_updates.keys():
            context_embeddings.loc[context_word] -= learning_rate * context_updates[context_word]
        
        
        if verbose:
            loss = get_loss(training_data=training_data, center_embeddings=center_embeddings, context_embeddings=context_embeddings)
            print(f" ---- {min(batch_size*i + batch_size, len(training_data))} / {len(training_data)} completed -- loss: {loss:.4f}")
        
        
    return center_embeddings, context_embeddings


def train_embeddings(training_data, center_embeddings, context_embeddings, batch_size=1, learning_rate=.01, num_epochs=5, verbose=False):
    """
    returns the fully trained center and context word embeddings, trained using
    minibatch gradient descent for a total of num_epochs epochs. If verbose is
    True, then also returns a list of losses (list of floats), that is the loss
    at the end of every epoch
    
    Parameters:
    ----------
    training_data: list
        This is a list of dictionaries where each dictionary contains a key 
        "center" corresponding to the center word, a key "context" 
        corresponding to a single context word, and a key "negative"
        corresponding to a list of words which are negative context samples
        (i.e. samples of words not within the context of the center word)
    center_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as center words.
    context_embeddings: pandas DataFrame
        Dataframe indexed by word strings, each column corresponds to a
        different dimension of the embedding for word for that row. Contains
        the embeddings for when words are treated as context words.
    batch_size: int
        The number of training points to include in a single minibatch when
        computing the gradients in gradient descent. If set to 1, this reduces
        to pure SGD.
    learning_rate: float
        Scaling hyperparameter that determines how far in the direction of the
        computed gradients we should update our embedding parameters.
    num_epochs: int
        The number of epochs to train minibatch gradient descent for.
    verbose: bool
        If true, will print how many training points have been processed by
        gradient descent so far out of the total number after each minibatch
        is processed as well as print when a new epoch begins.
        
    
    Notes:
    -----
    The main idea is that at the start of every epoch we randomly shuffle the 
    list of training examples before deterministically iterating through it and
    computing the minibatch gradients in order to simulate randomly choosing
    the minibatches. We train minibatch SGD for however many epochs we want and
    we return both the center and context word embeddings.
    """
    losses = []
    for epoch_index in range(num_epochs):
        
        if verbose:
            if epoch_index == 0:
                loss = get_loss(training_data=training_data, center_embeddings=center_embeddings, context_embeddings=context_embeddings)
                print(f"Initial Loss: {loss:.4f}")
            print(f"CURRENT EPOCH: {epoch_index+1} / {num_epochs}")
        
        np.random.shuffle(training_data) 
        
        center_embeddings, context_embeddings = update_embeddings(
            training_data=training_data,
            center_embeddings=center_embeddings,
            context_embeddings=context_embeddings,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose
            )
        
        if verbose:
            loss = get_loss(training_data=training_data, center_embeddings=center_embeddings, context_embeddings=context_embeddings)
            losses.append(loss)
            
    if verbose == True:
        return center_embeddings, context_embeddings, losses
    else:
        return center_embeddings, context_embeddings


def find_closest_words(word, embeddings):
    
    d = {}
    for w in embeddings.index:
        d[w] = np.linalg.norm(embeddings.loc[word] - embeddings.loc[w], 2)
    
    df = pd.DataFrame(data=d.values(), index=d.keys())
    df.sort_values(by=0, inplace=True)
    return df
    

def plot_2d_grid(embeddings):
    
    x = embeddings[0]
    y = embeddings[1]
    
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    
    for w in embeddings.index:
        ax.annotate(w, (x[w], y[w]))
    
    