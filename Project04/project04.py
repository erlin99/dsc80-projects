
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    
    time.sleep(5) # make sure to not overload server
    # request the book text 
    resp = requests.get(url)
    book = resp.text

    book = book.replace('\r\n', '\n')

    # get starting and end index for content between START and END comments
    _, start = re.search('[\*]{3} START.* [\*]{3}', book).span()
    end, _ = re.search('[\*]{3} END', book).span()

    # get content between indexes 
    book = book[start:end]

    return book
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    def split_text(text):
        """
        Function to split a text into words and punctuation
        """
        result = re.split('\\b', text) 
        result = [x.strip() for x in result if x.strip() != '']
        return result

    # split the book into paragraphs
    paragraphs = re.split('\n[\n]+', book_string)
    # remove leading and trailing spaces and new line characters in each paragraph
    paragraphs = [x.strip() for x in paragraphs if x.strip().replace('\n', '') != '']

    # split the text of each paragraph and add start and end ascii codes 
    result = [['\x02'] + split_text(text) + ['\x03'] for text in paragraphs]

    # return the flattened list 
    return [item for sublist in result for item in sublist]
    

# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------

class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)


    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        unique = np.unique(tokens) # get all unique values 
 
        probability = 1/len(unique) # calculate probability 

        result = {key : probability for key in unique} # create dictionary  

        return pd.Series(result) # return series generated form dictionary  
    

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        prob = 1 # set prob to 1 to be able to multiply 
        for token in words:
            # check that token exists  
            if token in self.mdl.index:
                prob *= self.mdl[token]
            # return 0 otherwise 
            else:
                return 0

        return prob
        

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        sample = self.mdl.sample(1000, replace=True).index

        return ' '.join(sample)

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    

    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        frequency = {} # dictionary to count frequency of each word 

        def get_unigram_prob(val):
            """
            Function that updates the dictionary 
            """
            if val in frequency:
                frequency[val] += 1
            else:
                frequency[val] = 1

        # run function on each token in list  
        [get_unigram_prob(x) for x in tokens]

        # return series converted to the probability 
        return pd.Series(frequency) / len(tokens)
    

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """

        prob = 1 # set prob to 1 to be able to multiply 
        for token in words:
            # check that token exists  
            if token in self.mdl.index:
                prob *= self.mdl[token]
            # return 0 otherwise 
            else:
                return 0

        return prob
        

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        sample = self.mdl.sample(M, weights = self.mdl.values, replace=True).index
        
        return ' '.join(sample)
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl


    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """        

        return [tuple(tokens[i:i + self.N]) for i in range(len(tokens) - (self.N) + 1)]
        

    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe indexed on distinct tokens, with three
        columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """

        def update_dictionary(ngram, dictionary):
            """
            Function to update the givne dictionary with the given ngram 
            """
            if ngram in dictionary:
                dictionary[ngram] += 1
            else:
                dictionary[ngram] = 1

        # ngram counts C(w_1, ..., w_n)
        ngram_counts = {}
        [update_dictionary(ngram, ngram_counts) for ngram in self.ngrams] 

        # n-1 gram counts C(w_1, ..., w_(n-1))
        n1gram_counts = {}
        n1grams = [ngram[:-1] for ngram in self.ngrams] # get the n-1 grams 
        [update_dictionary(n1gram, n1gram_counts) for n1gram in n1grams] 

        # Create the conditional probabilities
        def get_prob(ngram):
            """
            Function to calculate the probability of a given ngram 
            """
            prob = (ngram_counts[ngram] / n1gram_counts[ngram[:-1]])
            return prob
        # call funcition in all ngrams 
        prob = [get_prob(ngram) for ngram in self.ngrams]

        # Put it all together in a df
        result = pd.DataFrame({
            'ngram':self.ngrams,
            'n1gram': n1grams,
            'prob':prob
        })

        # drop duplicate rows 
        return result.drop_duplicates().reset_index(drop=True)
    

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        # convert words given into tuples 
        tuples = [tuple(words[i:i + self.N]) for i in range(len(words) - (self.N) + 1)]

        def first_tuple(x, t):
            """
            Function to get the prob of the first tuple since we have to recursively
            go back to previous mdls 
            """
            # mdl for tuple size 1 is a series 
            if len(t) == 1:
                # check that ngram exists 
                if t[0] in x.mdl:
                    return x.mdl.loc[t[0]]
            # mdl for tuples sizes > 1 are dataframes 
            else:
                mdl = x.mdl # get the mdl 
                # check that ngram exists 
                if not mdl[mdl['ngram'] == t].empty:
                    prob = mdl[mdl['ngram'] == t]['prob'].values[0]
                    return prob * first_tuple(x.prev_mdl, t[:-1])
            return 0

        prob = first_tuple(self, tuples[0]) # initialize probability 

        # check that is not already 0 if it is we just return 0 
        if prob != 0:
            # loop through the rest of the tuples 
            for ngram in tuples[1:]:
                if len(ngram) == 1:
                    if ngram in self.mdl:
                        prob *= self.mdl.loc[ngram[0]]
                    else:
                        return 0
                else:
                    mdl = self.mdl
                    if not mdl[mdl['ngram'] == ngram].empty:
                        prob *= mdl[mdl['ngram'] == ngram]['prob'].values[0]
                    else:
                        return 0

        return prob


    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """

        # Use a helper function to generate sample tokens of length `length`
        # def first_N_words():

        
        # def recursive
        
        # Transform the tokens to strings
        ...

        return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
