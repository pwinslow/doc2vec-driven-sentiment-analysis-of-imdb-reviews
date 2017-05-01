# Analysis imports
import numpy as np

# NLP imports
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

# Doc2vec imports
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence

# Scikit-learn imports
from sklearn.model_selection import train_test_split

# Miscellaneous imports
from os import getcwd
from os.path import join
from unidecode import unidecode


class Prepare_Reviews(object):
    """
    The Prepare_Reviews class imports the condensed data and cleans and labels it in preparation for training with Doc2vec.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir


    def import_data(self):
        # Import all data
        with open(join(self.data_dir, "pos.txt"), "r+") as input_file:
            pos_reviews = input_file.readlines()
        with open(join(self.data_dir, "neg.txt"), "r+") as input_file:
            neg_reviews = input_file.readlines()
        with open(join(self.data_dir, "unlab.txt"), "r+") as input_file:
            unlab_reviews = input_file.readlines()

        # Organize positive/negative sentiment data into arrays
        X = np.concatenate((pos_reviews, neg_reviews))
        Y = np.concatenate( (np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))) )

        return X, Y, unlab_reviews


    @staticmethod
    def split_data(X, Y, test_size):
        # Perform train-test split on data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

        return X_train, X_test, Y_train, Y_test


    @staticmethod
    def remove_non_ascii(text):
        return unidecode(unicode(text, encoding = "utf-8"))


    def cleaner(self, corpus, full_clean=False):
        # Define punctuation
        punctuation = """'".!?,:;(){}[]-/"""

        # Remove non-ascii characters
        corpus = [self.remove_non_ascii(review) for review in corpus]

        # Remove capitalizations and all linebreak and EOF tags
        corpus = [review.lower().replace("<br />", " ").strip() for review in corpus]

        # Treat punctuation as individual words
        for p in punctuation:
            corpus = [review.replace(p, " %s " % p) for review in corpus]

        # Tokenize into individual words
        corpus = [review.split() for review in corpus]

        # Optional removal of stopwords and porter stemming
        if full_clean:
            new_corpus = []
            for review in corpus:
                new_review = [p_stemmer.stem(word) for word in review if word not in stopwords]
                new_corpus.append(new_review)
            return new_corpus

        return corpus


    def clean_reviews(self, X_train, X_test, unlab_reviews):
        # Clean all corpus data
        X_train = self.cleaner(X_train, full_clean=True)
        X_test = self.cleaner(X_test, full_clean=True)
        unlab_reviews = self.cleaner(unlab_reviews, full_clean=True)

        return X_train, X_test, unlab_reviews


    @staticmethod
    def labler(corpus, label_prefix):
        # Label each review, as required by Doc2vec
        labeled_reviews = []
        for idx, review in enumerate(corpus):
            label = "%s_%s" % (label_prefix, idx)
            labeled_reviews.append(LabeledSentence(review, [label]))

        return labeled_reviews


    def label_reviews(self, X_train, X_test, unlab_reviews):
        # Label all corpus data
        X_train = self.labler(X_train, "TRAIN")
        X_test = self.labler(X_test, "TEST")
        unlab_reviews = self.labler(unlab_reviews, "UNLABELED")

        return X_train, X_test, unlab_reviews


def prepare_data():
    # Initialize a Prepare_Reviews object
    data_dir = getcwd()
    preparer = Prepare_Reviews(data_dir)

    # Import data
    print "Importing the data..."
    X, Y, unlab_reviews = preparer.import_data()

    # Split the labeled data
    test_size = 0.3
    X_train, X_test, Y_train, Y_test = preparer.split_data(X, Y, test_size)

    # Clean reviews
    print "Cleaning the data..."
    X_train, X_test, unlab_reviews = preparer.clean_reviews(X_train, X_test, unlab_reviews)

    # Label reviews
    print "Labeling the data..."
    X_train, X_test, unlab_reviews = preparer.label_reviews(X_train, X_test, unlab_reviews)

    return X_train, Y_train, X_test, Y_test, unlab_reviews


if __name__ == "__main__":
    prepare_data()
