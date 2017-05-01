import unittest

from shutil import rmtree
from os import getcwd, mkdir
from os.path import join, isfile, isdir

from random import uniform, randint

import gensim
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
LabeledSentence = gensim.models.doc2vec.LabeledSentence

from data_gathering import Condense_Reviews
from data_prep import Prepare_Reviews
from data_modeling import Vectorize_Reviews


def setup_testarea(data_dir, output_dir):
    print "Setting up testing suite..."

    # Set up temporary directory for testing purposes
    if isdir(output_dir):
        rmtree(output_dir)
        mkdir(output_dir)
    else:
        mkdir(output_dir)

    # Initialize a Condense_Reviews object
    condenser = Condense_Reviews(data_dir, output_dir)

    # Generate data in temporary directory
    print "Downloading raw data..."
    condenser.get_data()

    # Condense reviews in test_dir
    print "\nCondensing raw data..."
    condenser.condense()


class RunTests(unittest.TestCase):
    """
    Docstring for testing suite for Condense_Reviews class methods
    """
    output_dir = "path/to/temp/test/area"
    data_dir = "path/to/data"


    def test_get_data(self):
        # Test that data directory exists and contains the correct folders
        self.assertTrue(isdir(self.data_dir), True)
        self.assertTrue(isdir(join(self.data_dir, "test/pos")))
        self.assertTrue(isdir(join(self.data_dir, "test/neg")))
        self.assertTrue(isdir(join(self.data_dir, "train/pos")))
        self.assertTrue(isdir(join(self.data_dir, "train/neg")))
        self.assertTrue(isdir(join(self.data_dir, "train/unsup")))

        # Check a few files within some folders to make sure they're not empty
        self.assertTrue(isfile(join(self.data_dir, "test/pos/5906_10.txt")))
        self.assertTrue(isfile(join(self.data_dir, "test/neg/5906_2.txt")))
        self.assertTrue(isfile(join(self.data_dir, "train/pos/4883_10.txt")))
        self.assertTrue(isfile(join(self.data_dir, "train/neg/2833_1.txt")))
        self.assertTrue(isfile(join(self.data_dir, "train/unsup/46814_0.txt")))

        # Check that tar.gz file has been removed
        self.assertFalse(isfile(join(self.output_dir, "aclImdb_v1.tar.gz")))


    def test_condense(self):
        # Test that condensed data files exist in test_dir
        self.assertTrue(isfile(join(self.output_dir, "pos.txt")))
        self.assertTrue(isfile(join(self.output_dir, "neg.txt")))
        self.assertTrue(isfile(join(self.output_dir, "unlab.txt")))


    def test_import_data(self):
        # Initialize Prepare_Reviews object and import data
        preparer = Prepare_Reviews(self.output_dir)
        X, Y, unlab_reviews = preparer.import_data()

        # Test that reviews were properly loaded
        self.assertIsNotNone(X)
        self.assertIsNotNone(Y)
        self.assertIsNotNone(unlab_reviews)

        # Test that there are 50,000 labeled and 50,000 unlabeled reviews
        self.assertTrue(len(X)==50000)
        self.assertTrue(len(Y)==50000)
        self.assertTrue(len(unlab_reviews)==50000)


    def test_cleaner_and_labler(self):
        # Initialize Prepare_Reviews object and import data
        preparer = Prepare_Reviews(self.output_dir)
        _, _, unlab_reviews = preparer.import_data()

        # Take a random review and clean it
        test_review = unlab_reviews[randint(0, len(unlab_reviews))]
        test_review = preparer.cleaner([test_review], full_clean=True)[0]

        print test_review

        # Define punctuation list
        punctuation = """'".!?,:;(){}[]-/"""

        # Test cleaner method
        self.assertTrue("<br />" not in test_review)
        self.assertTrue(False not in [word.islower() for word in test_review
                                      if (word not in punctuation) and (not word.isdigit())])
        self.assertFalse(any(word in stopwords for word in test_review))

        # Label test_review
        labeled_review = LabeledSentence(test_review, ["TEST_REVIEW"])

        # Test labler method
        self.assertTrue(labeled_review.words == test_review)


    def test_shuffle_lists(self):
        # Generate random lists and save the relative in a dictionary
        X = [uniform(0, 10) for x in range(100)]
        Y = [randint(0, 1) for x in range(100)]
        dict1 = {X[idx]: Y[idx] for idx in range(100)}

        # Initialize Vectorize_Reviews object
        vectorizer = Vectorize_Reviews(X, Y, X, Y, X)

        # Shuffle lists and create new dictionary
        X, Y = vectorizer.shuffle_lists(X, Y)
        dict2 = {X[idx]: Y[idx] for idx in range(100)}

        # Test shuffle_lists
        self.assertTrue(dict1 == dict2)


if __name__ == "__main__":
    # Define paths
    output_dir = join(getcwd(), "tmp")
    data_dir = join(output_dir, "aclImdb")

    # Setup test area
    setup_testarea(data_dir, output_dir)

    # Set paths
    RunTests.output_dir = output_dir
    RunTests.data_dir = data_dir

    # Run tests
    unittest.main(exit=False)

    # Tear down test area
    rmtree(output_dir)
