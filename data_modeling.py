# Analysis imports
import numpy as np

# Gensim imports
import gensim

# Scikit-learn imports
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve

# Visualization analysis imports
import matplotlib.pyplot as plt

# Miscellaneous imports
import os
from os.path import isfile, isdir
from random import shuffle
import cPickle


class Vectorize_Reviews(object):
    """
    The Reviews2Vec class trains a doc2vec model on the corpus of reviews and returns a feature matrix of doc2vec
    vectors for training predictive models.
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, unlab_reviews,
                 vec_size=400, min_count=1, window=10, sample=1e-3,
                 negative=5, workers=4, dm=1, epochs=5):
        # Define data structures as class variables
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.unlab_reviews = unlab_reviews

        # Define internal Doc2vec parameters
        self.vec_size = vec_size
        self.min_count = min_count
        self.window = window
        self.sample = sample
        self.negative = negative
        self.workers = workers
        self.dm = dm

        # Set number of training epochs
        self.epochs = epochs


    @staticmethod
    def shuffle_lists(X, Y):
        # Shuffle X and Y while maintaining index coordination
        X_shuf, Y_shuf = [], []
        idx_list = range(len(X))
        shuffle(idx_list)
        for idx in idx_list:
            X_shuf.append(X[idx])
            Y_shuf.append(Y[idx])
        return X_shuf, Y_shuf


    def review_to_vec(self, model, review):
        # Convert review to vector using trained model
        vec = np.array(model.infer_vector(review.words)).reshape((1, self.vec_size))

        return vec


    def corpus_to_vec(self, model, corpus):
        # Get list of doc2vec vectors from corpus
        vecs = [self.review_to_vec(model, review) for review in corpus]

        # Convert list to numpy array
        vec_arr = np.concatenate(vecs)

        return vec_arr


    def build_vocabulary(self):
        # Initialize doc2vec distributed memory object
        model = gensim.models.Doc2Vec(min_count=self.min_count,
                                      window=self.window,
                                      size=self.vec_size,
                                      sample=self.sample,
                                      negative=self.negative,
                                      workers=self.workers,
                                      dm=self.dm)

        # Build vocabulary over all reviews
        print "Building vocabulary...\n"
        all_reviews = self.X_train + self.X_test + self.unlab_reviews
        model.build_vocab(all_reviews)

        return model


    def train_on_reviews(self, model, X, Y, unlab_set=[]):
        # Run through the dataset multiple times, shuffling the data each time to improve accuracy
        train_reviews = X + unlab_set
        for epoch in range(self.epochs):
            print "Training epoch: {0}/{1}".format(epoch+1, self.epochs)
            model.train(train_reviews)

            # Shuffle data
            print "Shuffling data..."
            X, Y = self.shuffle_lists(X, Y)
            shuffle(unlab_set)
            train_reviews = X + unlab_set

        print "Calculating doc2vec vectors..."
        train_vecs = self.corpus_to_vec(model, X)
        print "Done training...\n"

        return train_vecs, model, X, Y, unlab_set


    def train_doc2vec(self):
        if isfile("models/doc2vec_model.doc2vec"):
            print "Doc2vec model is already trained. Loading existing model..."
            model = gensim.models.Doc2Vec.load("models/doc2vec_model.doc2vec")

            # Calculate doc2vec vectors
            print "Calculating doc2vec vectors..."
            train_vecs = self.corpus_to_vec(model, self.X_train)
            test_vecs = self.corpus_to_vec(model, self.X_test)

            return (train_vecs,
                    self.Y_train,
                    test_vecs,
                    self.Y_test)

        else:
            # Initialize doc2vec model and build vocabulary
            model = self.build_vocabulary()

            # Train doc2vec model and get semantic vectors for training set
            print "Training doc2vec model on training dataset..."
            (train_vecs,
             model,
             self.X_train,
             self.Y_train,
             self.unlab_reviews) = self.train_on_reviews(model=model,
                                                         X=self.X_train,
                                                         Y=self.Y_train,
                                                         unlab_set=self.unlab_reviews)

            # Extend doc2vec model training and get semantic vectors for testing set
            print "Training doc2vec model on testing dataset..."
            test_vecs, model, self.X_test, self.Y_test, _ = self.train_on_reviews(model=model,
                                                                                  X=self.X_test,
                                                                                  Y=self.Y_test)

            # Create model directory if necessary
            if not isdir("models"):
                os.mkdir("models")

            # Save model for persistence
            model.save("models/doc2vec_model.doc2vec")

            return (train_vecs,
                    self.Y_train,
                    test_vecs,
                    self.Y_test)


class Classify_Reviews(object):
    """
    The Review_Classification class trains a logistic regression classifier, employing stochastic gradient descent with L2 regularization.
    """
    def __init__(self, train_vecs, Y_train, test_vecs, Y_test):
        self.train_vecs = train_vecs
        self.Y_train = Y_train
        self.test_vecs = test_vecs
        self.Y_test = Y_test


    @staticmethod
    def persist_model(model):
        # Pickle model for persistence
        with open("models/logreg_model.pkl", 'wb') as output_model:
            cPickle.dump(model, output_model)


    @staticmethod
    def load_model():
        # Load pickled random forest model
        with open("models/logreg_model.pkl", 'rb') as input_model:
            model = cPickle.load(input_model)
        return model


    def train_model(self):
        if not isfile("models/logreg_model.pkl"):
            print "Training logistic regression classifier..."
            # Initialize logistic regression classifier employing stochastic gradient descent with L1 regularization
            lr = SGDClassifier(loss="log", penalty="l1")

            # Define a parameter grid to search over
            param_grid = {"alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}

            # Setup 5-fold stratified cross validation
            cross_validation = StratifiedKFold(n_splits=5)

            # Perform grid search over hyperparameter configurations with 5-fold cross validation for each
            clf = GridSearchCV(lr,
                               param_grid=param_grid,
                               cv=cross_validation,
                               n_jobs=10,
                               verbose=10)
            clf.fit(self.train_vecs, self.Y_train)

            # Extract best estimator and pickle model for persistence
            self.persist_model(clf.best_estimator_)
        else:
            print "Logistic regression classifier is already trained..."


    def validate_model(self):
        print "Validating classifier..."

        # Load classifier
        model = self.load_model()

        # Classify test dataset
        Y_predicted = model.predict(self.test_vecs)

        # Calculate AUC score
        roc_auc = roc_auc_score(self.Y_test, Y_predicted)

        # Print full classification report
        print classification_report(self.Y_test,
                                    Y_predicted,
                                    target_names=["negative", "positive"])
        print "Area under ROC curve: {:0.3f}".format(roc_auc)

        # Compute ROC curve and area under the curve
        probs = model.predict_proba(self.test_vecs)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.Y_test, probs)

        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.3f)'%(roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title('Receiver Operating Characteristic (ROC) curve', fontsize=12)
        plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
        plt.grid(True, linestyle = 'dotted')
        plt.savefig("doc2vec_roc.png")
        print "ROC curve created..."
