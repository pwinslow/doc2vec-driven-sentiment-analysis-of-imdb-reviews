# Project utility modules
from data_gathering import run_condenser
from data_prep import prepare_data
from data_modeling import Vectorize_Reviews, Classify_Reviews


def main():
    # Run condenser
    run_condenser()

    # Perform preprocessing
    X_train, Y_train, X_test, Y_test, unlab_reviews = prepare_data()

    # Initialize Vectorize_Reviews object and get doc2vec vector representations of reviews
    vectorizer = Vectorize_Reviews(X_train,
                                   Y_train,
                                   X_test,
                                   Y_test,
                                   unlab_reviews)
    train_vecs, Y_train, test_vecs, Y_test = vectorizer.train_doc2vec()

    # Initialize Classify_Reviews object and train logistic regression classifier on doc2vec features
    classifier = Classify_Reviews(train_vecs,
                                  Y_train,
                                  test_vecs,
                                  Y_test)
    classifier.train_model()

    # Validate classifier
    classifier.validate_model()


if __name__ == "__main__":
    main()
