from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def naive_bayes(X_train, y_train, tfidfvectorizer__ngram_range = ((1,1), (2,2)), multinomialnb__alpha = (0.1, 1)):
    ''' This function implements a fitted multinomial naive bayes model.
    The parameters of the pipeline are found through a grid-search based on inputs provided by the user:
    - tfidfvectorizer__ngram_range is a tuple (default : ((1,1), (2,2)))
    - multinomialnb__alpha is a float or a tuple of floats (default: (0.1, 1))
    '''
    # step 1 : building the pipepline
    pipeline = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB())

    # step 1 : defining the parameters to be gridsearched
    parameters = {'tfidfvectorizer__ngram_range': tfidfvectorizer__ngram_range,
                  'multinomialnb__alpha': multinomialnb__alpha}

    # step 2: defining the gridsearch
    grid_search = GridSearchCV(
        pipeline,
        parameters,
        scoring = "accuracy",
        cv = 5,
        n_jobs=-1,
        verbose=1
    )

    # step 3: fitting the gridsearch
    grid_search.fit(X_train,y_train)

    # step 4: getting the best estimators
    grid_search.best_estimator_

    return grid_search.best_estimator_.fit(X_train,y_train)
