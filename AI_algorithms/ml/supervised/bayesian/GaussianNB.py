import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.stats import norm # type: ignore

class GaussianNB:
    
    def __init__(self):
        self.classes = None
        self.features = None
        self.params = None
        self.priors = None
        self.discrete_probs = None
        self.discrete = []
    
    def fit(self, X, y, discrete=None, correction = None, m = 0.5):
        '''
        Fit the Gaussian Naive Bayes model to the training data.

        Parameters
           X: array-like, shape (n_samples, n_features)
               The input training data.
           y: array-like, shape (n_samples,)
               The class labels for the training data.
           discrete: list, optional
               List of indices or names of discrete features. If None, all features are treated as continuous.
           correction: str, optional
               Type of correction to apply for discrete features. Options are 'laplace' or 'm-estimate'.
           m: float, optional
                Parameter for m-estimate correction. Default is 0.5.
        '''
        self.classes = np.unique(y)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.features = X.columns
            
        if discrete is not None:
            if not isinstance(discrete, list):
                raise ValueError("Discrete features should be provided as a list.")
            if len(discrete) > X.shape[1]:
                raise ValueError("Number of discrete features cannot exceed number of features in the dataset.")
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X should be a pandas DataFrame to specify discrete columns.")
            self.discrete = set([X.columns.get_loc(col) for col in discrete])
            self.discrete_probs = {}
        
        self.priors = np.zeros(len(self.classes), dtype=float)
        self.params = {}
        for c_i, c in enumerate(self.classes):
            x_c = X[y == c] 
            self.priors[c_i] = len(x_c) / len(X)

            for i in range(X.shape[1]):
                values = x_c.iloc[:, i].values
                if i in self.discrete:
                    expected_values = X.iloc[:, i].unique()
                    # Actual values in the current class subset
                    unique, counts = np.unique(values, return_counts=True)
                    counts_dict = dict(zip(unique, counts))
                    # Map to expected values, defaulting to 0 if missing
                    full_counts = np.array([counts_dict.get(val, 0) for val in expected_values])
                    if correction is None:
                        total = full_counts.sum()
                        probs = full_counts / total if total > 0 else np.zeros_like(full_counts, dtype=float)
                    elif correction == 'laplace':
                        probs = (full_counts + 1) / (full_counts.sum() + len(self.classes))
                    elif correction == 'm-estimate':
                        probs = (full_counts + m * self.priors[c]) / (full_counts.sum() + m)
                    else:
                        raise ValueError(
                            f"Unsupported correction method '{correction}'. "
                            "Valid options are None, 'laplace', or 'm-estimate'."
                        )
                    self.discrete_probs[(c_i, i)] = dict(zip(expected_values, probs))
                else:
                    # Continuous feature — GaussianNB logic
                    mean = values.mean()
                    var = values.var()
                    self.params[(c_i, i)] = {
                        'mean': mean,
                        'var': var
                    }
    
    def __predict_probas_single(self,x):
        '''
        Predict the posterior probabilities for a single instance x.
        
        Parameters
           x: array-like, shape (n_features,)
               The input instance for which to predict the posterior probabilities.
        
        Returns
           posteriors: array-like, shape (n_classes,)s
                The posterior probabilities for each class.
        '''
        posteriors = self.priors.copy()
        for c_i in range(len(self.classes)):
            for i in range(len(self.features)):
                if i in self.discrete:
                    posteriors[c_i] *= self.discrete_probs[(c_i, i)].get(x[i], self.discrete_probs[(c_i, i)]['default'])
                else:
                    mean = self.params[(c_i, i)]['mean']
                    var = self.params[(c_i, i)]['var']
                    posteriors[c_i] *= norm.pdf(x[i], loc = mean, scale = np.sqrt(var))
        return posteriors / np.sum(posteriors)

    def predict_proba(self, X):
        '''
        Predict the posterior probabilities for each class for the given input data X.
        
        Parameters
           X: array-like, shape (n_samples, n_features)
               The input data for which to predict the posterior probabilities.
        
        Returns
           posteriors: array-like, shape (n_samples, n_classes)
                The posterior probabilities for each class for each input instance.
        '''
        if self.priors is None:
            raise ValueError("Model has not been fitted yet.")
        if isinstance(X,pd.DataFrame):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.features):
            raise ValueError("Number of features in X does not match number of features used for training.")
        
        posteriors = np.zeros((X.shape[0], len(self.classes)), dtype=float)
        for i,x in enumerate(X):
            posteriors[i] = self.__predict_probas_single(x)
        return posteriors

    
    def predict(self, X):
        '''
        Predict the class labels for the given input data X.
        
        Parameters
           X: array-like, shape (n_samples, n_features)
               The input data for which to predict the class labels.
        
        Returns
           predictions: array-like, shape (n_samples,)
               The predicted class labels for each input instance.
        '''
        if self.priors is None:
            raise ValueError("Model has not been fitted yet.")
        if isinstance(X,pd.DataFrame):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.features):
            raise ValueError("Number of features in X does not match number of features used for training.")
        
        predictions = np.zeros(X.shape[0], dtype=self.classes.dtype)
        for i,x in enumerate(X):
            predictions[i] = self.classes[np.argmax(self.__predict_probas_single(x))]
        return predictions
        
        
        
        

        

