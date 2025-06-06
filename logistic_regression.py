from typing import NoReturn
import numpy as np
from base_estimator import BaseEstimator
from gradient_descent import GradientDescent
from modules import L1, L2 ,LogisticModule, RegularizedModule



class LogisticRegression(BaseEstimator):
    """
    Logistic Regression Classifier

    Attributes
    ----------
    solver_: GradientDescent, default=GradientDescent()
        Descent method solver to use for the logistic regression objective optimization

    penalty_: str, default="none"
        Type of regularization term to add to logistic regression objective. Supported values
        are "none", "l1", "l2"

    lam_: float, default=1
        Regularization parameter to be used in case `self.penalty_` is not "none"

    alpha_: float, default=0.5
        Threshold value by which to convert class probability to class value

    include_intercept_: bool, default=True
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LogisticRegression.fit` function.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        solver: GradientDescent, default=GradientDescent()
            Descent method solver to use for the logistic regression objective optimization

        penalty: str, default="none"
            Type of regularization term to add to logistic regression objective. Supported values
            are "none", "l1", "l2"

        lam: float, default=1
            Regularization parameter to be used in case `self.penalty_` is not "none"

        alpha: float, default=0.5
            Threshold value by which to convert class probability to class value

        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.solver_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        n_features = X.shape[1] + int(self.include_intercept_)
        if self.include_intercept_:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.coefs_ = np.random.normal(0, 1 / n_features, n_features)
        if self.penalty_ == "none":
            objective_module = LogisticModule(weights=self.coefs_)
        elif self.penalty_ == "l1":
            objective_module = RegularizedModule(fidelity_module=LogisticModule(), regularization_module=L1(),
                                                     lam=self.lam_, weights=self.coefs_, include_intercept=self.include_intercept_)
        elif self.penalty_ == "l2":
            objective_module = RegularizedModule(fidelity_module=LogisticModule(), regularization_module=L2(),
                                                     lam=self.lam_, weights=self.coefs_, include_intercept=self.include_intercept_)
        self.solver_.fit(objective_module, X, y)
        self.coefs_ = objective_module.weights

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= self.alpha_).astype(int)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        if self.coefs_ is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before prediciting probabilities.")
        if self.include_intercept_:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        z = X @ self.coefs_
        probabilities = 1 / (1 + np.exp(-z))
        return probabilities

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        predictions = self._predict(X)
        return np.mean(predictions != y)

