import logging
from inspect import signature

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class ZeroInflatedRegressor(RegressorMixin, MetaEstimatorMixin, BaseEstimator):
    """A meta regressor for zero-inflated datasets, i.e. the targets contain a lot of zeroes.

    `ZeroInflatedRegressor` consists of a classifier and a regressor.

    - The classifier's task is to find if the target is zero or not.
    - The regressor's task is to output a (usually positive) prediction whenever the classifier indicates that
    there should be a non-zero prediction.

    The regressor is only trained on examples where the target is non-zero, which makes it easier for it to focus.

    At prediction time, the classifier is first asked if the output should be zero. If yes, output zero.
    Otherwise, ask the regressor for its prediction and output it.

    When `sample_weight` is passed to `.fit()`, it drives both training stages: samples with zero weight are
    excluded from training altogether, so they cannot introduce a class of their own in the classifier.
    The classifier learns from the remaining ("effective") samples whether the target is non-zero, and the
    regressor is trained only on the effective samples with a non-zero target, using their weights.

    Every call to `.fit()` trains fresh clones of `classifier` and `regressor` on the current data: previously
    fitted sub-estimators are never reused, and the objects passed at construction time are never mutated.

    Parameters
    ----------
    classifier : scikit-learn compatible classifier
        A classifier that answers the question "Should the output be zero?".
    regressor : scikit-learn compatible regressor
        A regressor for predicting the target. Its prediction is only used if `classifier` says that the output is
        non-zero.
    handle_zero : Literal["error", "ignore"], default="error"
            How to behave in the case that all train set output consists of zero values only.

            - `handle_zero = 'error'`: will raise a `ValueError` (default).
            - `handle_zero = 'ignore'`: will continue to train the regressor on the entire dataset.

    Attributes
    ----------
    classifier_ : scikit-learn compatible classifier
        The fitted classifier.
    regressor_ : scikit-learn compatible regressor
        The fitted regressor.

    Examples
    --------
    ```py
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklego.meta import ZeroInflatedRegressor

    np.random.seed(0)
    X = np.random.randn(10000, 4)
    y = ((X[:, 0]>0) & (X[:, 1]>0)) * np.abs(X[:, 2] * X[:, 3]**2)

    model = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(random_state=0, max_depth=10),
        regressor=ExtraTreesRegressor(random_state=0)
    ).fit(X, y)

    model.predict(X[:5])
    # array([4.91483294, 0.        , 0.        , 0.04941909, 0.        ])

    model.score_samples(X[:5]).round(2)
    # array([3.73, 0.  , 0.11, 0.03, 0.06])
    ```
    """

    _required_parameters = ["classifier", "regressor"]

    def __init__(self, classifier, regressor, handle_zero="error") -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.handle_zero = handle_zero

    def fit(self, X, y, sample_weight=None):
        """Fit the underlying classifier and regressor using `X` and `y` as training data. The regressor is only
        trained on examples where the target is non-zero.

        Both sub-estimators are fresh clones of `self.classifier` and `self.regressor`, so every call to `fit`
        retrains independently on the current data and the objects passed at construction time are left untouched.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples, ), default=None
            Individual weights for each sample. Samples with zero weight are excluded from both training stages:
            the classifier is trained on the remaining samples, and the regressor on the remaining samples with a
            non-zero target.

        Returns
        -------
        self : ZeroInflatedRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            If `classifier` is not a classifier
            If `regressor` is not a regressor
            If all sample weights are zero
            If all train target entirely consists of zeros and `handle_zero="error"`
        """
        X, y = validate_data(self, X=X, y=y, reset=True)

        if not is_classifier(self.classifier):
            raise ValueError(
                f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead."
            )
        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")
        if self.handle_zero not in {"ignore", "error"}:
            raise ValueError(
                f"`handle_zero` has to be one of {'ignore', 'error'}. Received '{self.handle_zero}' instead."
            )

        sample_weight = _check_sample_weight(sample_weight, X)

        # Samples with zero weight take no part in either training stage, hence they cannot introduce a class
        # of their own in the classifier nor influence the regressor. Note that `_check_sample_weight` already
        # raised if all weights are zero, so at least one effective sample remains under normal circumstances.
        effective_samples = sample_weight > 0
        if not np.any(effective_samples):
            raise ValueError(
                "Cannot fit `ZeroInflatedRegressor`: no samples with a positive sample weight to train on."
            )

        X_eff = X[effective_samples]
        y_eff = y[effective_samples]
        weight_eff = sample_weight[effective_samples]
        non_zero = y_eff != 0

        indices_for_training = np.flatnonzero(non_zero)  # these are the effective non-zero indices
        if indices_for_training.size == 0:
            if self.handle_zero == "error":
                raise ValueError(
                    """The predicted training labels are all zero, making the regressor obsolete. Change the classifier
                or use a plain regressor instead. Alternatively, you can choose to ignore that predicted labels are
                all zero by setting flag handle_zero = 'ignore'"""
                )
            # if we choose to ignore that all train set output is 0
            logging.warning("Regressor will be training on `y` consisting of zero values only.")
            indices_for_training = np.arange(len(y_eff))  # use the whole (effective) train set

        # Always train fresh clones on the current data: previously fitted sub-estimators are never reused and
        # the objects passed at construction time are never mutated.
        self.classifier_ = clone(self.classifier)
        if "sample_weight" in signature(self.classifier_.fit).parameters:
            self.classifier_.fit(X_eff, non_zero, sample_weight=weight_eff)
        else:
            logging.warning("Classifier ignores sample_weight.")
            self.classifier_.fit(X_eff, non_zero)

        self.regressor_ = clone(self.regressor)
        if "sample_weight" in signature(self.regressor_.fit).parameters:
            self.regressor_.fit(
                X_eff[indices_for_training],
                y_eff[indices_for_training],
                sample_weight=weight_eff[indices_for_training],
            )
        else:
            logging.warning("Regressor ignores sample_weight.")
            self.regressor_.fit(
                X_eff[indices_for_training],
                y_eff[indices_for_training],
            )

        return self

    def predict(self, X):
        """Predict target values for `X` using fitted estimator by first asking the classifier if the output should be
        zero. If yes, output zero. Otherwise, ask the regressor for its prediction and output it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ["n_features_in_", "classifier_", "regressor_"])
        X = validate_data(self, X=X, reset=False)

        output = np.zeros(len(X))
        non_zero_indices = np.where(self.classifier_.predict(X))[0]

        if non_zero_indices.size > 0:
            output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices])

        return output

    @available_if(lambda self: hasattr(self.classifier_, "predict_proba"))
    def score_samples(self, X):
        r"""Predict risk estimate of `X` as the probability of `X` to not be zero times the expected value of `X`:

        $$\text{score_sample(X)} = (1-P(X=0)) \cdot E[X]$$

        where:

        - $P(X=0)$ is calculated using the `.predict_proba()` method of the underlying classifier. The probability
        of the non-zero class is looked up via `classifier_.classes_` (the classifier is trained on boolean labels,
        so the non-zero class is `True`), not via a fixed column position.
        - $E[X]$ is the regressor prediction on `X`.

        !!! info

            This method requires the underlying classifier to implement `.predict_proba()` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted risk.
        """

        check_is_fitted(self, ["n_features_in_", "classifier_", "regressor_"])
        X = validate_data(self, X=X, reset=False)

        # The classifier is always trained on boolean labels (`y != 0`), so `True` marks the non-zero class.
        # Its column is looked up explicitly because a fixed position would be wrong when the classifier was
        # trained on a single class (all-zero or all-non-zero targets yield a single probability column).
        non_zero_columns = np.flatnonzero(np.asarray(self.classifier_.classes_, dtype=bool))
        if non_zero_columns.size == 0:
            # The classifier never saw a non-zero target, so the probability of a non-zero output is zero.
            return np.zeros(X.shape[0])

        non_zero_proba = self.classifier_.predict_proba(X)[:, non_zero_columns[0]]
        expected_impact = self.regressor_.predict(X)

        return non_zero_proba * expected_impact
