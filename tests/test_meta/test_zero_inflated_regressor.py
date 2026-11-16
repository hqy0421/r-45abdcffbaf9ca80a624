"""Test the ZeroInflatedRegressor."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from sklego.meta import ZeroInflatedRegressor
from sklego.testing import check_shape_remains_same_regressor

params = dict(n_estimators=10, max_depth=3, random_state=0, n_jobs=-1)
# Single-threaded, fully deterministic params for tests that assert exact equality between two fits
# (`n_jobs=-1` introduces last-bit floating point variation between otherwise identical fits).
det_params = dict(n_estimators=10, max_depth=3, random_state=0)


@pytest.mark.parametrize("test_fn", [check_shape_remains_same_regressor])
def test_zir(test_fn):
    regr = ZeroInflatedRegressor(classifier=ExtraTreesClassifier(**params), regressor=ExtraTreesRegressor(**params))
    test_fn(ZeroInflatedRegressor.__name__, regr)


@parametrize_with_checks(
    [ZeroInflatedRegressor(classifier=ExtraTreesClassifier(**params), regressor=ExtraTreesRegressor(**params))]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_zero_inflated_example():
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    X = np.random.randn(10_000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)  # many zeroes here, in about 75% of the cases.

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
        regressor=ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
    )

    zir_score = cross_val_score(zir, X, y).mean()
    et_score = cross_val_score(ExtraTreesRegressor(), X, y).mean()

    assert zir_score > 0.85
    assert zir_score > et_score


@pytest.mark.parametrize(
    "classifier,regressor,performance",
    [
        (
            ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
            ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
            0.85,
        ),
        (KNeighborsClassifier(), KNeighborsRegressor(), 0.55),
    ],
)
def test_zero_inflated_with_sample_weights_example(classifier, regressor, performance):
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    X = np.random.randn(10_000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)  # many zeroes here, in about 75% of the cases.

    zir = ZeroInflatedRegressor(classifier=classifier, regressor=regressor)
    zir_score = cross_val_score(zir, X, y, params={"sample_weight": np.arange(len(y))}).mean()

    assert zir_score > performance


def test_zero_inflated_with_handle_zero_ignore_example():
    """Test that if handle_zero='ignore' and all y are 0, no Exception will be thrown"""

    np.random.seed(0)
    size = 1_000
    X = np.random.randn(size, 4)
    y = np.zeros(size)  # all outputs are 0

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
        regressor=ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
        handle_zero="ignore",
    ).fit(X, y)

    # The predicted values should all be 0
    assert (zir.predict(X) == np.zeros(size)).all()


def test_wrong_estimators_exceptions():
    X = np.array([[0.0]])
    y = np.array([0.0])

    with pytest.raises(ValueError, match="`classifier` has to be a classifier."):
        zir = ZeroInflatedRegressor(ExtraTreesRegressor(), ExtraTreesRegressor())
        zir.fit(X, y)

    with pytest.raises(ValueError, match="`regressor` has to be a regressor."):
        zir = ZeroInflatedRegressor(ExtraTreesClassifier(), ExtraTreesClassifier())
        zir.fit(X, y)

    with pytest.raises(
        ValueError, match="`handle_zero` has to be one of \('ignore', 'error'\). Received 'ignor' instead."
    ):
        zir = ZeroInflatedRegressor(
            classifier=ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
            regressor=ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
            handle_zero="ignor",
        )
        zir.fit(X, y)

    error_text = """The predicted training labels are all zero, making the regressor obsolete\. Change the classifier
                or use a plain regressor instead\. Alternatively, you can choose to ignore that predicted labels are
                all zero by setting flag handle_zero = 'ignore'"""

    with pytest.raises(ValueError, match=error_text):
        zir = ZeroInflatedRegressor(
            classifier=ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
            regressor=ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
        )
        zir.fit(X, y)  # default is handle_zero = 'error'


def approx_lte(x, y):
    return ((x <= y) | np.isclose(x, y)).all()


def approx_gte(x, y):
    return ((x >= y) | np.isclose(x, y)).all()


def test_score_samples():
    np.random.seed(0)
    X = np.random.randn(1_000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
        regressor=ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
    ).fit(X, y)

    scores = zir.score_samples(X)
    preds = zir.predict(X)

    pred_is_non_zero = zir.classifier_.predict(X)

    # Where the classifier prediction is non-zero, then the value is multiplied by something less than 1.
    assert approx_lte(scores[pred_is_non_zero], preds[pred_is_non_zero])
    # Where the classifier prediction is zero, then the score is by something greater than 0.
    assert approx_gte(scores[~pred_is_non_zero], preds[~pred_is_non_zero])


def test_no_predict_proba():
    np.random.seed(0)
    X = np.random.randn(1_000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)

    zir = ZeroInflatedRegressor(
        classifier=RidgeClassifier(),
        regressor=ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
    ).fit(X, y)

    with pytest.raises(AttributeError, match="This 'ZeroInflatedRegressor' has no attribute 'score_samples'"):
        zir.score_samples(X)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier that records the data it was fitted on."""

    def fit(self, X, y, sample_weight=None):
        self.X_fit_ = np.asarray(X)
        self.y_fit_ = np.asarray(y)
        self.sample_weight_fit_ = None if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(self.y_fit_)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[-1])


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """A regressor that records the data it was fitted on."""

    def fit(self, X, y, sample_weight=None):
        self.X_fit_ = np.asarray(X)
        self.y_fit_ = np.asarray(y)
        self.sample_weight_fit_ = None if sample_weight is None else np.asarray(sample_weight)
        return self

    def predict(self, X):
        return np.ones(len(X))


def test_fit_uses_only_effective_samples_with_weights():
    """Zero-weight samples are excluded from both stages; the remaining weights are forwarded, not dropped."""
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0])
    sample_weight = np.array([1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0])

    classifier = RecordingClassifier()
    regressor = RecordingRegressor()
    zir = ZeroInflatedRegressor(classifier=classifier, regressor=regressor).fit(X, y, sample_weight=sample_weight)

    effective = sample_weight > 0
    # The classifier learns whether the *effective* samples are non-zero, weighted accordingly.
    assert_array_equal(zir.classifier_.X_fit_, X[effective])
    assert_array_equal(zir.classifier_.y_fit_, (y != 0)[effective])
    assert_array_equal(zir.classifier_.sample_weight_fit_, sample_weight[effective])

    # The regressor only sees the effective non-zero samples, with their weights.
    non_zero = effective & (y != 0)
    assert_array_equal(zir.regressor_.X_fit_, X[non_zero])
    assert_array_equal(zir.regressor_.y_fit_, y[non_zero])
    assert_array_equal(zir.regressor_.sample_weight_fit_, sample_weight[non_zero])


def test_zero_weight_samples_do_not_introduce_non_zero_class():
    """If every non-zero target has zero weight, the classifier must not see the non-zero class."""
    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = np.zeros(100)
    y[:10] = 5.0  # non-zero targets ...
    sample_weight = np.ones(100)
    sample_weight[:10] = 0.0  # ... but all of them are zero-weight

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(**params),
        regressor=ExtraTreesRegressor(**params),
        handle_zero="ignore",
    ).fit(X, y, sample_weight=sample_weight)

    assert_array_equal(zir.classifier_.classes_, [False])
    assert_array_equal(zir.predict(X), np.zeros(100))
    assert_array_equal(zir.score_samples(X), np.zeros(100))

    # ... and with handle_zero="error" this is treated as "all train output is zero".
    with pytest.raises(ValueError, match="all zero"):
        ZeroInflatedRegressor(
            classifier=ExtraTreesClassifier(**params),
            regressor=ExtraTreesRegressor(**params),
        ).fit(X, y, sample_weight=sample_weight)


def test_zero_weight_samples_equivalent_to_removal():
    """Fitting with zero weights must be identical to fitting without those samples."""
    np.random.seed(0)
    X = np.random.randn(500, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)

    sample_weight = np.ones(len(y))
    sample_weight[:100] = 0.0

    zir_weighted = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(**det_params), regressor=ExtraTreesRegressor(**det_params)
    ).fit(X, y, sample_weight=sample_weight)
    zir_removed = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(**det_params), regressor=ExtraTreesRegressor(**det_params)
    ).fit(X[100:], y[100:])

    assert_array_equal(zir_weighted.predict(X), zir_removed.predict(X))
    assert_array_equal(zir_weighted.score_samples(X), zir_removed.score_samples(X))


def test_all_zero_sample_weights_error():
    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2])

    with pytest.raises(ValueError, match="at least one non-zero"):
        ZeroInflatedRegressor(classifier=ExtraTreesClassifier(**params), regressor=ExtraTreesRegressor(**params)).fit(
            X, y, sample_weight=np.zeros(len(y))
        )


def test_fit_retrains_independently():
    """A second `fit` must retrain from scratch on the new data, not reuse the previous fit."""
    np.random.seed(0)
    X = np.random.randn(1_000, 4)
    y1 = (X[:, 0] > 0) * np.abs(X[:, 2])
    y2 = (X[:, 1] > 0) * np.abs(X[:, 3])

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(**det_params), regressor=ExtraTreesRegressor(**det_params)
    ).fit(X, y1)

    first_classifier = zir.classifier_
    first_regressor = zir.regressor_

    zir.fit(X, y2)
    assert zir.classifier_ is not first_classifier
    assert zir.regressor_ is not first_regressor

    # The refit model must be identical to a fresh model fitted on the same data.
    zir_fresh = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(**det_params), regressor=ExtraTreesRegressor(**det_params)
    ).fit(X, y2)

    assert_array_equal(zir.predict(X), zir_fresh.predict(X))
    assert_array_equal(zir.score_samples(X), zir_fresh.score_samples(X))


def test_prefitted_prototypes_are_not_reused():
    """Pre-fitted sub-estimators must be retrained on the current data, never reused as-is."""
    np.random.seed(0)
    X = np.random.randn(1_000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)

    # Fit the prototypes on unrelated data first.
    X_other = np.random.randn(1_000, 4)
    y_other = (X_other[:, 0] > 0) * np.abs(X_other[:, 1])
    classifier = ExtraTreesClassifier(**det_params).fit(X_other, y_other != 0)
    regressor = ExtraTreesRegressor(**det_params).fit(X_other, y_other)

    zir = ZeroInflatedRegressor(classifier=classifier, regressor=regressor).fit(X, y)
    zir_fresh = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(**det_params), regressor=ExtraTreesRegressor(**det_params)
    ).fit(X, y)

    # The fitted sub-estimators are fresh clones, not the objects that were passed in ...
    assert zir.classifier_ is not classifier
    assert zir.regressor_ is not regressor
    # ... and they were retrained on the current data.
    assert_array_equal(zir.predict(X), zir_fresh.predict(X))
    assert_array_equal(zir.score_samples(X), zir_fresh.score_samples(X))


def test_prototypes_are_not_mutated():
    """The objects passed at construction time must remain untouched by `fit`."""
    np.random.seed(0)
    X = np.random.randn(200, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2])

    classifier = ExtraTreesClassifier(**params)
    regressor = ExtraTreesRegressor(**params)
    ZeroInflatedRegressor(classifier=classifier, regressor=regressor).fit(X, y)

    with pytest.raises(NotFittedError):
        check_is_fitted(classifier)
    with pytest.raises(NotFittedError):
        check_is_fitted(regressor)


def test_all_non_zero_targets():
    """All non-zero targets: the classifier learns a single class, predict/score_samples still work."""
    np.random.seed(0)
    X = np.random.randn(500, 4)
    y = np.abs(X[:, 0]) + 1.0  # all strictly positive

    zir = ZeroInflatedRegressor(classifier=ExtraTreesClassifier(**params), regressor=ExtraTreesRegressor(**params)).fit(
        X, y
    )

    assert_array_equal(zir.classifier_.classes_, [True])
    predictions = zir.predict(X)
    assert (predictions > 0).all()
    # The probability of the non-zero class is 1, so the score equals the regressor prediction.
    assert_allclose(zir.score_samples(X), zir.regressor_.predict(X))


def test_all_zero_targets_handle_zero_ignore_score_samples():
    """All-zero targets with handle_zero='ignore': score_samples is all zero (there is no non-zero class)."""
    np.random.seed(0)
    X = np.random.randn(200, 4)
    y = np.zeros(200)

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(**params),
        regressor=ExtraTreesRegressor(**params),
        handle_zero="ignore",
    ).fit(X, y)

    assert_array_equal(zir.classifier_.classes_, [False])
    assert_array_equal(zir.predict(X), np.zeros(200))
    assert_array_equal(zir.score_samples(X), np.zeros(200))
