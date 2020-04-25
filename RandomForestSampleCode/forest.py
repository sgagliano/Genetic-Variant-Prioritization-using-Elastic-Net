# This file is included in GWAVA and is based on source code from 
# the scikit-learn library. The original code is released under
# the BSD 3 clause license and, as required by this license, the
# original copyright notice is included here. All changes to the
# original code are released under the Apache License, Version 2.0
# and a copyright notice for these is also included below.

# Original code license:

# Copyright (c) 2007-2013 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  a. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  b. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  c. Neither the name of the Scikit-learn Developers  nor the names of
#     its contributors may be used to endorse or promote products
#     derived from this software without specific prior written
#     permission. 
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

# Modified code license:

# Copyright 2013 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on 
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
# KIND, either express or implied. See the License for the 
# specific language governing permissions and limitations 
# under the License.

"""Forest of trees-based ensemble methods

Those methods include random forests and extremely randomized trees.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestClassifier`` and ``ForestRegressor`` base classes further
  implement the prediction logic by computing an average of the predicted
  outcomes of the sub-estimators.

- The ``RandomForestClassifier`` and ``RandomForestRegressor`` derived
  classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeClassifier`` and ``DecisionTreeRegressor`` as
  sub-estimator implementations.

- The ``ExtraTreesClassifier`` and ``ExtraTreesRegressor`` derived
  classes provide the user with concrete implementations of the
  forest ensemble method using the extremly randomized trees
  ``ExtraTreeClassifier`` and ``ExtraTreeRegressor`` as
  sub-estimator implementations.

"""

# Authors: Gilles Louppe, Brian Holt
# License: BSD 3

import itertools
import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.feature_selection.selector_mixin import SelectorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, \
                   ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score

from sklearn.ensemble.base import BaseEnsemble

__all__ = ["RandomForestClassifier",
           "RandomForestRegressor",
           "ExtraTreesClassifier",
           "ExtraTreesRegressor"]

MAX_INT = np.iinfo(np.int32).max


def _parallel_build_trees(n_trees, forest, X, y,
                          sample_mask, X_argsorted, seed, verbose):
    """Private function used to build a batch of trees within a job."""
    random_state = check_random_state(seed)
    trees = []

    for i in xrange(n_trees):
        if verbose > 1:
            print("building tree %d of %d" % (i + 1, n_trees))
        seed = random_state.randint(MAX_INT)

        tree = forest._make_estimator(append=False)
        tree.set_params(compute_importances=forest.compute_importances)
        tree.set_params(random_state=check_random_state(seed))

        if forest.balance_classes:
            # undersample all but the minority class to equalise the class counts
            class_counts = np.bincount(y)
            minority_class = class_counts.argmin()
            minority_size = class_counts[minority_class]
            indices = []
            for cls in range(len(class_counts)):
                class_count = class_counts[cls]
                if class_count == 0:
                    next
                
                # add in a random sample of indices for this class of
                # the same size as the minority class 

                sample_indices = random_state.randint(0, class_count, minority_size)
                indices.extend(np.where(y==cls)[0][sample_indices])

            tree.fit(X[indices], y[indices],
                     sample_mask=sample_mask, X_argsorted=X_argsorted)
            tree.indices_ = indices
        
        elif forest.bootstrap:
            n_samples = X.shape[0]
            indices = random_state.randint(0, n_samples, n_samples)
            tree.fit(X[indices], y[indices],
                     sample_mask=sample_mask, X_argsorted=X_argsorted)
            tree.indices_ = indices

        else:
            tree.fit(X, y,
                     sample_mask=sample_mask, X_argsorted=X_argsorted)

        trees.append(tree)

    return trees


def _parallel_predict_proba(trees, X, n_classes):
    """Private function used to compute a batch of predictions within a job."""
    p = np.zeros((X.shape[0], n_classes))

    for tree in trees:
        if n_classes == tree.n_classes_:
            p += tree.predict_proba(X)

        else:
            proba = tree.predict_proba(X)

            for j, c in enumerate(tree.classes_):
                p[:, c] += proba[:, j]

    return p


def _parallel_predict_regression(trees, X):
    """Private function used to compute a batch of predictions within a job."""
    return sum(tree.predict(X) for tree in trees)


def _partition_trees(forest):
    """Private function used to partition trees between jobs."""
    # Compute the number of jobs
    if forest.n_jobs == -1:
        n_jobs = min(cpu_count(), forest.n_estimators)

    else:
        n_jobs = min(forest.n_jobs, forest.n_estimators)

    # Partition trees between jobs
    n_trees = [forest.n_estimators / n_jobs] * n_jobs

    for i in xrange(forest.n_estimators % n_jobs):
        n_trees[i] += 1

    starts = [0] * (n_jobs + 1)

    for i in xrange(1, n_jobs + 1):
        starts[i] = starts[i - 1] + n_trees[i - 1]

    return n_jobs, n_trees, starts


def _parallel_X_argsort(X):
    """Private function used to sort the features of X."""
    return np.asarray(np.argsort(X.T, axis=1).T, dtype=np.int32, order="F")


def _partition_features(forest, n_total_features):
    """Private function used to partition features between jobs."""
    # Compute the number of jobs
    if forest.n_jobs == -1:
        n_jobs = min(cpu_count(), n_total_features)

    else:
        n_jobs = min(forest.n_jobs, n_total_features)

    # Partition features between jobs
    n_features = [n_total_features / n_jobs] * n_jobs

    for i in xrange(n_total_features % n_jobs):
        n_features[i] += 1

    starts = [0] * (n_jobs + 1)

    for i in xrange(1, n_jobs + 1):
        starts[i] = starts[i - 1] + n_features[i - 1]

    return n_jobs, n_features, starts


class BaseForest(BaseEnsemble, SelectorMixin):
    """Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, base_estimator,
                       n_estimators=10,
                       estimator_params=[],
                       bootstrap=False,
                       compute_importances=False,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        super(BaseForest, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.bootstrap = bootstrap
        self.compute_importances = compute_importances
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)

        self.feature_importances_ = None
        self.verbose = verbose

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # Precompute some data
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        if self.bootstrap:
            sample_mask = None
            X_argsorted = None

        else:
            if self.oob_score:
                raise ValueError("Out of bag estimation only available"
                        " if bootstrap=True")

            sample_mask = np.ones((X.shape[0],), dtype=np.bool)

            n_jobs, _, starts = _partition_features(self, X.shape[1])

            all_X_argsorted = Parallel(n_jobs=n_jobs)(
                delayed(_parallel_X_argsort)(
                    X[:, starts[i]:starts[i + 1]])
                for i in xrange(n_jobs))

            X_argsorted = np.asfortranarray(np.hstack(all_X_argsorted))

        if isinstance(self.base_estimator, ClassifierMixin):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y = np.searchsorted(self.classes_, y)

        # Assign chunk of trees to jobs
        n_jobs, n_trees, _ = _partition_trees(self)

        # Parallel loop
        all_trees = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_trees)(
                n_trees[i],
                self,
                X,
                y,
                sample_mask,
                X_argsorted,
                self.random_state.randint(MAX_INT),
                verbose=self.verbose)
            for i in xrange(n_jobs))

        # Reduce
        self.estimators_ = [tree for tree in itertools.chain(*all_trees)]

        # Calculate out of bag predictions and score
        if self.oob_score:
            if isinstance(self, ClassifierMixin):
                predictions = np.zeros((X.shape[0], self.n_classes_))
                for estimator in self.estimators_:
                    mask = np.ones(X.shape[0], dtype=np.bool)
                    mask[estimator.indices_] = False
                    predictions[mask, :] += estimator.predict_proba(X[mask, :])

                self.oob_decision_function_ = (predictions
                        / predictions.sum(axis=1)[:, np.newaxis])
                self.oob_score_ = np.mean(y == np.argmax(predictions, axis=1))

            else:
                # Regression:
                predictions = np.zeros(X.shape[0])
                n_predictions = np.zeros(X.shape[0])
                for estimator in self.estimators_:
                    mask = np.ones(X.shape[0], dtype=np.bool)
                    mask[estimator.indices_] = False
                    predictions[mask] += estimator.predict(X[mask, :])
                    n_predictions[mask] += 1
                predictions /= n_predictions

                self.oob_prediction_ = predictions
                self.oob_score_ = r2_score(y, predictions)

        # Sum the importances
        if self.compute_importances:
            self.feature_importances_ = \
                sum(tree.feature_importances_ for tree in self.estimators_) \
                / self.n_estimators

        return self


class ForestClassifier(BaseForest, ClassifierMixin):
    """Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, base_estimator,
                       n_estimators=10,
                       estimator_params=[],
                       bootstrap=False,
                       compute_importances=False,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        super(ForestClassifier, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            compute_importances=compute_importances,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the majority
        prediction of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        return self.classes_.take(
            np.argmax(self.predict_proba(X), axis=1),  axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        # Check data
        X = np.atleast_2d(X)

        # Assign chunk of trees to jobs
        n_jobs, n_trees, starts = _partition_trees(self)

        # Parallel loop
        all_p = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                X, self.n_classes_)
            for i in xrange(n_jobs))

        # Reduce
        p = sum(all_p) / self.n_estimators

        return p

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the mean predicted class log-probabilities of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class log-probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        return np.log(self.predict_proba(X))


class ForestRegressor(BaseForest, RegressorMixin):
    """Base class for forest of trees-based regressors.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, base_estimator,
                       n_estimators=10,
                       estimator_params=[],
                       bootstrap=False,
                       compute_importances=False,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        super(ForestRegressor, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            compute_importances=compute_importances,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y: array of shape = [n_samples]
            The predicted values.
        """
        # Check data
        X = np.atleast_2d(X)

        # Assign chunk of trees to jobs
        n_jobs, n_trees, starts = _partition_trees(self)

        # Parallel loop
        all_y_hat = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]:starts[i + 1]], X)
            for i in xrange(n_jobs))

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat


class RandomForestClassifier(ForestClassifier):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of classifical
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=1)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    min_density : float, optional (default=0.1)
        This parameter controls a trade-off in an optimization heuristic. It
        controls the minimum density of the `sample_mask` (i.e. the
        fraction of samples in the mask). If the density falls below this
        threshold the mask is recomputed and the input data is packed
        which results in data copying.  If `min_density` equals to one,
        the partitions are always represented as copies of the original
        data. Otherwise, partitions are represented as bit masks (aka
        sample masks).
        Note: this parameter is tree-specific.

    max_features : int, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
          - If "auto", then `max_features=sqrt(n_features)` on
            classification tasks and `max_features=n_features` on regression
            problems.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: this parameter is tree-specific.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    compute_importances : boolean, optional (default=True)
        Whether feature importances are computed and stored into the
        ``feature_importances_`` attribute when calling fit.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controlls the verbosity of the tree building process.

    Attributes
    ----------
    `feature_importances_` : array, shape = [n_features]
        The feature importances (the higher, the more important the feature).

    `oob_score_` : float
        Score of the training dataset obtained using an out-of-bag estimate.

    `oob_decision_function_` : array, shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set.


    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeClassifier, ExtraTreesClassifier
    """
    def __init__(self, n_estimators=10,
                       criterion="gini",
                       max_depth=None,
                       min_samples_split=1,
                       min_samples_leaf=1,
                       min_density=0.1,
                       max_features="auto",
                       bootstrap=True,
                       balance_classes=False,
                       compute_importances=False,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        super(RandomForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                "min_samples_leaf", "min_density", "max_features",
                "random_state"),
            bootstrap=bootstrap,
            compute_importances=compute_importances,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.max_features = max_features
        self.balance_classes = balance_classes


class RandomForestRegressor(ForestRegressor):
    """A random forest regressor.

    A random forest is a meta estimator that fits a number of classifical
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. The only supported
        criterion is "mse" for the mean squared error.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=1)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    min_density : float, optional (default=0.1)
        This parameter controls a trade-off in an optimization heuristic. It
        controls the minimum density of the `sample_mask` (i.e. the
        fraction of samples in the mask). If the density falls below this
        threshold the mask is recomputed and the input data is packed
        which results in data copying.  If `min_density` equals to one,
        the partitions are always represented as copies of the original
        data. Otherwise, partitions are represented as bit masks (aka
        sample masks).
        Note: this parameter is tree-specific.

    max_features : int, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
          - If "auto", then `max_features=sqrt(n_features)` on
            classification tasks and `max_features=n_features`
            on regression problems.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: this parameter is tree-specific.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    compute_importances : boolean, optional (default=True)
        Whether feature importances are computed and stored into the
        ``feature_importances_`` attribute when calling fit.

    oob_score : bool
        whether to use out-of-bag samples to estimate
        the generalization error.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controlls the verbosity of the tree building process.

    Attributes
    ----------
    `feature_importances_` : array of shape = [n_features]
        The feature mportances (the higher, the more important the feature).

    `oob_score_` : float
        Score of the training dataset obtained using an out-of-bag estimate.

    `oob_prediction_` : array, shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.



    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    See also
    --------
    DecisionTreeRegressor, ExtraTreesRegressor
    """
    def __init__(self, n_estimators=10,
                       criterion="mse",
                       max_depth=None,
                       min_samples_split=1,
                       min_samples_leaf=1,
                       min_density=0.1,
                       max_features="auto",
                       bootstrap=True,
                       balance_classes=False,
                       compute_importances=False,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        super(RandomForestRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                "min_samples_leaf", "min_density", "max_features",
                "random_state"),
            bootstrap=bootstrap,
            compute_importances=compute_importances,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.max_features = max_features
        self.balance_classes = balance_classes


class ExtraTreesClassifier(ForestClassifier):
    """An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=1)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    min_density : float, optional (default=0.1)
        This parameter controls a trade-off in an optimization heuristic. It
        controls the minimum density of the `sample_mask` (i.e. the
        fraction of samples in the mask). If the density falls below this
        threshold the mask is recomputed and the input data is packed
        which results in data copying.  If `min_density` equals to one,
        the partitions are always represented as copies of the original
        data. Otherwise, partitions are represented as bit masks (aka
        sample masks).
        Note: this parameter is tree-specific.

    max_features : int, string or None, optional (default="auto")
        The number of features to consider when looking for the best split.
          - If "auto", then `max_features=sqrt(n_features)` on
            classification tasks and `max_features=n_features`
            on regression problems.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: this parameter is tree-specific.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    compute_importances : boolean, optional (default=True)
        Whether feature importances are computed and stored into the
        ``feature_importances_`` attribute when calling fit.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controlls the verbosity of the tree building process.

    Attributes
    ----------
    `feature_importances_` : array of shape = [n_features]
        The feature mportances (the higher, the more important the feature).

    `oob_score_` : float
        Score of the training dataset obtained using an out-of-bag estimate.

    `oob_decision_function_` : array, shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.
    """
    def __init__(self, n_estimators=10,
                       criterion="gini",
                       max_depth=None,
                       min_samples_split=1,
                       min_samples_leaf=1,
                       min_density=0.1,
                       max_features="auto",
                       bootstrap=False,
                       balance_classes=False,
                       compute_importances=False,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        super(ExtraTreesClassifier, self).__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                "min_samples_leaf", "min_density", "max_features",
                "random_state"),
            bootstrap=bootstrap,
            compute_importances=compute_importances,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.max_features = max_features
        self.balance_classes = balance_classes


class ExtraTreesRegressor(ForestRegressor):
    """An extra-trees regressor.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. The only supported
        criterion is "mse" for the mean squared error.
        Note: this parameter is tree-specific.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Note: this parameter is tree-specific.

    min_samples_split : integer, optional (default=1)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.

    min_samples_leaf : integer, optional (default=1)
        The minimum number of samples in newly created leaves.  A split is
        discarded if after the split, one of the leaves would contain less then
        ``min_samples_leaf`` samples.
        Note: this parameter is tree-specific.

    min_density : float, optional (default=0.1)
        This parameter controls a trade-off in an optimization heuristic. It
        controls the minimum density of the `sample_mask` (i.e. the
        fraction of samples in the mask). If the density falls below this
        threshold the mask is recomputed and the input data is packed
        which results in data copying.  If `min_density` equals to one,
        the partitions are always represented as copies of the original
        data. Otherwise, partitions are represented as bit masks (aka
        sample masks).
        Note: this parameter is tree-specific.

    max_features : int, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
          - If "auto", then `max_features=sqrt(n_features)` on
            classification tasks and `max_features=n_features`
            on regression problems.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: this parameter is tree-specific.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.
        Note: this parameter is tree-specific.

    compute_importances : boolean, optional (default=True)
        Whether feature importances are computed and stored into the
        ``feature_importances_`` attribute when calling fit.

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel. If -1, then the number of jobs
        is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controlls the verbosity of the tree building process.

    Attributes
    ----------
    `feature_importances_` : array of shape = [n_features]
        The feature mportances (the higher, the more important the feature).

    `oob_score_` : float
        Score of the training dataset obtained using an out-of-bag estimate.

    `oob_prediction_` : array, shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeRegressor: Base estimator for this ensemble.
    RandomForestRegressor: Ensemble regressor using trees with optimal splits.
    """
    def __init__(self, n_estimators=10,
                       criterion="mse",
                       max_depth=None,
                       min_samples_split=1,
                       min_samples_leaf=1,
                       min_density=0.1,
                       max_features="auto",
                       bootstrap=False,
                       balance_classes=False,
                       compute_importances=False,
                       oob_score=False,
                       n_jobs=1,
                       random_state=None,
                       verbose=0):
        super(ExtraTreesRegressor, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                "min_samples_leaf", "min_density", "max_features",
                "random_state"),
            bootstrap=bootstrap,
            compute_importances=compute_importances,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_density = min_density
        self.max_features = max_features
        self.balance_classes = balance_classes
