"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Joly Arnaud <arnaud.v.joly@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# Licence: BSD 3 clause

from __future__ import division


import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import issparse

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..externals import six
from ..feature_selection.from_model import _LearntSelectorMixin
from ..utils import check_array, check_random_state, compute_sample_weight
from ..utils.validation import NotFittedError


from ._tree import Criterion
from ._tree import Splitter
from ._tree import DepthFirstTreeBuilder, BestFirstTreeBuilder
from ._tree import Tree
from . import _tree
##################################################################
#EVT code
import libmr
import copy
import math
##################################################################

__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"gini": _tree.Gini, "entropy": _tree.Entropy}

CRITERIA_REG = {"mse": _tree.MSE, "friedman_mse": _tree.FriedmanMSE}

DENSE_SPLITTERS = {"best": _tree.BestSplitter,
                   "presort-best": _tree.PresortBestSplitter,
                   "random": _tree.RandomSplitter}

SPARSE_SPLITTERS = {"best": _tree.BestSparseSplitter,
                    "random": _tree.RandomSparseSplitter}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(six.with_metaclass(ABCMeta, BaseEstimator,
                                          _LearntSelectorMixin)):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self,
                 criterion,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 max_features,
                 max_leaf_nodes,
                 random_state,
                 class_weight=None):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight

        self.n_features_ = None
        self.n_outputs_ = None
        self.classes_ = None
        self.n_classes_ = None
##################################################################
#EVT code
        self.leaf_parents = None
        self.confidence = None
        self.pertinence = None
        self.thresholds = None
        self.parent_children = None
        self.node_points = None
##################################################################
        self.tree_ = None
        self.max_features_ = None

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression). In the regression case, use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based "
                                     "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        is_classification = isinstance(self, ClassifierMixin)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            for k in range(self.n_outputs_):
                classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original)

        else:
            self.classes_ = [None] * self.n_outputs_
            self.n_classes_ = [1] * self.n_outputs_

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if self.min_samples_split <= 0:
            raise ValueError("min_samples_split must be greater than zero.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than zero.")
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either smaller than "
                              "0 or larger than 1").format(max_leaf_nodes))

        if sample_weight is not None:
            if (getattr(sample_weight, "dtype", None) != DOUBLE or
                    not sample_weight.flags.contiguous):
                sample_weight = np.ascontiguousarray(
                    sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more "
                                 "than one dimension: %d" %
                                 len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError("Number of weights=%d does not match "
                                 "number of samples=%d" %
                                 (len(sample_weight), n_samples))

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        # Set min_samples_split sensibly
        min_samples_split = max(self.min_samples_split,
                                2 * self.min_samples_leaf)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
                                                         self.n_classes_)
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion,
                                                self.max_features_,
                                                self.min_samples_leaf,
                                                min_weight_leaf,
                                                random_state)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                            self.min_samples_leaf,
                                            min_weight_leaf,
                                            max_depth)
        else:
            builder = BestFirstTreeBuilder(splitter, min_samples_split,
                                           self.min_samples_leaf,
                                           min_weight_leaf,
                                           max_depth,
                                           max_leaf_nodes)

        builder.build(self.tree_, X, y, sample_weight)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            
##################################################################
#EVT code
        self.leaf_parents, self.parent_children = self.fill_evt_leaf_dict()
        self.find_leaf_points(X,y)
        
    
        print('# nodes: ' + str(self.tree_.node_count) + ',  # leafs: ' + str(len(self.leaf_parents)))
##################################################################
            
        return self

##################################################################
#EVT code
    def fill_evt_leaf_dict(self):
        """find all the nodes that are the parent of a leaf and return a dictionary which stores the parent for each leaf"""
       
        left = self.tree_.children_left
        right = self.tree_.children_right
        leaf_parents = dict()
        parent_children = dict()

        if self.tree_.feature[0] == -2:
            leaf_parents[0] = [0]
        self.recurse_find_leaf(left[0], [0], left, right, leaf_parents, parent_children)
        self.recurse_find_leaf(right[0],[0], left, right, leaf_parents, parent_children)
        return leaf_parents, parent_children
        
    def recurse_find_leaf(self, node, path, left, right, leaf_parents, parent_children):
        """retursive function used by fill_evt_leaf_dict which iterates through the nodes of the tree to find parents of leaf nodes"""
        if self.tree_.feature[node] == -2:
            new_path = copy.deepcopy(path)
            new_path.append(node)
            leaf_parents[node] = new_path
            if new_path[-2] in parent_children:
                parent_children[new_path[-2]].append(node)
            else:
                parent_children[new_path[-2]] = [node]
        else:
            new_path = copy.deepcopy(path)
            new_path.append(node)
            self.recurse_find_leaf(left[node], new_path, left, right, leaf_parents, parent_children)
            self.recurse_find_leaf(right[node], new_path, left, right, leaf_parents, parent_children)

    def fill_node_points(self, leaf_points, node_points):
        self.help_fill_node_points(0, leaf_points, node_points)
        
    def help_fill_node_points(self, node, leaf_points, node_points):
        if self.tree_.feature[node] == -2:
            node_points[node] = leaf_points[node]
            return leaf_points[node]
        else:
            left = self.help_fill_node_points(self.tree_.children_left[node], leaf_points, node_points)
            right = self.help_fill_node_points(self.tree_.children_right[node], leaf_points, node_points)
            node_points[node] = left + right
            return left + right
        
    def find_leaf_points(self, X, y):
        """creates the MR models for the the leafs of the tree."""
        self.node_points = dict()
        leaf_classes = dict()
        self.confidence = dict()
        self.pertinence = dict()
        leaf_points = dict()

        #initialize the dictionaries to contain arrays
        for key in self.leaf_parents:
            leaf_points[key] = []
                                   
        #populate the parent_class dict with classes and parent_points with the datapoints it is associated with
        for i in range(X.shape[0]):
            temp_leaf = self.apply(X[i])[0]
            cur_class = self.predict(X[i])[0]
            leaf_classes[temp_leaf] = cur_class
            leaf_points[temp_leaf].append(i)
            
        self.fill_node_points(leaf_points, self.node_points)
         
        for node in self.node_points:
            pertinence_left = libmr.MR()
            pertinence_right = libmr.MR()
            distances = []
            for point in self.node_points[node]:
                distances.append(X[point, self.tree_.feature[node]] - self.tree_.threshold[node])
            if len(distances) > 25:
                tail = 5
            else:
                tail = len(distances) / 5
            if tail < 3:
                tail = 3     
            if len(distances) >= 5:
                pertinence_left.fit_low(distances, tail)
                pertinence_right.fit_high(distances, tail)
                '''if len(distances) < 500:
                    print self.tree_.feature[node]
                    print distances
                    self.make_graph(distances, distances, pertinence_left, pertinence_left, pertinence_right, pertinence_right)'''
                if not pertinence_left.is_valid:
                    pertinence_left = min(distances)
                if not pertinence_right.is_valid:
                    pertinence_right = max(distances)
            else:
                pertinence_left = None
                pertinence_right = None
            self.pertinence[node] = (pertinence_left, pertinence_right)
            
       
        counter = 0
        #Create the libmr EVT model at each leaf
        for leaf in leaf_classes:

            #self.set_thresholds(leaf, node_points, X, y, leaf_classes)
            in_class_distance = []
            non_class_distance = []
            '''for point in node_points[self.leaf_parents[leaf]]:
                if y[point][0] == leaf_classes[leaf]:
                    in_class_distance.append(X[point, self.tree_.feature[self.leaf_parents[leaf]]] - self.tree_.threshold[self.leaf_parents[leaf]])
                else:
                    non_class_distance.append(X[point, self.tree_.feature[self.leaf_parents[leaf]]] - self.tree_.threshold[self.leaf_parents[leaf]])
            class_distances = np.array(in_class_distance)
            non_class_distances = np.array(non_class_distance)
            
            confidence_in = libmr.MR()
            confidence_out = libmr.MR()
            pertinence_in = libmr.MR()
            pertinence_out = libmr.MR()
            
            if len(class_distances) > 250:
                tail = 50
            else:
                tail = len(class_distances) / 5
            if tail < 3:
                tail = 3     
            if len(class_distances) >= 5:
                if np.median(class_distances) > 0:
                    confidence_in.fit_low(class_distances, tail)
                else:
                    confidence_in.fit_high(class_distances, tail)
            else:
                confiedence_in = None
                pertinence_in = None
            if len(non_class_distances) > 250:
                tail = 50
            else:
                tail = len(non_class_distances) / 5
            if tail < 3:
                tail = 3     
            if len(non_class_distances) >= 5:
                if np.median(class_distances) > 0:
                    confidence_out.fit_high(non_class_distances, tail)
                else:
                    confidence_out.fit_low(non_class_distances, tail)
            else:
                confidence_out = None'''
            
            #if confidence_in != None and  confidence_in.is_valid and pertinence_in != None and pertinence_in.is_valid and confidence_out != None and confidence_out.is_valid and pertinence_out != None and pertinence_out.is_valid:
                #self.make_graph(class_distances, non_class_distances, confidence_in, confidence_out, pertinence_in, pertinence_out)
            #self.confidence[leaf] = [confidence_in, confidence_out]
                
    def set_thresholds(self, leaf, node_points, X, y, leaf_classes):
        '''Reset leaf thresholds to more accuratly split the class'''
        self.thresholds = dict()
        in_class_distance= []
        non_class_distance= []
        #Set the threshold to a more balanced location
        for point in node_points[self.leaf_parents[leaf]]:
            if y[point][0] == leaf_classes[leaf]:
                in_class_distance.append(X[point][self.tree_.feature[self.leaf_parents[leaf]]] - self.tree_.threshold[self.leaf_parents[leaf]])
            else:
                non_class_distance.append(X[point][self.tree_.feature[self.leaf_parents[leaf]]] - self.tree_.threshold[self.leaf_parents[leaf]])
        if non_class_distance != []:
            class_distances = np.array(in_class_distance)
            non_class_distances = np.array(non_class_distance)
            self.thresholds[leaf] = self.tree_.threshold[self.leaf_parents[leaf]] + (np.median(class_distances) + np.median(non_class_distances)) / 2.0
        else:
            self.thresholds[leaf] = self.tree_.threshold[self.leaf_parents[leaf]]
        
    def make_graph(self, class_distances, non_class_distances, confidence_in, confidence_out, pertinence_in, pertinence_out):
        import time
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        min_in = np.amin(class_distances)
        min_out = np.amin(non_class_distances)
        max_in = np.amax(class_distances)
        max_out = np.amax(non_class_distances)
        points_in = np.linspace(min_in, max_in, 100)
        points_out = np.linspace(min_out, max_out, 100)
        fig,(ax1,ax2) = plt.subplots(2,1)
        ax1.plot(points_in, 1 - confidence_in.w_score_vector(points_in), label="confidence_in " + str(len(class_distances)))
        ax1.plot(points_out, 1 - confidence_out.w_score_vector(points_out), label="confidence_out " + str(len(non_class_distances)))
        ax2.plot(points_in, 1 - pertinence_in.w_score_vector(points_in), label="pertinence_in " + str(len(class_distances)))
        ax2.plot(points_out, 1 - pertinence_out.w_score_vector(points_out), label="pertinence_out " + str(len(non_class_distances)))
        for point in class_distances:
            ax2.plot(point, .6, marker='o', color='b')
            ax1.plot(point, .6, marker='o', color='b')
        for point in non_class_distances:
            ax2.plot(point, .1, marker='o', color='g')
            ax1.plot(point, .1, marker='o', color='g')
        ax1.legend()
        ax2.legend()
        plt.show()
        time.sleep(5)
##################################################################
    
    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if self.tree_ is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))

        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        n_samples = X.shape[0]
        # Classification
        if isinstance(self, ClassifierMixin):
            if self.n_outputs_ == 1:
                return self.classes_.take(np.argmax(proba, axis=1), axis=0)
            
            else:
                predictions = np.zeros((n_samples, self.n_outputs_))

                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1),
                        axis=0)
                return predictions

        # Regression
        else:
            if self.n_outputs_ == 1:
                return proba[:, 0]

            else:
                return proba[:, :, 0]

    def apply(self, X, check_input=True):
        """
        Returns the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : array_like or sparse matrix, shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array_like, shape = [n_samples,]
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.tree_ is None:
            raise NotFittedError("Estimator not fitted, call `fit` before"
                                 " `feature_importances_`.")

        return self.tree_.compute_feature_importances()


# =============================================================================
# Public estimators
# =============================================================================

class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.

    class_weight : dict, list of dicts, "balanced" or None, optional
                   (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------

    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None):
        super(DecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)
        
##################################################################
#EVT code
    def evt_predict_proba(self, X, check_input=True):
        X = self._validate_X_predict(X, check_input)
        point_class = self.predict_proba(X)
        predictions = []
        for j in range(len(point_class)):
            point_leaf = self.apply(X[j])[0]
            #if point_leaf not in self.thresholds:
                #predictions.append([])
            #else:
            pertinence = 0
            min_pertinence = 1
            product_pertinence = 1
            n_nodes = 0
            for node in self.leaf_parents[point_leaf]:
                if node in self.pertinence and self.pertinence[node][0] != None:
                    distance = X[j, self.tree_.feature[node]] - self.tree_.threshold[node]
                    if type(self.pertinence[node][0]) == np.float64:
                        if distance < self.pertinence[node][0]:
                            low = 0.05
                        else:
                            low = 0.95
                    else:
                        low = 1 - self.pertinence[node][0].w_score(distance)
                        #self.make_graph([distance+10, distance, distance -10], [distance], self.pertinence[node][0], self.pertinence[node][0], self.pertinence[node][0], self.pertinence[node][0])
                    if type(self.pertinence[node][1]) == np.float64:
                        if distance <= self.pertinence[node][1]:
                            high = 0.95
                        else:
                            high = 0.05
                    else:
                        high = 1 - self.pertinence[node][1].w_score(distance)
                       # self.make_graph([distance+10,distance,distance-10], [distance], self.pertinence[node][1], self.pertinence[node][1], self.pertinence[node][1], self.pertinence[node][1])

                    pertinence += min(low, high)
                    product_pertinence *= min(low,high)
                    if min(low, high) < min_pertinence:
                        min_pertinence = min(low, high)
                    n_nodes += 1
            if n_nodes == 0:
                average_pertinence = 0
            else:
                average_pertinence = pertinence / float(n_nodes)
                if product_pertinence != 0.0:
                    product_pertinence = math.log(product_pertinence) / math.log(10.0 ** n_nodes)
            # cur_class = np.argmax(point_class[j])
            #for i in range(len(point_class[j])):
             #   if i != cur_class:
              #      point_class[j][i] = 0
               # else:
                #    point_class[j][i] = 1
            #confidence_in = self.confidence[point_leaf][0].w_score(distance) 
            #confidence_out = self.confidence[point_leaf][1].w_score(distance)
            #confidence = confidence_in + (1 - confidence_out)
            predictions.append((point_class[j], 1, (min_pertinence, average_pertinence, product_pertinence)))
        if len(predictions) == 1:
            predictions = predictions[0]
        return predictions
##################################################################
        
    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)
        
        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)
                
            return all_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = np.log(proba[k])

            return proba


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. The only supported
        criterion is "mse" for the mean squared error.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=n_features`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_leaf_nodes`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
        If not None then ``max_depth`` will be ignored.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    See also
    --------
    DecisionTreeClassifier

    References
    ----------

    .. [1] http://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.cross_validation import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> boston = load_boston()
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, boston.data, boston.target, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
            0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
    """
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None):
        super(DecisionTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)


class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    See also
    --------
    ExtraTreeRegressor, ExtraTreesClassifier, ExtraTreesRegressor

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    """
    def __init__(self,
                 criterion="gini",
                 splitter="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None):
        super(ExtraTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)


class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    See also
    --------
    ExtraTreeClassifier, ExtraTreesClassifier, ExtraTreesRegressor

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    """
    def __init__(self,
                 criterion="mse",
                 splitter="random",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 random_state=None,
                 max_leaf_nodes=None):
        super(ExtraTreeRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)

