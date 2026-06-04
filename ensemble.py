"""Ensemble strategies for temporal pavement deterioration forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class FixedRatioEnsemble:
    """Weighted-average ensemble between RF and graph predictions."""

    def __init__(self, weight_rf: float) -> None:
        self.weight_rf = float(weight_rf)

    def predict(self, rf_pred: np.ndarray, gcn_pred: np.ndarray) -> np.ndarray:
        rf_arr = np.asarray(rf_pred, dtype=float)
        gcn_arr = np.asarray(gcn_pred, dtype=float)
        return self.weight_rf * rf_arr + (1.0 - self.weight_rf) * gcn_arr


@dataclass
class MetaFeatureBundle:
    """Bundle of engineered meta-features for stacking."""

    feature_names: list[str]
    train_like_features: np.ndarray
    apply_features: np.ndarray


def fit_climate_pc1(train_matrix: np.ndarray, apply_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a one-component climate PCA on the training-like matrix and transform both sets."""

    train_arr = np.asarray(train_matrix, dtype=float)
    apply_arr = np.asarray(apply_matrix, dtype=float)
    if train_arr.ndim == 1:
        train_arr = train_arr.reshape(-1, 1)
    if apply_arr.ndim == 1:
        apply_arr = apply_arr.reshape(-1, 1)
    if train_arr.shape[1] == 0:
        return np.zeros((train_arr.shape[0],), dtype=float), np.zeros((apply_arr.shape[0],), dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=1)
    train_imp = imputer.fit_transform(train_arr)
    apply_imp = imputer.transform(apply_arr)
    train_std = scaler.fit_transform(train_imp)
    apply_std = scaler.transform(apply_imp)
    train_pc1 = pca.fit_transform(train_std).reshape(-1)
    apply_pc1 = pca.transform(apply_std).reshape(-1)
    return train_pc1, apply_pc1


def build_meta_feature_matrix(
    rf_pred: np.ndarray,
    gcn_pred: np.ndarray,
    node_age_proxy: np.ndarray,
    climate_pc1: np.ndarray,
    traffic_log: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the stacking feature matrix in the requested feature order."""

    features = np.column_stack(
        [
            np.asarray(rf_pred, dtype=float),
            np.asarray(gcn_pred, dtype=float),
            np.asarray(node_age_proxy, dtype=float),
            np.asarray(climate_pc1, dtype=float),
            np.asarray(traffic_log, dtype=float),
        ]
    )
    feature_names = ["rf_pred", "gcn_pred", "node_age_proxy", "climate_pc1", "traffic_log"]
    return features, feature_names


class StackedEnsemble:
    """Stacking ensemble trained strictly on validation predictions."""

    def __init__(self, meta_model: str = "ridge", alphas: tuple[float, ...] = (0.1, 1.0, 10.0), random_state: int = 42) -> None:
        self.meta_model = meta_model
        self.alphas = alphas
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.best_alpha: float | None = None
        self.feature_names: list[str] = []

    @staticmethod
    def _split_meta_validation(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(features)
        if n < 4:
            return features, features, target, target
        split_idx = max(int(round(0.7 * n)), 1)
        split_idx = min(split_idx, n - 1)
        return features[:split_idx], features[split_idx:], target[:split_idx], target[split_idx:]

    def fit(self, features_val: np.ndarray, y_val: np.ndarray, feature_names: list[str] | None = None) -> "StackedEnsemble":
        """Fit the meta-learner on validation predictions only."""

        x_val = np.asarray(features_val, dtype=float)
        y_arr = np.asarray(y_val, dtype=float).reshape(-1)
        x_fit, x_tune, y_fit, y_tune = self._split_meta_validation(x_val, y_arr)
        self.feature_names = feature_names or [f"meta_{idx}" for idx in range(x_val.shape[1])]

        x_fit_scaled = self.scaler.fit_transform(x_fit)
        x_tune_scaled = self.scaler.transform(x_tune)

        if self.meta_model == "ridge":
            best_model = None
            best_score = -np.inf
            for alpha in self.alphas:
                model = Ridge(alpha=alpha)
                model.fit(x_fit_scaled, y_fit)
                score = float(r2_score(y_tune, model.predict(x_tune_scaled)))
                if score > best_score:
                    best_score = score
                    best_model = model
                    self.best_alpha = float(alpha)
            assert best_model is not None
            self.scaler = StandardScaler().fit(x_val)
            self.model = Ridge(alpha=self.best_alpha)
            self.model.fit(self.scaler.transform(x_val), y_arr)
            return self

        if self.meta_model == "mlp":
            self.scaler = StandardScaler().fit(x_val)
            self.model = MLPRegressor(
                hidden_layer_sizes=(32,),
                random_state=self.random_state,
                early_stopping=True,
                max_iter=500,
                validation_fraction=0.2,
            )
            self.model.fit(self.scaler.transform(x_val), y_arr)
            return self

        raise ValueError(f"Unsupported meta_model: {self.meta_model}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""

        if self.model is None:
            raise RuntimeError("StackedEnsemble must be fitted before predict().")
        x = np.asarray(features, dtype=float)
        return np.asarray(self.model.predict(self.scaler.transform(x)), dtype=float)
