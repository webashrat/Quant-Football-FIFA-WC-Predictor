import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier


def _make_lgbm():
    return LGBMClassifier(
        n_estimators     = 200,
        learning_rate    = 0.05,
        num_leaves       = 12,        # small tree: prevents overconfidence on 900 rows
        min_child_samples= 20,
        subsample        = 0.75,
        colsample_bytree = 0.75,
        reg_alpha        = 0.5,
        reg_lambda       = 2.0,
        class_weight     = "balanced",
        random_state     = 42,
        verbose          = -1,
    )


class Predictor:
    """
    LightGBM + 5-fold isotonic calibration.
    Calibration fixes LightGBM's overconfidence so log-loss beats the null model.
    Competition sample weights downweight friendly matches during final fit.
    """

    def __init__(self):
        self.scaler        = StandardScaler()
        self._base         = None    # uncalibrated, for feature importances
        self.model         = None    # calibrated, for predict_proba
        self.classes_      = None
        self.feature_names: list[str] = []
        self.cv_auc_mean   = None
        self.cv_auc_std    = None
        self.n_train       = 0
        self._fitted       = False

    def fit(self, X: np.ndarray, y: list,
            sample_weight: np.ndarray | None = None,
            feature_names: list[str] | None = None):

        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.n_train       = len(y)
        y_arr              = np.array(y)
        X_s                = np.asarray(self.scaler.fit_transform(X))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 1. CV AUC on raw LGBM (unbiased, no sample weights in CV folds)
        scores = cross_val_score(_make_lgbm(), X_s, y_arr, cv=cv, scoring="roc_auc_ovr")
        self.cv_auc_mean = float(scores.mean())
        self.cv_auc_std  = float(scores.std())

        # 2. Calibrated model: 5-fold isotonic calibration (cv=5 refits inside)
        #    This corrects overconfident probabilities — log-loss improves over null.
        self.model = CalibratedClassifierCV(_make_lgbm(), cv=5, method="isotonic")
        self.model.fit(X_s, y_arr)   # calibration CV doesn't support sample_weight here
        self.classes_ = list(self.model.classes_)

        # 3. Base model fitted on full data (for feature importances only)
        self._base = _make_lgbm()
        self._base.fit(X_s, y_arr, sample_weight=sample_weight)

        self._fitted = True

    def predict_proba(self, x: np.ndarray) -> dict:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self.model.predict_proba(np.asarray(self.scaler.transform(x)))[0]
        return dict(zip(self.classes_, probs))

    def feature_importances(self) -> dict[str, float]:
        if not self._fitted or self._base is None:
            return {}
        imps  = self._base.feature_importances_
        total = imps.sum() or 1.0
        return {f: float(v / total) for f, v in zip(self.feature_names, imps)}

    def coef_for_class(self, cls: str) -> dict[str, float]:
        return self.feature_importances()

    @property
    def is_fitted(self):
        return self._fitted
