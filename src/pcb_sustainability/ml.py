"""Dependency-light ML layer for component sustainability prediction.

The model is a supervised TF-IDF nearest-neighbor regressor implemented with
standard Python. It trains from labeled examples and predicts a 0-100 component
sustainability score without requiring scikit-learn or other heavy packages.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .models import ComponentScore
from .recommendations import make_recommendation
from .utils import clamp, normalize_text


TEXT_COLUMNS = [
    "part_number",
    "value",
    "footprint",
    "manufacturer",
    "description",
    "material",
    "compliance",
    "lifecycle",
]


@dataclass
class MLPrediction:
    """ML prediction with demo-friendly explainability fields."""

    predicted_score: float
    confidence: float
    risk_band: str
    influential_terms: list[str]


def row_to_text(row: pd.Series, enrichment: dict | None = None) -> str:
    """Convert a component row into model text features."""

    enrichment = enrichment or {}
    fields = [normalize_text(row.get(column, "")) for column in TEXT_COLUMNS]
    fields.extend(
        normalize_text(enrichment.get(key, ""))
        for key in ["title", "category", "manufacturer", "package", "availability"]
    )
    return " ".join(field for field in fields if field)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", text.lower())


def _risk_band(score: float) -> str:
    if score < 45:
        return "high risk"
    if score < 70:
        return "moderate risk"
    return "lower risk"


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class MLSustainabilityPredictor:
    """Trainable nearest-neighbor TF-IDF regressor."""

    def __init__(self, k_neighbors: int = 4):
        self.k_neighbors = k_neighbors
        self.idf: dict[str, float] = {}
        self.training_vectors: list[dict[str, float]] = []
        self.training_scores: list[float] = []
        self.training_texts: list[str] = []
        self.is_trained = False

    def train(self, training_df: pd.DataFrame) -> "MLSustainabilityPredictor":
        if "target_score" not in training_df.columns:
            raise ValueError("ML training data must include a target_score column.")

        texts = training_df.apply(row_to_text, axis=1).fillna("").astype(str).tolist()
        tokenized = [_tokens(text) for text in texts]
        document_count = max(len(tokenized), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        self.idf = {
            token: math.log((1 + document_count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        self.training_vectors = [self._vectorize_tokens(tokens) for tokens in tokenized]
        self.training_scores = [
            float(clamp(value))
            for value in pd.to_numeric(training_df["target_score"], errors="coerce").fillna(60)
        ]
        self.training_texts = texts
        self.is_trained = True
        return self

    def predict(self, row: pd.Series, enrichment: dict | None = None) -> MLPrediction:
        if not self.is_trained:
            raise RuntimeError("ML predictor must be trained before prediction.")

        text = row_to_text(row, enrichment)
        vector = self._vectorize_tokens(_tokens(text))
        similarities = [
            (_cosine(vector, training_vector), score)
            for training_vector, score in zip(self.training_vectors, self.training_scores)
        ]
        similarities.sort(key=lambda item: item[0], reverse=True)
        top = similarities[: self.k_neighbors]
        similarity_total = sum(similarity for similarity, _ in top)
        if similarity_total <= 0:
            predicted = sum(self.training_scores) / len(self.training_scores)
        else:
            predicted = sum(similarity * score for similarity, score in top) / similarity_total

        max_similarity = top[0][0] if top else 0.0
        confidence = round(clamp(35 + max_similarity * 60, 0, 90) / 100, 2)
        terms = self._influential_terms(vector)
        score = round(clamp(predicted), 1)
        return MLPrediction(
            predicted_score=score,
            confidence=confidence,
            risk_band=_risk_band(score),
            influential_terms=terms,
        )

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(token for token in tokens if token in self.idf)
        total = sum(counts.values()) or 1
        return {token: (count / total) * self.idf[token] for token, count in counts.items()}

    def _influential_terms(self, vector: dict[str, float], limit: int = 5) -> list[str]:
        return [token for token, _ in sorted(vector.items(), key=lambda item: item[1], reverse=True)[:limit]]

    def save(self, path: str | Path) -> None:
        data = {
            "k_neighbors": self.k_neighbors,
            "idf": self.idf,
            "training_vectors": self.training_vectors,
            "training_scores": self.training_scores,
            "training_texts": self.training_texts,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MLSustainabilityPredictor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        predictor = cls(k_neighbors=int(data.get("k_neighbors", 4)))
        predictor.idf = data["idf"]
        predictor.training_vectors = data["training_vectors"]
        predictor.training_scores = data["training_scores"]
        predictor.training_texts = data.get("training_texts", [])
        predictor.is_trained = True
        return predictor


def train_predictor_from_csv(path_or_buffer) -> MLSustainabilityPredictor:
    """Train a predictor from a CSV containing component fields and target_score."""

    return MLSustainabilityPredictor().train(pd.read_csv(path_or_buffer))


def predict_bom(
    df: pd.DataFrame,
    predictor: MLSustainabilityPredictor,
    enrichments: dict[str, dict] | None = None,
) -> dict[str, MLPrediction]:
    enrichments = enrichments or {}
    predictions: dict[str, MLPrediction] = {}
    for _, row in df.iterrows():
        key = normalize_text(row.get("part_number")) or normalize_text(row.get("description"))
        predictions[key] = predictor.predict(row, enrichments.get(key, {}))
    return predictions


def apply_ml_predictions(
    component_scores: list[ComponentScore],
    predictions: dict[str, MLPrediction],
    blend_weight: float = 0.25,
) -> tuple[list[ComponentScore], dict]:
    """Blend ML predictions into existing rule scores and return a new summary."""

    blend_weight = clamp(blend_weight, 0.0, 1.0)
    for score in component_scores:
        prediction = predictions.get(score.part_number)
        if not prediction:
            continue
        rule_score = score.score
        score.score = round(rule_score * (1 - blend_weight) + prediction.predicted_score * blend_weight, 1)
        score.enrichment["ml_prediction"] = asdict(prediction) | {
            "blend_weight": blend_weight,
            "rule_score_before_ml": rule_score,
        }
        score.recommendations.append(
            make_recommendation(
                "ML sustainability prediction",
                "A trained TF-IDF similarity model predicted "
                f"{prediction.predicted_score}/100 ({prediction.risk_band}) using terms such as "
                f"{', '.join(prediction.influential_terms) or 'available BoM text'}. "
                f"The final component score blends {int(blend_weight * 100)}% ML with the rule score.",
                "low" if prediction.predicted_score >= 70 else "medium",
                prediction.confidence,
                "ml_prediction",
            )
        )

    total_qty = sum(max(item.quantity, 1) for item in component_scores) or 1
    summary = {
        "component_count": len(component_scores),
        "total_quantity": total_qty,
        "summary_score": round(
            sum(item.score * max(item.quantity, 1) for item in component_scores) / total_qty,
            1,
        ),
        "high_priority_recommendations": sum(
            1 for item in component_scores for rec in item.recommendations if rec.priority == "high"
        ),
        "ml_enabled": True,
        "ml_blend_weight": blend_weight,
    }
    return component_scores, summary
