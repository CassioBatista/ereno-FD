"""GenericResultado equivalent — holds evaluation metrics for a single classifier run."""
from __future__ import annotations
import python.config as config


class Result:
    """Stores VP/VN/FP/FN and derived metrics for one classifier evaluation."""

    def __init__(
        self,
        classifier_name: str,
        VP: float,
        FN: float,
        VN: float,
        FP: float,
        avg_time: float = 0.0,
        confusion_matrix: list[list[int]] | None = None,
    ):
        self.cx = classifier_name
        self.VP = VP
        self.FN = FN
        self.VN = VN
        self.FP = FP
        self.avg_time = avg_time
        self.nanotime: float = avg_time
        self.confusion_matrix: list[list[int]] = confusion_matrix or []
        self.used_fs: list[int] = list(config.FEATURE_SELECTION)

        # Derived metrics
        total = VP + VN + FP + FN
        self.accuracy: float = (VP + VN) / total * 100 if total else 0.0
        self.precision: float = VP / (VP + FP) * 100 if (VP + FP) else 0.0
        self.recall: float = VP / (VP + FN) * 100 if (VP + FN) else 0.0
        denom = self.precision + self.recall
        self.f1score: float = (2 * self.precision * self.recall / denom) if denom else 0.0

    # ------------------------------------------------------------------
    # Property aliases matching Java getters
    # ------------------------------------------------------------------
    def get_accuracy(self) -> float:
        return self.accuracy

    def get_precision(self) -> float:
        return self.precision

    def get_recall(self) -> float:
        return self.recall

    def get_f1score(self) -> float:
        return self.f1score

    def get_confusion_matrix(self) -> list[list[int]]:
        return self.confusion_matrix

    def get_nanotime(self) -> float:
        return self.nanotime

    def get_classifier_name(self) -> str:
        return self.cx

    # average of fold results
    @staticmethod
    def average(results: list["Result"]) -> "Result":
        if not results:
            raise ValueError("Empty result list")
        vp = sum(r.VP for r in results) / len(results)
        fn = sum(r.FN for r in results) / len(results)
        vn = sum(r.VN for r in results) / len(results)
        fp = sum(r.FP for r in results) / len(results)
        t = sum(r.avg_time for r in results) / len(results)
        return Result(results[0].cx, vp, fn, vn, fp, t)
