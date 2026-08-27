"""Adapter contract. One module per vendor; all of them emit canonical quotes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from schemas import OptionQuote


@dataclass
class NormalizeResult:
    """Quotes that parsed, plus every row that did not and why.

    Rejects are returned, not printed. A silent `print()` in a loop over a
    million rows is indistinguishable from a clean load when it scrolls past.
    """
    quotes: list = field(default_factory=list)
    rejects: list = field(default_factory=list)   # (reason, row)

    @property
    def reject_reasons(self) -> dict:
        out: dict = {}
        for reason, _ in self.rejects:
            out[reason] = out.get(reason, 0) + 1
        return out


class OptionChainAdapter(ABC):
    """Normalizes a source dataset into canonical OptionQuote records."""

    @abstractmethod
    def load(self, source) -> Iterable[OptionQuote]:
        ...
