from dataclasses import dataclass  # Python 3.6 github.com/ericvsmith/dataclasses, for python 3.7 delete this import
import numpy as np


@dataclass
class AvgCounter:
    total_value: float = 0
    n: int = 0

    def add(self, val: np.float32):
        if not np.isnan(val):
            self.total_value += val
            self.n += 1

    @property
    def average(self) -> float:
        if self.n != 0:
            return self.total_value / self.n
        raise Exception("No data, add data beforehand")

    def __str__(self):
        return self.average
