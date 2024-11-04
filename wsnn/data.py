from dataclasses import dataclass, field
from typing import Sequence

def create_one_hot(index: int, num_labels: int) -> list[float]:
    # Creates a one-hot encoded vector with a 1 at the specified index.
    one_hot = [0.0] * num_labels
    one_hot[index] = 1.0
    return one_hot

@dataclass
class DataPoint:
    inputs: Sequence[float]
    label: int
    num_labels: int
    expected_outputs: Sequence[int] = field(default_factory=list)

    def __post_init__(self):
        # Initializes one-hot encoded expected outputs if not provided.
        if not self.expected_outputs:
            self.expected_outputs = create_one_hot(self.label, self.num_labels)
