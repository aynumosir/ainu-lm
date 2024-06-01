import numpy as np
from numpy.typing import NDArray


def leven(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row: NDArray[np.int_] = np.arange(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row: NDArray[np.int_] = np.zeros(len(s2) + 1, dtype=int)
        current_row[0] = i + 1

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)

        previous_row = current_row

    return int(previous_row[-1])
