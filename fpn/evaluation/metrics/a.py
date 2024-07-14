import numpy as np


def interpolated_precision_recall(recalls, precisions):
    # Sort by recall
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]

    # Interpolate precision
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    return recalls, precisions


# Example data
recalls = np.array([0.1, 0.2, 0.3, 0.4])
precisions = np.array([1.0, 0.8, 0.9, 0.7])

int_recalls, int_precisions = interpolated_precision_recall(recalls, precisions)

print("Original P-R pairs:")
for r, p in zip(recalls, precisions):
    print(f"Recall: {r:.1f}, Precision: {p:.1f}")

print("\nInterpolated P-R pairs:")
for r, p in zip(int_recalls, int_precisions):
    print(f"Recall: {r:.1f}, Precision: {p:.1f}")
