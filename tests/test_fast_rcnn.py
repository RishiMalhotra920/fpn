import torch

from fpn.models.fast_rcnn_classifier import FastRCNNClassifier


def test_fast_rcnn_outputs():
    fast_rcnn = FastRCNNClassifier(num_classes=10)
    # c, h, w

    tensor_224_by_224 = torch.randn(3, 256, 224, 224)
    cls, bbox = fast_rcnn(tensor_224_by_224)

    assert cls.shape == (3, 10)
    assert bbox.shape == (3, 40)
