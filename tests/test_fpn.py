from fpn.models import FPN


def test_fpn_outputs(image_224_by_224):
    fpn = FPN()
    # c, h, w
    fpn_output = fpn(image_224_by_224.unsqueeze(0))

    M = 200

    assert len(fpn_output) == 4
    assert fpn_output[0].shape == (1, 256, M, M)
    assert fpn_output[1].shape == (1, 256, M / 2, M / 2)
    assert fpn_output[2].shape == (1, 256, M / 4, M / 4)
    assert fpn_output[3].shape == (1, 256, M / 8, M / 8)
