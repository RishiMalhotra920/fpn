from fpn.models import FPN, RPN


def test_rpn_outputs(image_224_by_224, anchor_scales, anchor_ratios):
    fpn = FPN()
    rpn = RPN(in_channels=256, anchor_scales=anchor_scales, anchor_ratios=anchor_ratios)
    # c, h, w
    fpn_output = fpn(image_224_by_224.unsqueeze(0))

    fpn_map1 = fpn_output[0]  # 256, 200, 200

    rpn_output = rpn(fpn_map1)
    bbox_pred, cls = rpn_output

    assert bbox_pred.shape[1] == len(anchor_scales) * len(anchor_ratios)  # bbox_pred
    assert cls.shape[1] == len(anchor_scales) * len(anchor_ratios) * 4  # cls_pred
