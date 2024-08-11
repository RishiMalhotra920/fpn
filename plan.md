"""this class takes in multiple cropped feature maps of arbitrary size, passes them to the RPN, gets the crops and objectness scores from
the RPN, takes certain crops ta passes them to the FastRCNNClassifier, gets the softmax scores and bounding box regression offsets and

    In this experiment, B=256, C=3, H=224, W=224, nF=3,
    nBB=2000 (number of feature maps produced by FPN)
    nBB_pos = 500 (number of positive bounding boxes)
    nBB_neg = 1500 (number of negative bounding boxes)
    F=256 (feature map size) and R=256 (RoI)
    Compute the feature maps with FPN
        1. Incoming batch of images of shape (B, C, H, W)
        2. The batch is passed through the FPN to get three feature maps with the following ratios:
            [(B, F, H, W), (B, F, H/2, W/2), (B, F, H/4, W/4)]
        3. I freeze the pre-trained FPN.
    Pass the feature maps through the RPN
        3. We slide the RPN over each of [(B, F, H, W), (B, F, H/2, W/2), (B, F, H/4, W/4)] feature maps
           and produce [(B, H*W*nA), (B, H/2*W/2*nA), (B, H/4*W/4*nA)] cls scores and
            [(B, H*W*nA*4), (B, H/2*W/2*nA*4), (B, H/4*W/4*nA*4)] bboxes.
        4. We collapse the array and apply bbox regression offsets to the anchor boxes to get the bounding boxes in the image.
        5. Apply NMS here across the collated array per image and prepend batch_feature_map_index to the bounding box
            sample nBB_neg RoIs randomly where IoU(bbox, gt) < 0.3 (There is also a strategy for hard negative mining where we sample some negatives with high IoU to teach the model how to differentiate)
            sample nBB_pos RoIs where IoU(bbox, gt) > 0.5 to get a (num_rois, 5) num_rois <= nBB
            augmented_bbox[i] = np.ndarray([b_idx, x1, y1, x2, y2])

            3.1 we calculate the loss for the rpn:
                L_rpn_cls = cross_entropy(cls_i, cls_i_star)
                L_rpn_bbox = smooth_l1_loss(bbox_i, bbox_i_star)
                3.1.1
                    Translate the image space boxes to (B*nF, F, H, W) cls scores and (B*nF, F, nA*4) bbox regression offsets.
                    When translating bbox, find center of bbox and find the anchor box  aspect ratio that matches it most closely.
                    Then calculate feature_space_box_center/image_space_box_center to find which feature map pixel the box center is in.
                    Add the bbox regression offsets, t_x = (x - x_a) / w_a, t_y = (y - y_a) / h_a, t_w = log(w / w_a), t_h = log(h / h_a)
                    While you're here, also calculate the cli_s label for later use in 7.1.1
    Pass the cropped feature maps to the FastRCNNClassifier
        7. We pass the feature maps and augmented_bboxes to the FastRCNNClassifier
           We receive (num_rois, num_classes) softmax scores and (num_rois, 4) bounding box regression offsets.
            7.1 we calculate the loss for the fast_rcnn:
                L_fast_rcnn_cls = cross_entropy(cls_i, cls_i_star)
                L_fast_rcnn_bbox = smooth_l1_loss(bbox_i, bbox_i_star)
                Propagate the cls_i and bbox_i from 3.1.1 to here.
        8. Calculate bbox offsets compared to the original image.
            8.1 We use the bounding box regression offsets to adjust the bounding box coordinates output by the RPN.
        9. We pick the class with the highest softmax score and use the corresponding bounding box.
    Calculate the total loss
        11. The loss function is as follows:
            L_total = lambda_1 * L_rpn_cls + lambda_2 * L_rpn_bbox + lambda_3 * L_fast_rcnn_cls + lambda_4 * L_fast_rcnn_bbox


    Utility functions that I should code up:
        RPN output visualizer
        FastRCNN classifier output visualizer
        Full network output visualizer

    Other todos:
        Finetuning with optuna:
            Params to finetune: learning_rate, lambda_1, lambda_2, lambda_3, lambda_4, dropout_prob, nms_threshold
        Gradient clipping with max_norm to 1.0 to prevent exploding gradients early on
        Use warmup with cosine annealing learning rate scheduler.
        ---kaiming initialization--- (for another day :) )

    Args:
        Model (_type_): _description_
    """
