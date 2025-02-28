import torch

background_dist_const = 300 # gt val to indicate the pixel is a background


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def L1(pr, gt, ignore_channels=None, ignore_val=None):
    """Calculate L1 score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
    Returns:
        float: L1 score
    """
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    kept_mask = torch.ones(pr.shape)
    if not ignore_val:
        kept_mask = ~torch.eq(gt, ignore_val)
        pr = torch.mul(pr, kept_mask)
        gt = torch.mul(gt, kept_mask)

    abs_diff = torch.abs(torch.sub(pr, gt))
    score = torch.sum(abs_diff, dtype=pr.dtype) / torch.sum(kept_mask, dtype=pr.dtype)
    return score


def L1_object(pr, gt, ignore_channels=None):
    """Calculate L1 score between ground truth and prediction only on object pixels
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
    Returns:
        float: L1_object score
    """
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    object_mask = ~torch.eq(gt, background_dist_const)
    pr = torch.mul(pr, object_mask)
    gt = torch.mul(gt, object_mask)
    abs_diff = torch.abs(torch.sub(pr, gt))
    return torch.sum(abs_diff, (0, 2, 3)) / torch.sum(object_mask, (0, 2, 3))


def L1_background(pr, gt, ignore_channels=None):
    """Calculate L1 score between ground truth and prediction only on background pixels
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
    Returns:
        float: L1_background score
    """
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    background_mask = torch.eq(gt, background_dist_const)
    pr = torch.mul(pr, background_mask)
    gt = torch.mul(gt, background_mask)
    abs_diff = torch.abs(torch.sub(pr, gt))
    return torch.sum(abs_diff, (0, 2, 3)) / torch.sum(background_mask, (0, 2, 3))


def L2(pr, gt, ignore_channels=None):
    """Calculate L2 score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
    Returns:
        float: L2 score
    """
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    sqr_diff = torch.square(torch.sub(pr, gt))
    score = torch.sum(sqr_diff, dtype=pr.dtype)
    return score


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score
