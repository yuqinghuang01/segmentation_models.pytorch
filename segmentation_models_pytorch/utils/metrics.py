from . import base
from . import functional as F
from ..base.modules import Activation


class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class L1Score_left_object(base.Metric):
    __name__ = 'L1_left_object'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_object(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[0]


class L1Score_top_object(base.Metric):
    __name__ = 'L1_top_object'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_object(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[1]


class L1Score_right_object(base.Metric):
    __name__ = 'L1_right_object'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_object(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[2]


class L1Score_bot_object(base.Metric):
    __name__ = 'L1_bot_object'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_object(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[3]


class L1Score_left_background(base.Metric):
    __name__ = 'L1_left_background'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_background(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[0]


class L1Score_top_background(base.Metric):
    __name__ = 'L1_top_background'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_background(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[1]


class L1Score_right_background(base.Metric):
    __name__ = 'L1_right_background'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_background(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[2]


class L1Score_bot_background(base.Metric):
    __name__ = 'L1_bot_background'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L1_background(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )[3]


class L2Score(base.Metric):
    __name__ = 'L2_score'
    
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.L2(
            y_pr, y_gt,
            ignore_channels=self.ignore_channels,
        )
