from .kqi import KQI
from .branch import Branch, SimplePass

from .linear import Linear
# from .linear import Identity, Linear, Bilinear, LazyLinear
from .conv import Conv2d, Conv3d
# from .conv import Conv1d, Conv2d, Conv3d, \
#     ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, \
#     LazyConv1d, LazyConv2d, LazyConv3d, LazyConvTranspose1d, LazyConvTranspose2d, LazyConvTranspose3d
from .activation import ReLU, Tanh, SoftMax
# from .activation import Threshold, ReLU, Hardtanh, ReLU6, Sigmoid, Tanh, \
#     Softmax, Softmax2d, LogSoftmax, ELU, SELU, CELU, GELU, Hardshrink, LeakyReLU, LogSigmoid, \
#     Softplus, Softshrink, MultiheadAttention, PReLU, Softsign, Softmin, Tanhshrink, RReLU, GLU, \
#     Hardsigmoid, Hardswish, SiLU, Mish
# from .loss import L1Loss, NLLLoss, KLDivLoss, MSELoss, BCELoss, BCEWithLogitsLoss, NLLLoss2d, \
#     CosineEmbeddingLoss, CTCLoss, HingeEmbeddingLoss, MarginRankingLoss, \
#     MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss, SmoothL1Loss, HuberLoss, \
#     SoftMarginLoss, CrossEntropyLoss, TripletMarginLoss, TripletMarginWithDistanceLoss, PoissonNLLLoss, GaussianNLLLoss
from .container import Sequential
# from .container import Container, Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict
from .pooling import MaxPool2d
# from .pooling import AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d, \
#     MaxUnpool1d, MaxUnpool2d, MaxUnpool3d, FractionalMaxPool2d, FractionalMaxPool3d, LPPool1d, LPPool2d, \
#     AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d
# from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm, \
#     LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d
# from .instancenorm import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, \
#     LazyInstanceNorm1d, LazyInstanceNorm2d, LazyInstanceNorm3d
# from .normalization import LocalResponseNorm, CrossMapLRN2d, LayerNorm, GroupNorm
from .dropout import Dropout
# from .dropout import Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout, FeatureAlphaDropout
# from .padding import ReflectionPad1d, ReflectionPad2d, ReflectionPad3d, ReplicationPad1d, ReplicationPad2d, \
#     ReplicationPad3d, ZeroPad2d, ConstantPad1d, ConstantPad2d, ConstantPad3d
# from .sparse import Embedding, EmbeddingBag
from .rnn import RNN
# from .rnn import RNNBase, RNN, LSTM, GRU, \
#     RNNCellBase, RNNCell, LSTMCell, GRUCell
# from .pixelshuffle import PixelShuffle, PixelUnshuffle
# from .upsampling import UpsamplingNearest2d, UpsamplingBilinear2d, Upsample
# from .distance import PairwiseDistance, CosineSimilarity
# from .fold import Fold, Unfold
# from .adaptive import AdaptiveLogSoftmaxWithLoss
# from .transformer import TransformerEncoder, TransformerDecoder, \
#     TransformerEncoderLayer, TransformerDecoderLayer, Transformer
# from .flatten import Flatten, Unflatten
# from .channelshuffle import ChannelShuffle