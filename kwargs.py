from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
import datetime
import colour
from utils.logger_ import match_keywords, log_

__all__ = ['Config']

rand_int = 42

@dataclass
class FFCOptions:
    model_path: str = "best_models/PD_trained_512_dauntless-sweep-1/weights/best.pt" # put path to model here
    manual_crop: bool = False # True if you want to crop the image manually else use yolo model to autocrop
    smooth_window: int = 5 # smoothing to apply to image
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    bins: int = 75
    degree: int = 5
    interactions: bool = True
    fit_method: str = "svm" # linear, nn, pls, svm
    max_iter: int = 20000
    tol: float = 1e-12
    verbose: bool = False
    random_seed: int = rand_int
    get_deltaE: bool = True
    show: bool = False


@dataclass
class SaturationOptions:
    check_saturation: bool = True


@dataclass
class GammaCorrectionOptions:
    max_degree: int = 8
    get_deltaE: bool = True
    show: bool = False


@dataclass
class WhiteBalanceOptions:
    get_deltaE: bool = True
    show: bool = False


@dataclass
class ConventionalCCOptions:
    method: str = "Finlayson 2015" #"Cheung 2004", "Finlayson 2015", "Vandermonde"
    degree: int = 1 # 1, 2, 3, 4 only for Vandermonde, Finlayson 2015
    root_polynomial_expansion: Optional[bool] = True # True, False # number of terms depend on the degree
    terms: Optional[int] = None # 3, 4, 5, 7, 8, 10, 14, ... only for Cheung 2004 method
    get_deltaE: bool = True
    show: bool = False

# for linear method
@dataclass
class OurCCOptions:

    # General options
    degree: int = 2
    verbose: bool = False
    random_state: int = rand_int
    n_samples: int = 30
    tol: float = 1e-15
    get_deltaE: bool = True
    show: bool = False

    # # for linear regression
    # mtd: str = "linear"

    # for nn (perceptron network from sklearn) regression
    mtd: str = "nn" # linear, nn, pls, custom
    max_iterations: int = 9000 
    nlayers: int = 110 # depth of the network for nn
    param_search: bool = False # only for pls and nn

    # for pls regression
    #mtd: str = "pls" # linear, nn, pls, custom
    #max_iterations: int = 20000 
    #ncomp: int = -1
    #param_search: bool = False

    # # for custom neural network regression
    # mtd: str = "custom" # linear, nn, pls, custom
    # max_iterations: int = 5000  # cooresponds to epochs 
    # hidden_layers: list = field(default_factory=lambda: [128,32]) #[128] [64, 32, 16] # [128, 64, 32, 16]
    # batchsize: int = 32
    # learning_rate: float = 0.0015
    # patience: int = 1000
    # dropout_rate: float = 0.0
    # use_batch_norm: bool = True
    # optim_type: str = "Adam" # SGD, Adam, RMSprop etc.


@dataclass
class CCOptions:
    pass



@dataclass
class Config:

    # Steps (You can change these in the main code)
    do_ffc: bool = True
    do_gc: bool = True
    do_wb: bool = True
    do_cc: bool = True
    
    # Settings (You can change these in the main code)
    save: bool = True # True if you want to save the results (DeltaEs, Metrics_summary, Models)
    save_path: str = field(default_factory=lambda: f'Results/{datetime.datetime.now().strftime("%Y_%m_%d_T%H_%M_%S")}/')
    cc_method: str = "conv" # 'convetional', 'ours' # which color correction method to use 
    WP: str = "D65" # default white point
    CMFS: str = "CIE 1931 2 Degree Standard Observer"
    REF_ILLUMINANT: Union[None, np.ndarray] = None

    FFC_kwargs: FFCOptions = field(default_factory=FFCOptions)
    Saturation_kwargs: SaturationOptions = field(default_factory=SaturationOptions)
    GC_kwargs: GammaCorrectionOptions = field(default_factory=GammaCorrectionOptions)
    WB_kwargs: WhiteBalanceOptions = field(default_factory=WhiteBalanceOptions)
    OurCC_kwargs: OurCCOptions = field(default_factory=OurCCOptions)
    ConvCC_Kwargs: ConventionalCCOptions = field(default_factory=ConventionalCCOptions)
    CC_kwargs: CCOptions = field(default_factory=CCOptions)


    # update Config object if cc_method is changed
    def __post_init__(self):

        self.cc_method = match_keywords(self.cc_method, ['conv', 'ours'])
        if self.cc_method == 'ours':
            self.CC_kwargs = self.OurCC_kwargs
        elif self.cc_method == 'conv':
            self.CC_kwargs = self.ConvCC_Kwargs
        else:
            raise ValueError("cc_method must be 'ours' or 'conv'")
        
        # log_(f"Using '{self.cc_method}' color correction method", 'green', 'italic', 'info')
        
        self.REF_ILLUMINANT = colour.CCS_ILLUMINANTS[self.CMFS][self.WP]

    def update(self, **kwargs):

        # if kwargs is not empty:
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(f"Config has no attribute '{key}'")

        self.__post_init__()

        log_(f'Updated config {("with " + kwargs.keys().__str__()) if len(kwargs) > 0 else ""}', 'green', 'italic', 'info')



# example usage
# from kwargs_data import Config

# config = Config()

# config.update(
#     save=True,
#     save_path='Results/Test',
#     cc_method='ours',
#     WP='D65',
#     CMFS='CIE 1931 2 Degree Standard Observer',
# )

# print(config.REF_ILLUMINANT)
# print(config.CC_kwargs)

# config.CC_kwargs.nlayers = 200

# print(config.CC_kwargs)