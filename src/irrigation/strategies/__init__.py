from .sidss_like import SIDSSRealStub
from .fdss_like import FDSSReal
from .mrgcd_like import MRGCDReal
REAL_STRATEGIES={s.name:s for s in [SIDSSRealStub,FDSSReal,MRGCDReal]}
