from .constant import Constant as Constant, EMConstant as EMConstant
from _typeshed import Incomplete

class CODATA2010(Constant):
    default_reference: str
    _registry: Incomplete
    _has_incompatible_units: Incomplete
    def __new__(cls, abbrev, name, value, unit, uncertainty, reference=..., system: Incomplete | None = None): ...

class EMCODATA2010(CODATA2010, EMConstant):
    _registry: Incomplete

h: Incomplete
hbar: Incomplete
k_B: Incomplete
c: Incomplete
G: Incomplete
g0: Incomplete
m_p: Incomplete
m_n: Incomplete
m_e: Incomplete
u: Incomplete
sigma_sb: Incomplete
e: Incomplete
eps0: Incomplete
N_A: Incomplete
R: Incomplete
Ryd: Incomplete
a0: Incomplete
muB: Incomplete
alpha: Incomplete
atm: Incomplete
mu0: Incomplete
sigma_T: Incomplete
b_wien: Incomplete
e_esu: Incomplete
e_emu: Incomplete
e_gauss: Incomplete
