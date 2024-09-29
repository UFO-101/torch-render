# %%
# import numpy as np
import torch as t
from jaxtyping import Float32
from torch import Tensor

from torch_render.render_2d import add

a = t.tensor([1.0, 2.0, 3.0])
b = t.tensor([[4.0, 5.0, 6.0]])
print(add(a, b))
