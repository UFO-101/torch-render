# %%
import torch as t
from IPython.display import display
from jaxtyping import Float32
from torch import Tensor
from torchvision.transforms import ToPILImage


def add(
    a: Float32[Tensor, "batch embed"], b: Float32[Tensor, "batch embed"]
) -> Float32[Tensor, "batch embed"]:
    return a + b


a = t.tensor([1.0, 2.0, 3.0])
b = t.tensor([[4.0, 5.0, 6.0]])
print(add(a, b))


def show_tensor_image_pil(img_tensor):
    """
    Display a PyTorch tensor as an image using PIL in a Jupyter notebook.

    Parameters:
    - tensor: PyTorch tensor of shape (C, H, W)
    """
    # Ensure tensor is detached and on CPU
    img_tensor = img_tensor.detach().cpu()

    # Convert tensor to PIL image
    to_pil = ToPILImage()
    pil_image = to_pil(img_tensor)

    # Display the image
    display(pil_image)


# Example usage
tensor = t.rand(3, 100, 100)  # Random RGB image
show_tensor_image_pil(tensor)
