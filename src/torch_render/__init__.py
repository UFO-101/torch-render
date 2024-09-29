from beartype.claw import beartype_this_package  # <-- hype comes
from jaxtyping import install_import_hook

beartype_this_package()  # <-- hype goes

with install_import_hook("torch_render", "beartype.beartype"):
    import torch_render.render_2d  # noqa
