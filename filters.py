import torch
from typing import List

def get_directional_filters(normalize=True) -> List[torch.Tensor]:
    base_filters = [
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],  # Sobel X

        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],  # Sobel Y

        [[ 0, -1, 0],
         [-1, 4, -1],
         [ 0, -1, 0]],  # Laplacian

        [[-1, -1, -1],
         [-1,  8, -1],
         [-1, -1, -1]],  # Edge
    ]

    filters = []
    for f in base_filters:
        f_tensor = torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        if normalize:
            f_tensor = f_tensor / torch.norm(f_tensor)
        filters.append(f_tensor)

    return filters  # list of [1,1,3,3] tensors
