import torch
import random
import numpy as np


class GeneratedImages:
    def __init__(self, max_size=50):
        self._buffer = None
        self.max_size = max_size

    def update(self, generated_image: torch.Tensor):
        if self._buffer is None:
            self._buffer = generated_image.detach()
        else:
            if len(self._buffer) < self.max_size:
                self._buffer = torch.concat((self._buffer, generated_image.detach()))
            else:
                if random.randint(0, 10) < 5:
                    self._buffer = torch.concat((self._buffer, generated_image.detach()))[1:self.max_size + 1]

    @property
    def buffer(self):
        shuffled_indexes = torch.from_numpy(
            np.random.choice(self._buffer.shape[0], size=self._buffer.shape[0], replace=False)
        )
        return self._buffer[shuffled_indexes]
