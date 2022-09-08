import torch
import random


class GeneratedImages:
    def __int__(self, max_size=50):
        self.buffer = None
        self.max_size = max_size

    def update(self, generated_image: torch.Tensor):
        if self.buffer is None:
            self.buffer = generated_image.detach()
        else:
            if len(self.buffer) < self.max_size:
                self.buffer = torch.concat((self.buffer, generated_image.detach()))
            else:
                if random.randint(0, 10) < 5:
                    self.buffer = torch.concat((self.buffer, generated_image.detach()))[1:self.max_size + 1]
