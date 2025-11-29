import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# ============================
# Helper: ensure mask is on correct device
# ============================
def to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor


# ============================
# MCAR mask generator
# ============================
class MCARGenerator:
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        device = batch.device
        nan_mask = torch.isnan(batch).float().to(device)
        bernoulli = torch.from_numpy(
            np.random.choice(2, size=batch.shape, p=[1-self.p, self.p])
        ).float().to(device)
        return torch.max(bernoulli, nan_mask)


# ============================
# ImageMCAR generator
# ============================
class ImageMCARGenerator:
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        device = batch.device
        B, C, H, W = batch.shape

        bern = torch.from_numpy(
            np.random.choice(2, size=(B,1,H,W), p=[1-self.p, self.p])
        ).float().to(device)

        return bern.repeat(1, C, 1, 1)


# ============================
# Fixed rectangle mask
# ============================
class FixedRectangleGenerator:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, batch):
        mask = torch.zeros_like(batch, device=batch.device)
        mask[:, :, self.x1:self.x2, self.y1:self.y2] = 1
        return mask


# ============================
# Random rectangle mask
# ============================
class RectangleGenerator:
    def __init__(self, min_rect_rel_square=0.3, max_rect_rel_square=1):
        self.min_r = min_rect_rel_square
        self.max_r = max_rect_rel_square

    def gen_coords(self, width, height):
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = np.random.randint(0, height, 2)
        return min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)

    def __call__(self, batch):
        device = batch.device
        B, C, H, W = batch.shape
        mask = torch.zeros_like(batch, device=device)

        for i in range(B):
            x1, y1, x2, y2 = self.gen_coords(H, W)
            area = H*W
            while not (self.min_r * area <= (x2-x1+1)*(y2-y1+1) <= self.max_r*area):
                x1, y1, x2, y2 = self.gen_coords(H, W)
            mask[i,:,x1:x2+1,y1:y2+1] = 1

        return mask


# ============================
# Random Pattern Mask (heavy)
# ============================
class RandomPattern:
    def __init__(self, max_size=10000, resolution=0.06,
                 density=0.25, update_freq=1, seed=239):
        self.max_size = max_size
        self.resolution = resolution
        self.density = density
        self.update_freq = update_freq
        self.rng = np.random.RandomState(seed)
        self.regenerate_cache()

    def regenerate_cache(self):
        low_size = int(self.resolution * self.max_size)
        low_map = (self.rng.uniform(0,1,(low_size,low_size))*255).astype("float32")
        low_t = torch.from_numpy(low_map)[None]   # shape 1xH×W
        pattern = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.max_size, Image.BICUBIC),
            transforms.ToTensor(),
        ])(low_t)[0]

        self.pattern = (pattern < self.density).byte()
        self.points_used = 0

    def __call__(self, batch, density_std=0.05):
        device = batch.device
        B,C,H,W = batch.shape

        res = torch.zeros((B,1,H,W), dtype=torch.uint8)

        idx = list(range(B))
        while idx:
            nw_idx=[]
            xs = self.rng.randint(0, self.max_size-H+1, size=len(idx))
            ys = self.rng.randint(0, self.max_size-W+1, size=len(idx))
            for i,x,y in zip(idx,xs,ys):
                crop = self.pattern[x:x+H, y:y+W][None]
                cov = float(crop.float().mean())
                if (self.density-density_std) < cov < (self.density+density_std):
                    res[i]=crop
                else:
                    nw_idx.append(i)
            idx=nw_idx

        res = res.repeat(1,C,1,1)   # 1→C
        return res.float().to(device)


# ============================
# Mixture of generators
# ============================
class MixtureMaskGenerator:
    def __init__(self, generators, weights):
        self.generators = generators
        self.weights = weights

    def __call__(self, batch):
        device = batch.device
        B = batch.shape[0]

        w = np.array(self.weights, dtype="float32")
        w /= w.sum()
        choices = np.random.choice(len(w), B, True, w)

        mask = torch.zeros_like(batch, device=device)

        for i, gen in enumerate(self.generators):
            ids = np.where(choices == i)[0]
            if len(ids)==0:
                continue
            out = gen(batch[ids])       # might be CPU
            mask[ids] = out.to(device)  # FORCE GPU

        return mask


# ============================
# Final mask generator
# ============================
class ImageMaskGenerator:
    def __init__(self):
        siidgm = SIIDGMGenerator()
        gfc = GFCGenerator()
        common = RectangleGenerator()
        self.generator = MixtureMaskGenerator([siidgm, gfc, common], [1,1,2])

    def __call__(self, batch):
        return self.generator(batch)


# Wrapper classes
class GFCGenerator:
    def __init__(self):
        self.generator = MixtureMaskGenerator([
            FixedRectangleGenerator(52,33,116,71),
            FixedRectangleGenerator(52,57,116,95),
            FixedRectangleGenerator(52,29,74,99),
            FixedRectangleGenerator(52,29,74,67),
            FixedRectangleGenerator(52,61,74,99),
            FixedRectangleGenerator(86,40,124,88)
        ], [1]*6)

    def __call__(self, batch):
        return self.generator(batch)


class SIIDGMGenerator:
    def __init__(self):
        rnd = RandomPattern(max_size=10000, resolution=0.03)
        mcar = ImageMCARGenerator(0.95)
        center = FixedRectangleGenerator(32,32,96,96)
        h1 = FixedRectangleGenerator(0,0,128,64)
        h2 = FixedRectangleGenerator(0,0,64,128)
        h3 = FixedRectangleGenerator(0,64,128,128)
        h4 = FixedRectangleGenerator(64,0,128,128)

        self.generator = MixtureMaskGenerator([
            center, rnd, mcar, h1, h2, h3, h4
        ], [2,2,2,1,1,1,1])

    def __call__(self, batch):
        return self.generator(batch)

