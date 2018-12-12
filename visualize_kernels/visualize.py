import torch, sys
import matplotlib.pyplot as plt
import numpy as np

# REQUIRES COMMIT #4f126149daf26f511bea192cc1c2259c2e0b6b07 from `harmonic`
import harmonic.d2 as d2

def show_serie_color(serie, axes):
    serie = serie.transpose(0, 2, 3, 1)
    serie -= serie.min()
    serie /= serie.max()

    for image, ax in zip(serie, axes):
        ax.imshow(image)

def show_serie_single(serie, axes):
    serie = serie.transpose(0, 2, 3, 1)
    red = serie[..., 0]

    for image, ax in zip(red, axes):
        ax.imshow(image, vmin=red.min(), vmax=red.max())

SHOW_MODE = show_serie_single

# get two samples of each order: 0, 1, 2, 3
interesting_filters = np.array([0, 1, 5, 6, 10, 11, 15, 16])
fig, (b_axes, h_axes, r_axes) = plt.subplots(3, len(interesting_filters))

# BASELINE
baseline = torch.load('baseline_24.pth.tar', map_location='cpu')
baseline_kernel = baseline['model']['path_down.0.conv1.weight'].numpy()
baseline_kernels_i = baseline_kernel[interesting_filters]

SHOW_MODE(baseline_kernels_i, b_axes)

# RELAXED
relaxed = torch.load('relaxed_24.pth.tar', map_location='cpu')
relaxed_kernel = relaxed['model']['path_down.0.conv1.conv.kernel'].numpy()
relaxed_kernels_i = relaxed_kernel[0, interesting_filters]

SHOW_MODE(relaxed_kernels_i, r_axes)

# HARMONIC
harmonic = torch.load('harmonic_24.pth.tar', map_location='cpu')['model']
prefix = 'path_down.0.conv1'
conv_kernels = {k[len(prefix)+1:]: v for k, v in harmonic.items() if prefix in k}
conv = d2.HConv2d((3, ), (5, 5, 5, 5), 5, radius=2)
conv.load_state_dict(conv_kernels)
kernels = conv.conv.synthesize().detach().numpy()
harmonic_kernels_i = kernels[0, interesting_filters]

SHOW_MODE(harmonic_kernels_i, h_axes)

plt.show()
