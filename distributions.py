import numpy as np
import matplotlib.pyplot as plt

id = 0
img_ix = 0

img = np.load('img.npy')
fmaps = np.load('fmap_up_{}.npy'.format(id))

fig, axes = plt.subplots(fmaps.shape[2] // 4, 4)

def gaussian(xs, mu, sigma):
    norm = 1 / np.sqrt(2 * np.pi * sigma **2)
    exp = np.exp(-(xs - mu) ** 2 / (2 * sigma ** 2))
    return exp#norm * exp

def rayleigh(xs, sigma_sqr):
    return xs / sigma_sqr * np.exp(-(xs**2) / (2 * sigma_sqr))

for i, ax in enumerate(axes.flatten()):
    fmap = fmaps[:, :, i, ...]
    
    hist_c, edges_c = np.histogram(fmap, bins=100)
    mids_c = (edges_c[:-1] + edges_c[1:]) / 2

    ax.plot(mids_c, hist_c / 2, 'b-')

    mean = fmap.mean()
    std = fmap.std()
    peak_c = hist_c.max()
    fit = peak_c * gaussian(mids_c, mean, std)
    ax.plot(mids_c, fit / 2, 'b--')

    magnitudes_sqr = fmap[0] ** 2 + fmap[1] ** 2
    magnitudes = np.sqrt(magnitudes_sqr)
    hist_m, edges_m = np.histogram(magnitudes, bins=100)
    mids_m = (edges_m[:-1] + edges_m[1:]) / 2

    ax.plot(mids_m, hist_m, 'r-')

    samples = np.random.randn(2, magnitudes.size) * std + mean
    magnitudes_s = np.hypot(samples[0], samples[1])
    hist_s, edges_s = np.histogram(magnitudes_s, bins=100)
    mids_s = (edges_s[:-1] + edges_s[1:]) / 2
    
    ax.plot(mids_s, hist_s, 'g--')


plt.show()
