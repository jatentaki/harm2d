import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import matplotlib.gridspec as gridspec

scale = 0
id = 2

img = np.load('img.npy')
fmap = np.load('fmap_up_{}.npy'.format(id))
kernels = np.load('kernels_up_{}.npy'.format(id))

n_kernels = kernels.shape[2]
n_kernels_root = int(math.ceil(n_kernels ** 0.5))

resx, resy = fmap.shape[3], fmap.shape[4]
max_x = resx * 2 ** scale
max_y = resy * 2 ** scale
diff_x, diff_y = img.shape[2] - max_x, img.shape[3] - max_y

X, Y = np.meshgrid(
    np.linspace(diff_x / 2, diff_x / 2 + max_x, resx),
    np.linspace(diff_y / 2, diff_y / 2 + max_y, resy)
)

img_ix = 2
s = (50 / resx)
fmap = ndi.zoom(fmap, (1., 1., 1., s, s))
X = ndi.zoom(X, (s, s))
Y = ndi.zoom(Y, (s, s))

for feature in range(fmap.shape[2]):
    m_fig, m_ax = plt.subplots(figsize=(7, 7))
    m_fig.canvas.set_window_title('Feature map {}/{}'.format(feature, fmap.shape[2]))
    m_ax.set_aspect('equal')
    m_ax.imshow(img[img_ix].transpose(1, 2, 0), origin='lower')
    m_ax.quiver(X, Y, fmap[0, img_ix, feature], fmap[1, img_ix, feature])
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
#    m_fig.savefig('{}.png'.format(feature), bbox_inches='tight')
    plt.show()
