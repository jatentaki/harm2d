import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import matplotlib.gridspec as gridspec

scale = 0
id = 2
img_ix = 0

img = np.load('img.npy')
fmap_pre = np.load('fmap_pre_up_{}.npy'.format(id))
gates = np.load('gates_up_{}_gate2.npy'.format(id))
fmap_post = np.load('fmap_post_up_{}.npy'.format(id))

def plot_quiver(ax, map, img, scale_=300):
    resx, resy = map.shape[1], map.shape[2]
    max_x = resx * 2 ** scale
    max_y = resy * 2 ** scale
    diff_x, diff_y = img.shape[1] - max_x, img.shape[2] - max_y

    x_lo, y_lo = diff_x / 2, diff_y / 2
    x_hi, y_hi = x_lo + resx, y_lo + resy 

    X, Y = np.meshgrid(
        np.linspace(x_lo, x_hi, resx),
        np.linspace(y_lo, y_hi, resy)
    )

    s = (50 / resx)
    fmap = ndi.zoom(map, (1., s, s))
    X = ndi.zoom(X, (s, s))
    Y = ndi.zoom(Y, (s, s))
    
    ax.set_aspect('equal')
    ax.imshow(img.transpose(1, 2, 0), origin='lower')
    q = ax.quiver(X, Y, fmap[0], fmap[1], scale=scale_)


assert fmap_pre.shape == fmap_post.shape

n_fmaps = fmap_pre.shape[2]

for feature in range(n_fmaps):
    m_fig, (pre_ax, post_ax, gate_ax) = plt.subplots(1, 3, figsize=(7, 7))
    m_fig.canvas.set_window_title('Feature map {}/{}'.format(feature, n_fmaps))
    pre_ax.set_title('pre')
    post_ax.set_title('post')

    plot_quiver(pre_ax, fmap_pre[:, img_ix, feature], img[img_ix])
    plot_quiver(post_ax, fmap_post[:, img_ix, feature], img[img_ix])
    gate_ax.imshow(gates[img_ix, feature], vmin=0, vmax=1, origin='lower')
    gate_ax.set_title('gates')
    
#    g_ax.imshow(gates[img_ix, feature], vmin=0, vmax=1, origin='lower')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
#    m_fig.savefig('{}.png'.format(feature), bbox_inches='tight')
    plt.show()
