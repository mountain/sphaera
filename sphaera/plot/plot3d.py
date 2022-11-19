import numpy as np
import pyvista as pv


sphere = pv.Sphere(
    direction=(0, 0, 1),
    start_theta=0, end_theta=360, theta_resolution=1440,
    start_phi=0.001, end_phi=179.999, phi_resolution=721,
    radius=6371000.0)


def plot_scalar(varname, scl):
    f = scl[0, 0, :, :, 0].numpy()[::-1, ::-1].T.reshape(721 * 1440, 1)
    sphere[varname] = f
    sphere.set_active_scalars(varname)
    sphere.plot(cpos='xz', cmap='plasma')


def plot_vector(varname, vec):
    a, b, c = vec
    a, b, c = a[0, 0, :, :, 0].numpy(), b[0, 0, :, :, 0].numpy(), c[0, 0, :, :, 0].numpy()
    vectors = np.concatenate(
        (
            a[::-1, ::-1].T.reshape(721 * 1440, 1),
            b[::-1, ::-1].T.reshape(721 * 1440, 1),
            c[::-1, ::-1].T.reshape(721 * 1440, 1),
        ), axis=-1
    )
    sphere[varname] = vectors
    sphere.set_active_vectors(varname)
    plt = pv.Plotter()
    plt.add_mesh(sphere.arrows, lighting=False, scalar_bar_args={'title': "x"})
    plt.add_mesh(sphere, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    plt.show()
