import numpy as np # type: ignore
import pylab as pl # type: ignore
import plotly.graph_objects as go # type: ignore

def plot_losses(losses):
    """
    Plot the losses.
    :param losses: List of losses
    :param title: Title of the plot
    """
    pl.figure(figsize = (8,16/3))
    pl.plot(losses)
    pl.xlabel("Epochs")
    pl.ylabel("Loss")
    pl.title("Losses")
    pl.grid()
    pl.show()

def plot_data(X, y, title = "Data"):
    y_unique = np.unique(y)
    colors = pl.cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        pl.scatter(this_X[:, 0], this_X[:, 1],  color=color,
                    alpha=0.5, edgecolor='k',
                    label="Class %s" % this_y)
    pl.legend(loc="best")
    pl.title(title)

def plot_decision_region(X, pred_fun, vmin=-2, vmax=2):
    """
    X: corresponde a las instancias de nuestro conjunto de datos
    pred_fun: es una función que para cada valor de X, me regresa una predicción
    """
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    min_x = min_x - (max_x - min_x) * 0.05
    max_x = max_x + (max_x - min_x) * 0.05
    min_y = min_y - (max_y - min_y) * 0.05
    max_y = max_y + (max_y - min_y) * 0.05
    x_vals = np.linspace(min_x, max_x, 60)
    y_vals = np.linspace(min_y, max_y, 60)
    XX, YY = np.meshgrid(x_vals, y_vals)
    grid_r, grid_c = XX.shape
    ZZ = np.zeros((grid_r, grid_c))
    for i in range(grid_r):
        for j in range(grid_c):
            ZZ[i, j] = pred_fun(XX[i, j], YY[i, j])
    pl.contourf(XX, YY, ZZ, 30, cmap = pl.cm.coolwarm, vmin=vmin, vmax=vmax)
    pl.colorbar()
    pl.xlabel("x")
    pl.ylabel("y")


def plot_decision_region_3d(X, y, pred_fun, resolution=30, threshold=0.5, vmin=0, vmax=1, title="3D Decision Regions"):
    """
    Visualiza regiones de decisión 3D con Volume + Isosurface + puntos de datos + leyendas.
    
    X: ndarray (n_samples, 3)
    y: ndarray (n_samples,)
    pred_fun: función que toma (3,) y retorna una probabilidad o score en [0, 1]
    resolution: número de divisiones por eje
    threshold: valor del umbral de frontera (0.5 típico para clasificación binaria)
    """
    # Rango de valores
    pad = 0.05
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    z_min, z_max = X[:, 2].min(), X[:, 2].max()
    dx, dy, dz = (x_max - x_min) * pad, (y_max - y_min) * pad, (z_max - z_min) * pad

    x = np.linspace(x_min - dx, x_max + dx, resolution)
    y_ = np.linspace(y_min - dy, y_max + dy, resolution)
    z = np.linspace(z_min - dz, z_max + dz, resolution)

    Xg, Yg, Zg = np.meshgrid(x, y_, z, indexing='ij')
    grid_points = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]

    # Calcular predicciones en cada punto
    preds = np.array([pred_fun(p) for p in grid_points]).reshape(Xg.shape)

    # Crear figura
    fig = go.Figure()

    # Volume: regiones coloreadas por score de predicción
    fig.add_trace(go.Volume(
        x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
        value=preds.flatten(),
        opacity=0.12,
        surface_count=15,
        colorscale='RdBu_r',
        showscale=True,
        cmin=vmin,
        cmax=vmax,
        colorbar=dict(title="Score", tickvals=[vmin, threshold, vmax], len=0.5),
        name='Score Volume'
    ))

    # Isosuperficie para la frontera de decisión
    fig.add_trace(go.Isosurface(
        x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
        value=preds.flatten(),
        isomin=threshold,
        isomax=threshold,
        surface_count=1,
        opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False,
        name=f'Frontera @ {threshold}',
        showlegend=True
    ))

    # Puntos de clase 0
    fig.add_trace(go.Scatter3d(
        x=X[y == 0, 0], y=X[y == 0, 1], z=X[y == 0, 2],
        mode='markers',
        marker=dict(size=4, color='blue'),
        name='Clase 0'
    ))

    # Puntos de clase 1
    fig.add_trace(go.Scatter3d(
        x=X[y == 1, 0], y=X[y == 1, 1], z=X[y == 1, 2],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Clase 1'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='X3',
        ),
        title=title,
        legend=dict(x=0.8, y=0.95),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()