from dash import Dash, html, dcc
from CalSciPy.optics.psf import PSF
from plotly.express import imshow
import matplotlib
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np


psf = PSF(np.load("C:\\Users\\Darik\\psf.npy"), scaling=(0.5, 0.1, 0.1))

get_cmap = matplotlib.colormaps.get_cmap
cm = get_cmap("coolwarm")

images = cm(Normalize(*psf.range)(psf.denoised[:, :, :]))[:, :, :, :3]

app = Dash(__name__)

fig = imshow(images, animation_frame=0, title="BOOGERS")


fig.show()

app.layout = html.Div(
    children=[
        html.H1(children="Point Spread Function"),
        html.P(
            children=(
                "Interactive Analysis of PSF"
            )
        ),
        dcc.Graph(
            figure=fig
        ),
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True)
