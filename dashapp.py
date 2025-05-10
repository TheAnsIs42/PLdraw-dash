from dash import dcc, html, Input, Output, State, Dash, ctx
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
import os
import base64
import io
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import numpy as np

# Constants
PLANCK_CONST = 6.63e-34 / 1.6e-19  # eV energy
SPEED_OF_LIGHT = 3e8
SAVGOL_WINDOW = 150
SAVGOL_POLY = 2
DEFAULT_WIDTH = 520
DEFAULT_HEIGHT = 390

if not os.path.exists("image"):
    os.mkdir("image")

# Initialize the Dash app
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/popper.min.css"
]
app = Dash(__name__, external_stylesheets=external_stylesheets)


@lru_cache(maxsize=32)
def read_data(contents: str, filename: str) -> Optional[pd.DataFrame]:
    """Read and process uploaded data file with caching."""
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        df = pd.read_csv(
            io.StringIO(decoded.decode("utf-8")),
            sep=r"\s+",
            header=None,
            names=["wavelength", "count"],
        )

        # Calculate normalized and energy columns
        df["normalized"] = (df["count"] - df["count"].min()) / (
            df["count"].max() - df["count"].min()
        )
        df["energy"] = PLANCK_CONST * SPEED_OF_LIGHT / (df["wavelength"] * 1e-9)
        return df
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def process_selected_files(
    data_list: List[Dict], selected_files: List[str]
) -> List[Dict]:
    """Process and filter data based on selected files."""
    processed_data = []
    for item in data_list:
        if item["name"] in selected_files:
            item["data"] = pd.DataFrame(item["data"])
            processed_data.append(item)
    return processed_data


def smooth_chart(data_list: List[Dict]) -> go.Figure:
    """Create smoothed chart with peak annotations."""
    if not data_list:
        return go.Figure()

    chart = (
        go.Figure()
        .update_xaxes(title_text="Wavelength(nm)")
        .update_yaxes(title_text="Intensity(arb.u.)")
        .update_layout(legend=dict(x=0, y=0.5), margin=dict(l=50, r=50, t=50, b=50))
    )

    colors = px.colors.qualitative.Plotly
    for idx, data in enumerate(data_list):
        wavelength = data["data"]["wavelength"]
        normalized = data["data"]["normalized"]

        # Calculate smoothed data once
        filtered_data = signal.savgol_filter(normalized, SAVGOL_WINDOW, SAVGOL_POLY)
        peak_idx = filtered_data.argmax()

        chart.add_trace(
            go.Scatter(
                x=wavelength,
                y=filtered_data,
                name=data["name"],
                line_color=colors[idx % len(colors)],
            )
        )

        chart.add_vline(
            x=wavelength[peak_idx],
            line_color=colors[idx % len(colors)],
            annotation=dict(
                text=f"{wavelength[peak_idx]:.1f}nm",
                textangle=90,
            ),
            annotation_position="bottom left",
        )

    chart.write_image(
        "image/smooth_plot43.svg", width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
    )
    return chart


def simple_chart(
    data_list: List[Dict], ev_plot: bool, normalize: bool, log_y: bool = False
) -> go.Figure:
    """Create simple chart with various display options."""
    if not data_list:
        return go.Figure()

    chart = go.Figure().update_layout(
        legend=dict(x=0, y=0.5), margin=dict(l=50, r=50, t=50, b=50)
    )

    data_x = "energy" if ev_plot else "wavelength"
    chart.update_xaxes(
        title_text="Energy (eV)" if ev_plot else "Wavelength (nm)",
        autorange="reversed" if ev_plot else None,
    )

    data_y = "normalized" if normalize else "count"
    chart.update_yaxes(
        title_text=(
            "Normalized intensity (arb.u.)" if normalize else "Intensity (arb.u.)"
        )
    )

    if log_y:
        chart.update_yaxes(type="log", minor=dict(ticks="inside", showgrid=True))

    for data in data_list:
        chart.add_trace(
            go.Scatter(
                x=data["data"][data_x], y=data["data"][data_y], name=data["name"]
            )
        )
    chart.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    chart.write_image(
        "image/simple_plot43.svg", width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT
    )
    return chart


def read_matrixPL(
    filepath: str, savgol_window: int = 150, savgol_poly: int = 2
) -> Tuple[pd.DataFrame, pd.Index, np.ndarray, np.ndarray, pd.Index]:
    """Read and process matrix PL data file."""
    data = pd.read_csv(filepath, sep=r"\s+")
    filt_data = signal.savgol_filter(data, savgol_window, savgol_poly, axis=1)
    peak_idx = filt_data.argmax(1)
    peak_pos = data.columns.to_numpy(dtype=float)[peak_idx]
    return (
        data,
        data.index,
        peak_pos,
        filt_data.max(axis=1),
        data.columns.astype(float),
    )


def create_matrix_plot(
    data: pd.DataFrame,
    index: pd.Index,
    pos: np.ndarray,
    max_intensity: np.ndarray,
    column: pd.Index,
    title: str,
) -> go.Figure:
    """Create matrix PL plot using plotly."""
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        column_widths=(0.6, 0.2, 0.2),
        subplot_titles=("", "peak intensity", "peak position"),
    )
    fig.add_trace(
        go.Heatmap(z=data, x=column, y=index, colorscale="turbo"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=index, x=max_intensity, mode="lines+markers"),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(y=index, x=pos, mode="lines+markers"), row=1, col=3)

    # Calculate ranges with margins
    max_intensity_range = (
        max_intensity.min() - (max_intensity.max() - max_intensity.min()) * 0.05,
        max_intensity.max() + (max_intensity.max() - max_intensity.min()) * 0.05,
    )
    pos_range = (
        pos.min() - (pos.max() - pos.min()) * 0.05,
        pos.max() + (pos.max() - pos.min()) * 0.05,
    )

    # Update layout for each subplot
    fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
    fig.update_xaxes(
        title_text="Intensity (arb. u.)", range=max_intensity_range, row=1, col=2
    )
    fig.update_xaxes(title_text="Wavelength (nm)", range=pos_range, row=1, col=3)
    fig.update_yaxes(title_text="X Position (μm)")

    # Update overall layout
    fig.update_layout(
        # title=title,
        # height=DEFAULT_HEIGHT * 2,  # Double the height for better aspect ratio
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Save the plot
    fig.write_image("image/matrix_plot.svg", width=640, height=360)

    return fig


def create_matrix_extract_plot(
    data: pd.DataFrame,
) -> go.Figure:
    data = data.T
    data.index = data.index.astype(float)
    fig = px.line(data)
    return fig


def process_matrix_upload(contents: str, filename: str) -> Tuple[
    pd.DataFrame,
    pd.Index,
    np.ndarray,
    np.ndarray,
    pd.Index,
    int,
]:
    """Process uploaded matrix PL data file.

    Returns:
        Tuple containing (data, index, pos, max_intensity, column, error_message)
        If processing fails, all data values will be None and error_message
        will contain the error
    """
    if contents is None:
        return None, None, None, None, None, 0

    try:
        # Process the uploaded file
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Read and process the matrix PL data
        data, index, pos, max_intensity, column = read_matrixPL(
            io.StringIO(decoded.decode("utf-8"))
        )

        return data, index, pos, max_intensity, column, 1
    except Exception as e:
        error_msg = f"Error processing matrix PL file: {str(e)}"
        print(error_msg)
        return (
            pd.DataFrame(),
            pd.Index(),
            np.ndarray(1),
            np.ndarray(1),
            pd.Index(),
            0,
        )  # return empty values under error to fit type check


# Layout
app.layout = html.Div(
    [
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Simple PL Analysis",
                    children=[
                        dcc.Upload(
                            id="upload-list-pl",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select Files")]
                            ),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                            },
                            multiple=True,
                            loading_state=dict(
                                component_name="smooth-plot",
                                is_loading=True,
                                style={"padding": "30px 0"},
                            ),
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "center",
                            },
                            children=[
                                dcc.Checklist(
                                    id="file-selection",
                                    options=[],  # This will be populated dynamically
                                    value=[],  # Initially no files are selected
                                    inline=True,
                                    labelStyle={"display": "block"},
                                ),
                                html.Button(
                                    "Clear all data",
                                    id="clear-all-data",
                                    n_clicks=0,
                                    style={"marginLeft": "100px"},
                                ),
                            ],
                        ),
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label="Simple Plot",
                                    children=[
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="simple-plot",
                                                    config={
                                                        "toImageButtonOptions": {
                                                            "filename": "simple_plot",
                                                            "format": "svg",
                                                        }
                                                    },
                                                ),
                                                dcc.Checklist(
                                                    id="plot-options",
                                                    options=[
                                                        {
                                                            "label": "Energy plot",
                                                            "value": "ev_plot",
                                                        },
                                                        {
                                                            "label": "Normalizing",
                                                            "value": "normalize",
                                                        },
                                                        {
                                                            "label": "Log Y axis",
                                                            "value": "log_y",
                                                        },
                                                    ],
                                                    value=[],
                                                    inline=True,
                                                ),
                                                html.Button(
                                                    "Save Simple Plot",
                                                    id="save-simple-plot",
                                                    n_clicks=0,
                                                ),
                                                dcc.Download(id="download-simple-plot"),
                                            ]
                                        )
                                    ],
                                ),
                                dcc.Tab(
                                    label="Smooth & Peak",
                                    children=[
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="smooth-plot",
                                                    config={
                                                        "toImageButtonOptions": {
                                                            "filename": "smooth_plot",
                                                            "format": "svg",
                                                        }
                                                    },
                                                ),
                                                html.Button(
                                                    "Save Smooth Plot",
                                                    id="save-smooth-plot",
                                                    n_clicks=0,
                                                ),
                                                dcc.Download(id="download-smooth-plot"),
                                            ]
                                        )
                                    ],
                                ),
                            ]
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Matrix PL Analysis",
                    children=[
                        html.Div(
                            [
                                dcc.Upload(
                                    id="upload-matrix-pl",
                                    children=html.Div(
                                        ["Upload Data for Matrix PL Analysis"]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                    loading_state=dict(
                                        component_name="smooth-plot",
                                        is_loading=True,
                                        style={"padding": "30px 0"},
                                    ),
                                ),
                                html.H3(
                                    id="matrix-plot-title",
                                    style={"textAlign": "center", "margin": "10px"},
                                ),
                                dcc.Graph(
                                    id="matrix-plot",
                                    config={
                                        "toImageButtonOptions": {
                                            "filename": "matrix_plot",
                                            "format": "svg",
                                        }
                                    },
                                ),
                                html.Button(
                                    "Save Matrix Plot",
                                    id="save-matrix-plot",
                                    n_clicks=0,
                                ),
                                dcc.Download(id="download-matrix-plot"),
                                dcc.Graph(
                                    id="matrix-extract-plot",
                                    config={
                                        "toImageButtonOptions": {
                                            "filename": "matrix_extract_plot",
                                            "format": "svg",
                                        }
                                    },
                                ),
                            ]
                        )
                    ],
                ),
            ]
        ),
        dcc.Store(id="stored-data", data=[]),
    ]
)


@app.callback(
    Output("stored-data", "data"),
    Output("file-selection", "options"),
    Output("file-selection", "value"),
    Input("upload-list-pl", "contents"),
    State("upload-list-pl", "filename"),
    State("stored-data", "data"),
    State("file-selection", "value"),
    Input("clear-all-data", "n_clicks"),
)
def update_data(contents, filenames, existing_data, existing_selection, n_clicks):
    if contents is None:
        raise PreventUpdate
    if ctx.triggered_id == "clear-all-data":
        return [], [], []

    data_list = existing_data if existing_data else []
    file_options = [stored_data["name"] for stored_data in existing_data]
    selected_files = list(existing_selection)

    for content, filename in zip(contents, filenames):
        df = read_data(content, filename)
        if df is not None:
            data_list.append({"name": filename, "data": df.to_dict("list")})
            file_options.append(filename)
            selected_files.append(filename)

    return data_list, file_options, selected_files


@app.callback(
    Output("simple-plot", "figure"),
    Input("stored-data", "data"),
    Input("plot-options", "value"),
    State("file-selection", "value"),
)
def update_simple_plot(data_list, options, selected_files):
    processed_data = process_selected_files(data_list, selected_files)
    return simple_chart(
        processed_data, "ev_plot" in options, "normalize" in options, "log_y" in options
    )


@app.callback(
    Output("smooth-plot", "figure"),
    Input("stored-data", "data"),
    State("file-selection", "value"),
)
def update_smooth_plot(data_list, selected_files):
    processed_data = process_selected_files(data_list, selected_files)
    return smooth_chart(processed_data)


@app.callback(
    Output("download-smooth-plot", "data"),
    Input("save-smooth-plot", "n_clicks"),
    State("smooth-plot", "figure"),
    prevent_initial_call=True,
)
def save_smooth_plot(n_clicks, figure):
    if ctx.triggered_id == "save-smooth-plot":
        return dcc.send_file("image/smooth_plot43.svg")


@app.callback(
    Output("download-simple-plot", "data"),
    Input("save-simple-plot", "n_clicks"),
    State("simple-plot", "figure"),
    prevent_initial_call=True,
)
def save_simple_plot(n_clicks, figure):
    if ctx.triggered_id == "save-simple-plot":
        return dcc.send_file("image/simple_plot43.svg")


@app.callback(
    [Output("matrix-plot", "figure"), Output("matrix-plot-title", "children")],
    Input("upload-matrix-pl", "contents"),
    State("upload-matrix-pl", "filename"),
    prevent_initial_call=True,
)
def update_matrix_plot(contents, filename):
    data, index, pos, max_intensity, column, health = process_matrix_upload(
        contents, filename
    )

    if health == 0:
        return go.Figure()

    fig = create_matrix_plot(
        data,
        index,
        pos,
        max_intensity,
        column,
        f"Matrix PL Analysis: {filename}",
    )

    return fig, f"Matrix PL Analysis: {filename}"


@app.callback(
    Output("matrix-extract-plot", "figure"),
    Input("upload-matrix-pl", "contents"),
    State("upload-matrix-pl", "filename"),
    prevent_initial_call=True,
)
def update_matrix_extract_plot(contents, filename):
    (data, _, _, _, _, health) = process_matrix_upload(contents, filename)

    if health == 0:
        return go.Figure()

    fig = create_matrix_extract_plot(
        data,
    )

    return fig


@app.callback(
    Output("download-matrix-plot", "data"),
    Input("save-matrix-plot", "n_clicks"),
    prevent_initial_call=True,
)
def save_matrix_plot(n_clicks):
    if ctx.triggered_id == "save-matrix-plot":
        return dcc.send_file("image/matrix_plot.svg")


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")  # fix host to 0.0.0.0 for docker
