from dash import dcc, html, Input, Output, State, Dash
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
import os
import base64
import io
from functools import lru_cache
from typing import List, Dict, Optional, Union

# Constants
PLANCK_CONST = 6.63e-34 / 1.6e-19  # eV energy
SPEED_OF_LIGHT = 3e8
SAVGOL_WINDOW = 150
SAVGOL_POLY = 2

if not os.path.exists("image"):
    os.mkdir("image")

# Initialize the Dash app
app = Dash(__name__)


@lru_cache(maxsize=32)
def read_data(contents: str, filename: str) -> Optional[pd.DataFrame]:
    """Read and process uploaded data file with caching."""
    try:
        content_type, content_string = contents.split(",")
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
        filter = signal.savgol_filter(normalized, SAVGOL_WINDOW, SAVGOL_POLY)
        peak_idx = filter.argmax()

        chart.add_trace(
            go.Scatter(
                x=wavelength,
                y=filter,
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

    chart.write_image("image/smooth_plot.svg")
    return chart


def simple_chart(
    data_list: List[Dict], ev_plot: bool, normalize: bool, log_y: bool = False
) -> go.Figure:
    """Create simple chart with various display options."""
    if not data_list:
        return go.Figure()

    chart = go.Figure().update_layout(legend=dict(x=0, y=0.5))

    if ev_plot:
        data_x = "energy"
        chart.update_xaxes(title_text="Energy (eV)", autorange="reversed")
    else:
        data_x = "wavelength"
        chart.update_xaxes(title_text="Wavelength (nm)")

    if normalize:
        data_y = "normalized"
        chart.update_yaxes(title_text="Normalized intentsity (arb.u.)")
    else:
        data_y = "count"
        chart.update_yaxes(title_text="Intentsity (arb.u.)")

    if log_y:
        chart.update_yaxes(type="log", minor=dict(ticks="inside", showgrid=True))

    for data in data_list:
        chart.add_trace(
            go.Scatter(
                x=data["data"][data_x], y=data["data"][data_y], name=data["name"]
            )
        )
    chart.write_image("image/simple_plot.svg")
    return chart


# Layout
app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
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
        ),
        dcc.Checklist(
            id="file-selection",
            options=[],  # This will be populated dynamically
            value=[],  # Initially no files are selected
            inline=True,
            labelStyle={"display": "block"},  # Display each option on a new line
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Simple Plot",
                    children=[
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="plot-options",
                                    options=[
                                        {"label": "Energy plot", "value": "ev_plot"},
                                        {"label": "Normalizing", "value": "normalize"},
                                        {"label": "Log Y axis", "value": "log_y"},
                                    ],
                                    value=[],
                                    inline=True,
                                ),
                                dcc.Graph(id="simple-plot"),
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
                                dcc.Graph(id="smooth-plot"),
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
        # Store for keeping track of uploaded data
        dcc.Store(id="stored-data", data=[]),
    ]
)


@app.callback(
    Output("stored-data", "data"),
    Output("file-selection", "options"),  # Update file options
    Output("file-selection", "value"),  # Add this output to set initial values
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("stored-data", "data"),
    State("file-selection", "value"),  # Add this to preserve existing selections
)
def update_data(contents, filenames, existing_data, existing_selection):
    if contents is None:
        raise PreventUpdate

    # Initialize data_list with existing data
    data_list = existing_data if existing_data else []
    file_options = []  # To store file options for the checklist
    selected_files = list(existing_selection)  # Convert to list to modify

    for content, filename in zip(contents, filenames):
        df = read_data(content, filename)
        if df is not None:
            data_list.append({"name": filename, "data": df.to_dict("list")})
            file_options.append({"label": filename, "value": filename})
            selected_files.append(filename)  # Add new file to selected files

    return (
        data_list,
        file_options,
        selected_files,
    )  # Return updated list, options, and selections


@app.callback(
    Output("simple-plot", "figure"),
    Input("stored-data", "data"),
    Input("plot-options", "value"),
    Input("file-selection", "value"),  # Get selected files
)
def update_simple_plot(data_list, options, selected_files):
    if not data_list:
        raise PreventUpdate

    # Convert stored dict data back to DataFrame format
    processed_data = []
    for item in data_list:
        if item["name"] in selected_files:  # Only process selected files
            item["data"] = pd.DataFrame(item["data"])
            processed_data.append(item)

    return simple_chart(
        processed_data, "ev_plot" in options, "normalize" in options, "log_y" in options
    )


@app.callback(Output("smooth-plot", "figure"), Input("stored-data", "data"))
def update_smooth_plot(data_list):
    if not data_list:
        raise PreventUpdate

    # Convert stored dict data back to DataFrame format
    processed_data = []
    for item in data_list:
        item["data"] = pd.DataFrame(item["data"])
        processed_data.append(item)

    return smooth_chart(processed_data)


@app.callback(
    Output("download-smooth-plot", "data"),
    Input("save-smooth-plot", "n_clicks"),
    State("smooth-plot", "figure"),
    prevent_initial_call=True,
)
def save_smooth_plot(n_clicks, figure):
    if n_clicks > 0:
        # Save the figure as an SVG file
        return dcc.send_file(
            "image/smooth_plot.svg",
        )


@app.callback(
    Output("download-simple-plot", "data"),
    Input("save-simple-plot", "n_clicks"),
    State("simple-plot", "figure"),
    prevent_initial_call=True,
)
def save_simple_plot(n_clicks, figure):
    if n_clicks > 0:
        # Save the figure as an SVG file
        return dcc.send_file(
            "image/simple_plot.svg",
        )


if __name__ == "__main__":
    app.run_server(debug=True)
