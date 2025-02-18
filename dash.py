import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from plotly import graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy import signal
import os
from dash.exceptions import PreventUpdate

if not os.path.exists("image"):
    os.mkdir("image")

# Initialize the Dash app
app = dash.Dash(__name__)


def read_data(contents, filename) -> pd.DataFrame:
    import base64
    import io

    h = 6.63e-34 / 1.6e-19  # eV energy
    c = 3e8

    # Decode the uploaded file contents
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        # Read the file into pandas DataFrame
        df = pd.read_csv(
            io.StringIO(decoded.decode("utf-8")),
            sep=r"\s+",
            header=None,
            names=["wavelength", "count"],
        )

        # normalize
        df["normalized"] = (df["count"] - df["count"].min()) / (
            df["count"].max() - df["count"].min()
        )
        df["energy"] = h * c / (df["wavelength"] * 1e-9)
        return df
    except Exception as e:
        print(e)
        return None


def make_chart1(data_list):
    if not data_list:
        return go.Figure()

    chart = (
        go.Figure()
        .update_xaxes(title_text="Wavelength(nm)")
        .update_yaxes(title_text="Intentsity(arb.u.)")
        .update_layout(legend=dict(x=0, y=0.5))
    )

    colors = px.colors.qualitative.Plotly
    for idx, data in enumerate(data_list):
        filter = signal.savgol_filter(data["data"]["normalized"], 150, 2)
        chart.add_trace(
            go.Scatter(
                x=data["data"]["wavelength"],
                y=filter,
                name=data["name"],
                line_color=colors[idx],
            )
        )
        chart.add_vline(
            x=data["data"]["wavelength"][filter.argmax()],
            line_color=colors[idx],
            annotation=dict(
                text=str(data["data"]["wavelength"][filter.argmax()]),
                textangle=90,
            ),
            annotation_position="bottom left",
        )
    return chart


def simple_chart(data_list, ev_plot, normalize, log_y=False):
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
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Smooth & Peak",
                    children=[html.Div([dcc.Graph(id="smooth-plot")])],
                ),
            ]
        ),
        # Store for keeping track of uploaded data
        dcc.Store(id="stored-data", data=[]),
    ]
)


@app.callback(
    Output("stored-data", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_data(contents, filenames):
    if contents is None:
        raise PreventUpdate

    data_list = []
    for content, filename in zip(contents, filenames):
        df = read_data(content, filename)
        if df is not None:
            data_list.append({"name": filename, "data": df.to_dict("list")})
    return data_list


@app.callback(
    Output("simple-plot", "figure"),
    Input("stored-data", "data"),
    Input("plot-options", "value"),
)
def update_simple_plot(data_list, options):
    if not data_list:
        raise PreventUpdate

    # Convert stored dict data back to DataFrame format
    processed_data = []
    for item in data_list:
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

    return make_chart1(processed_data)


if __name__ == "__main__":
    app.run_server(debug=True)
