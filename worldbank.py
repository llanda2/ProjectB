import datetime
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from pandas_datareader import wb

app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

indicators = {
    "SP.DYN.CBRT.IN": "Birth rate, crude (per 1,000 people)",
    "EG.USE.COMM.FO.ZS": "Fossil fuel energy consumption (% of total)",
    "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total final energy consumption)",
    "SH.DYN.MORT": "Mortality rate, under-5 (per 1,000 live births)"
}

countries = wb.get_countries()
countries["capitalCity"].replace({"": None}, inplace=True)
countries.dropna(subset=["capitalCity"], inplace=True)
countries = countries[["name", "iso3c"]]
countries = countries[countries["name"] != "Kosovo"]
countries = countries.rename(columns={"name": "country"})
countries = countries[countries["country"] != "Korea, Dem. People's Rep."]


def update_wb_data():
    df = wb.download(
        indicator=list(indicators), country=countries["iso3c"], start=2005, end=2016
    )
    df = df.reset_index()
    df.year = df.year.astype(int)
    df = pd.merge(df, countries, on="country")
    df = df.rename(columns=indicators)
    return df


@app.callback(
    Output("last-updated", "children"),
    Input("timer", "n_intervals"),
)
def update_last_fetched_time(n_intervals):
    now = datetime.now()
    human_readable_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"Data last fetched: {human_readable_time}"


app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        "Comparison of World Bank Country Data",
                        style={"textAlign": "center"},
                    ),
                    html.H4(
                        id="last-updated",
                        children="Data last fetched: Not yet updated",
                        style={"textAlign": "center", "color": "gray", "marginTop": "10px"},
                    ),
                    dcc.Graph(id="my-choropleth", figure={}),
                ],
                width=12,
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "Select Data Set:",
                            className="fw-bold",
                            style={"textDecoration": "underline", "fontSize": 20},
                        ),
                        dcc.Dropdown(
                            id="dropdown-indicator",
                            options=[{"label": i, "value": i} for i in indicators.values()],
                            value=list(indicators.values())[0],
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Label(
                            "Select Years:",
                            className="fw-bold",
                            style={"textDecoration": "underline", "fontSize": 20},
                        ),
                        dcc.RangeSlider(
                            id="years-range",
                            min=2005,
                            max=2016,
                            step=1,
                            value=[2005, 2006],
                            marks={
                                i: str(i) if i == 2005 or i == 2016 else f"'{str(i)[-2:]}"
                                for i in range(2005, 2017)
                            },
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="line-chart"), width=12),
                dbc.Col(
                    dcc.Dropdown(
                        id="compare-countries",
                        multi=True,
                        placeholder="Select additional countries to compare",
                    ),
                    width=12,
                ),
            ]
        ),
        dcc.Store(id="storage", storage_type="session", data={}),
        dcc.Interval(id="timer", interval=1000 * 60, n_intervals=0),
    ]
)


@app.callback(Output("storage", "data"), Input("timer", "n_intervals"))
def store_data(n_time):
    dataframe = update_wb_data()
    return dataframe.to_dict("records")


@app.callback(
    Output("my-choropleth", "figure"),
    Input("years-range", "value"),
    Input("dropdown-indicator", "value"),
    State("storage", "data"),
)
def update_graph(years_chosen, indct_chosen, stored_dataframe):
    dff = pd.DataFrame.from_records(stored_dataframe)

    if years_chosen[0] != years_chosen[1]:
        dff = dff[dff.year.between(years_chosen[0], years_chosen[1])]
        dff = dff.groupby(["iso3c", "country"])[indct_chosen].mean().reset_index()
    else:
        dff = dff[dff["year"].isin(years_chosen)]

    fig = px.choropleth(
        data_frame=dff,
        locations="iso3c",
        color=indct_chosen,
        scope="world",
        hover_data={"iso3c": False, "country": True},
        labels=indicators,
    )
    fig.update_layout(
        geo={"projection": {"type": "natural earth"}},
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig


@app.callback(
    Output("line-chart", "figure"),
    Input("my-choropleth", "clickData"),
    Input("compare-countries", "value"),
    State("storage", "data"),
)
def update_line_chart(clickData, compare_countries, stored_dataframe):
    dff = pd.DataFrame.from_records(stored_dataframe)
    selected_countries = []

    if clickData:
        selected_countries.append(clickData["points"][0]["location"])
    if compare_countries:
        selected_countries.extend(compare_countries)

    dff = dff[dff["iso3c"].isin(selected_countries)]

    fig = px.line(
        dff,
        x="year",
        y=["Birth rate, crude (per 1,000 people)", "Mortality rate, under-5 (per 1,000 live births)",
           "Fossil fuel energy consumption (% of total)",
           "Renewable energy consumption (% of total final energy consumption)"],
        color="country",
        title="Country Trends",
        labels={"value": "Rate", "variable": "Indicator"},
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
