import datetime
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from pandas_datareader import wb


app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

indicators = {
    "SP.DYN.CBRT.IN": "Birth rate, crude (per 1,000 people)",
    "EG.USE.COMM.FO.ZS": "Fossil fuel energy consumption (% of total)",
    "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total final energy consumption)",
}

# get country name and ISO id for mapping on choropleth
countries = wb.get_countries()
countries["capitalCity"].replace({"": None}, inplace=True)
countries.dropna(subset=["capitalCity"], inplace=True)
countries = countries[["name", "iso3c"]]
countries = countries[countries["name"] != "Kosovo"]
countries = countries.rename(columns={"name": "country"})
countries = countries[countries["country"] != "Korea, Dem. People's Rep."]
# print(countries.head().to_string())
# exit(0)


def update_wb_data():
    # Retrieve specific world bank data from API
    df = wb.download(
        indicator=(list(indicators)), country=countries["iso3c"], start=2005, end=2016
    )
    df = df.reset_index()
    df.year = df.year.astype(int)

    # Add country ISO3 id to main df
    df = pd.merge(df, countries, on="country")
    df = df.rename(columns=indicators)
    # print(df)
    return df


# Callback to update the last updated time in the subheading
@app.callback(
    Output("last-updated", "children"),
    Input("timer", "n_intervals"),
)
def update_last_fetched_time(n_intervals):
    # Use the current time for the update
    now = datetime.now()
    human_readable_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"Data last fetched: {human_readable_time}"


# Callback to update the parameter change count and adjust the range slider selection
@app.callback(
    [
        Output("click-count", "children"),
        Output("years-range", "value"),
    ],
    [
        Input("my-button", "n_clicks"),
    ],
    [
        State("years-range", "value"),
    ],
)
def update_parameter_changes_and_slider_value(n_clicks, current_range):
    # If the button hasn't been clicked, no changes are made
    if n_clicks == 0:
        return "Choropleth parameters updated: 0 times", current_range

    # Increment the upper bound of the slider value range
    new_range = [current_range[0], current_range[1] + 1]

    # Update the click count text
    new_count_text = f"Choropleth parameters updated: {n_clicks} times"

    return new_count_text, new_range



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
                    width=6,  # First half of the row
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
                                2005: "2005",
                                2006: "'06",
                                2007: "'07",
                                2008: "'08",
                                2009: "'09",
                                2010: "'10",
                                2011: "'11",
                                2012: "'12",
                                2013: "'13",
                                2014: "'14",
                                2015: "'15",
                                2016: "2016",
                            },
                        ),
                    ],
                    width=6,  # Second half of the row
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Span(
                        id="click-count",
                        children="Choropleth parameters updated: 0 times",
                        style={"fontSize": "16px", "fontWeight": "bold"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button(
                        id="my-button",
                        children="Submit",
                        n_clicks=0,
                        color="primary",
                        className="fw-bold",
                    ),
                    width="auto",
                    className="d-flex justify-content-end",
                ),
            ],
            justify="end",
            align="center",
            style={"marginTop": "10px"},
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
    Input("my-button", "n_clicks"),
    Input("storage", "data"),
    State("years-range", "value"),
    State("dropdown-indicator", "value"),
)
def update_graph(n_clicks, stored_dataframe, years_chosen, indct_chosen):
    dff = pd.DataFrame.from_records(stored_dataframe)
    print(years_chosen)

    if years_chosen[0] != years_chosen[1]:
        dff = dff[dff.year.between(years_chosen[0], years_chosen[1])]
        dff = dff.groupby(["iso3c", "country"])[indct_chosen].mean()
        dff = dff.reset_index()

        fig = px.choropleth(
            data_frame=dff,
            locations="iso3c",
            color=indct_chosen,
            scope="world",
            hover_data={"iso3c": False, "country": True},
            labels={
                indicators["EG.FEC.RNEW.ZS"]: "% Renewable energy",
                indicators["EG.USE.COMM.FO.ZS"]: "% of Fossil Fuel Consumption",
            },
        )
        fig.update_layout(
            geo={"projection": {"type": "natural earth"}},
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig

    if years_chosen[0] == years_chosen[1]:
        dff = dff[dff["year"].isin(years_chosen)]
        fig = px.choropleth(
            data_frame=dff,
            locations="iso3c",
            color=indct_chosen,
            scope="world",
            hover_data={"iso3c": False, "country": True},
            labels={
                indicators["SG.GEN.PARL.ZS"]: "% parliament women",
                indicators["IT.NET.USER.ZS"]: "pop % using internet",
            },
        )
        fig.update_layout(
            geo={"projection": {"type": "natural earth"}},
            margin=dict(l=50, r=50, t=50, b=50),
        )
        return fig


if __name__ == "__main__":
    app.run_server(debug=True)
