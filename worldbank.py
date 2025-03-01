import datetime
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import pandas as pd
from pandas_datareader import wb
import numpy as np

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Define indicators
indicators = {
    "SP.DYN.CBRT.IN": "Birth rate, crude (per 1,000 people)",
    "EG.USE.COMM.FO.ZS": "Fossil fuel energy consumption (% of total)",
    "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total final energy consumption)",
    "SH.DYN.MORT": "Mortality rate, under-5 (per 1,000 live births)"
}

# Get country name and ISO id for mapping on choropleth
countries = wb.get_countries()
countries["capitalCity"].replace({"": None}, inplace=True)
countries.dropna(subset=["capitalCity"], inplace=True)
countries = countries[["name", "iso3c", "region"]]
countries = countries[countries["name"] != "Kosovo"]
countries = countries.rename(columns={"name": "country"})
countries = countries[countries["country"] != "Korea, Dem. People's Rep."]


def update_wb_data():
    # Retrieve specific world bank data from API
    df = wb.download(
        indicator=(list(indicators)), country=countries["iso3c"], start=2000, end=2020
    )
    df = df.reset_index()
    df.year = df.year.astype(int)

    # Add country ISO3 id and region to main df
    df = pd.merge(df, countries, on="country")
    df = df.rename(columns=indicators)

    # Calculate additional metrics for advanced analysis
    # Energy balance ratio (renewable vs fossil)
    energy_columns = [
        "Fossil fuel energy consumption (% of total)",
        "Renewable energy consumption (% of total final energy consumption)"
    ]

    # Create a sustainability score (higher is better)
    # Low mortality + high renewable - high fossil fuel = better sustainability
    df['Sustainability Score'] = df.apply(
        lambda row: (
                (100 - row['Mortality rate, under-5 (per 1,000 live births)'] if pd.notnull(
                    row['Mortality rate, under-5 (per 1,000 live births)']) else 50) * 0.5 +
                (row['Renewable energy consumption (% of total final energy consumption)'] if pd.notnull(
                    row['Renewable energy consumption (% of total final energy consumption)']) else 0) * 0.3 -
                (row['Fossil fuel energy consumption (% of total)'] if pd.notnull(
                    row['Fossil fuel energy consumption (% of total)']) else 0) * 0.2
        ),
        axis=1
    )

    return df


# Callback to update the last updated time in the subheading
@app.callback(
    Output("last-updated", "children"),
    Input("timer", "n_intervals"),
)
def update_last_fetched_time(n_intervals):
    now = datetime.now()
    human_readable_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return f"Data last fetched: {human_readable_time}"


# Callback for the time-series animation slider
@app.callback(
    Output("animation-progress", "children"),
    Output("animation-year", "data"),
    Input("play-button", "n_clicks"),
    Input("animation-interval", "n_intervals"),
    State("animation-year", "data"),
    State("years-range", "value"),
    prevent_initial_call=True
)
def update_animation(play_clicks, n_intervals, current_year, year_range):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if current_year is None:
        current_year = year_range[0]

    if triggered_id == "play-button":
        # Reset to start when play button is clicked
        current_year = year_range[0]
    elif triggered_id == "animation-interval":
        # Increment year for animation
        current_year += 1
        if current_year > year_range[1]:
            current_year = year_range[0]

    return f"Animation Year: {current_year}", current_year


# Store the selected country for secondary visualization
@app.callback(
    Output("selected-country", "data"),
    Input("my-choropleth", "clickData"),
    prevent_initial_call=True
)
def store_clicked_country(click_data):
    if click_data is None:
        return None
    country_code = click_data["points"][0]["location"]
    return country_code


# Layout - RESTRUCTURED as requested
app.layout = dbc.Container(
    [
        # Header section
        dbc.Row(
            dbc.Col(
                [
                    html.H1(
                        "Socioeconomic Development & Energy Usage Patterns",
                        style={"textAlign": "center", "marginBottom": "15px", "marginTop": "30px"},
                    ),
                    html.H5(
                        id="last-updated",
                        children="Data last fetched: Not yet updated",
                        style={"textAlign": "center", "color": "gray", "marginBottom": "25px"},
                    ),
                    dbc.Alert(
                        "Click on any country in the map to see detailed statistics!",
                        color="info",
                        style={"textAlign": "center", "marginBottom": "30px", "maxWidth": "800px", "margin": "0 auto"}
                    ),
                ],
                width=12,
            ),
            className="mb-4",  # Add bottom margin to the header row
        ),

        # Controls section
        dbc.Card(
            dbc.CardBody([
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Select Indicator:",
                                    className="fw-bold",
                                    style={"fontSize": 18, "marginBottom": "10px"},
                                ),
                                dcc.Dropdown(
                                    id="dropdown-indicator",
                                    options=[{"label": i, "value": i} for i in indicators.values()],
                                    value=list(indicators.values())[0],
                                    clearable=False,
                                    style={"width": "100%"},
                                ),
                            ],
                            width=12,
                            lg=4,
                            style={"paddingRight": "20px"},
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Select Year Range:",
                                    className="fw-bold",
                                    style={"fontSize": 18, "marginBottom": "10px"},
                                ),
                                dcc.RangeSlider(
                                    id="years-range",
                                    min=2000,
                                    max=2020,
                                    step=1,
                                    value=[2010, 2015],
                                    marks={i: str(i) if i % 5 == 0 else "" for i in range(2000, 2021)},
                                ),
                            ],
                            width=12,
                            lg=4,
                            style={"paddingRight": "20px"},
                        ),
                        dbc.Col(
                            [
                                dbc.Label(
                                    "Visualization Type:",
                                    className="fw-bold",
                                    style={"fontSize": 18, "marginBottom": "10px"},
                                ),
                                dbc.RadioItems(
                                    id="viz-type",
                                    options=[
                                        {"label": "Standard Map", "value": "standard"},
                                        {"label": "Sustainability Score", "value": "sustainability"},
                                    ],
                                    value="standard",
                                    inline=True,
                                    inputClassName="me-2",
                                    labelClassName="me-3",
                                ),
                            ],
                            width=12,
                            lg=4,
                        ),
                    ],
                    className="mb-4",  # Add margin to the control rows
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div([
                                html.Div(
                                    id="animation-progress",
                                    children="Animation Year: Not Started",
                                    style={"marginRight": "15px", "fontWeight": "bold"}
                                ),
                                dbc.Button(
                                    "Play Time Animation",
                                    id="play-button",
                                    color="success",
                                    className="me-3",
                                ),
                                dbc.Button(
                                    "Update Map",
                                    id="update-button",
                                    n_clicks=0,
                                    color="primary",
                                    className="fw-bold",
                                ),
                            ]),
                            width=12,
                            className="d-flex justify-content-center align-items-center",
                            style={"marginTop": "10px", "marginBottom": "10px"},
                        ),
                    ],
                ),
            ]),
            className="mb-4",  # Add bottom margin to the card
            style={"boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)", "border": "none"},
        ),

        # 1. WORLD MAP SECTION (Moved to top as requested)
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Global Data Visualization",
                                style={"textAlign": "center", "marginBottom": "15px", "marginTop": "10px"}),
                        dcc.Graph(id="my-choropleth", figure={}, style={"height": "65vh"}),
                    ],
                    width=12,
                ),
            ],
            className="mb-5",  # Add bottom margin to this row
        ),

        # 2. REGIONAL STATS SECTION (Moved to middle as requested)
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(id="country-detail-title", children=[
                            html.H3("Regional Statistics", style={"textAlign": "center", "marginBottom": "15px"}),
                            html.P("Click on a country in the map to see regional details",
                                   style={"textAlign": "center", "fontStyle": "italic", "marginBottom": "20px"})
                        ]),
                        dcc.Graph(id="country-detail", figure={}, style={"height": "60vh"}),
                    ],
                    width=12,
                ),
            ],
            className="mb-5",  # Add bottom margin to this row
        ),

        # 3. CROSS INDICATOR SECTION (Moved to bottom as requested)
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Cross-Indicator Analysis",
                                style={"textAlign": "center", "marginBottom": "15px"}),
                        html.P(
                            "This analysis shows the relationship between indicators across countries.",
                            style={"textAlign": "center", "marginBottom": "15px"}
                        ),

                        # Scatter plot controls
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Row(
                                    [
                                        dbc.Col([
                                            dbc.Label("X-Axis Indicator:", style={"marginBottom": "8px"}),
                                            dcc.Dropdown(
                                                id="x-axis-indicator",
                                                options=[{"label": i, "value": i} for i in indicators.values()],
                                                value=list(indicators.values())[0],
                                                clearable=False,
                                            ),
                                        ], width=12, lg=4, style={"paddingRight": "15px"}),
                                        dbc.Col([
                                            dbc.Label("Y-Axis Indicator:", style={"marginBottom": "8px"}),
                                            dcc.Dropdown(
                                                id="y-axis-indicator",
                                                options=[{"label": i, "value": i} for i in indicators.values()],
                                                value=list(indicators.values())[2],
                                                clearable=False,
                                            ),
                                        ], width=12, lg=4, style={"paddingRight": "15px"}),
                                        dbc.Col([
                                            dbc.Label("Size Indicator (Optional):", style={"marginBottom": "8px"}),
                                            dcc.Dropdown(
                                                id="size-indicator",
                                                options=[{"label": "None", "value": "none"}] +
                                                        [{"label": i, "value": i} for i in indicators.values()],
                                                value="none",
                                            ),
                                        ], width=12, lg=4),
                                    ],
                                ),
                            ]),
                            className="mb-3",
                            style={"boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)", "border": "none"},
                        ),

                        dcc.Graph(id="scatter-correlation", figure={}, style={"height": "60vh"}),
                    ],
                    width=12,
                ),
            ],
            className="mb-5",  # Add bottom margin to this row
        ),

        # Hidden components for state management
        dcc.Store(id="storage", storage_type="session", data={}),
        dcc.Store(id="selected-country", storage_type="memory", data=None),
        dcc.Store(id="animation-year", storage_type="memory", data=None),
        dcc.Interval(id="timer", interval=1000 * 60, n_intervals=0),
        dcc.Interval(id="animation-interval", interval=1000, n_intervals=0, disabled=True),
    ],
    fluid=True,
    style={"paddingTop": "20px", "paddingBottom": "60px", "maxWidth": "1440px"}
    # Limit max width for better readability
)


# Enable/disable animation interval
@app.callback(
    Output("animation-interval", "disabled"),
    Input("play-button", "n_clicks"),
    Input("animation-interval", "n_intervals"),
    State("years-range", "value"),
    State("animation-year", "data"),
    prevent_initial_call=True
)
def toggle_animation_interval(play_clicks, n_intervals, year_range, current_year):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "play-button":
        # Start animation
        return False

    if triggered_id == "animation-interval" and current_year == year_range[1]:
        # Stop animation after completing a cycle
        return True

    return False


# Callback to update the data store
@app.callback(
    Output("storage", "data"),
    Input("timer", "n_intervals")
)
def store_data(n_time):
    dataframe = update_wb_data()
    return dataframe.to_dict("records")


# Main callback to update the choropleth
@app.callback(
    Output("my-choropleth", "figure"),
    Input("update-button", "n_clicks"),
    Input("play-button", "n_clicks"),
    Input("animation-year", "data"),
    Input("storage", "data"),
    Input("viz-type", "value"),
    State("years-range", "value"),
    State("dropdown-indicator", "value"),
)
def update_graph(n_clicks, play_clicks, animation_year, stored_dataframe, viz_type, years_chosen, indct_chosen):
    dff = pd.DataFrame.from_records(stored_dataframe)

    # If animation is active, use that year
    if animation_year is not None:
        dff = dff[dff.year == animation_year]
    else:
        # Otherwise use the selected year range
        if years_chosen[0] != years_chosen[1]:
            dff = dff[dff.year.between(years_chosen[0], years_chosen[1])]

            # FIX: Select only numeric columns for aggregation
            numeric_cols = dff.select_dtypes(include=['number']).columns
            # Make sure the groupby columns are included separately
            groupby_cols = ['iso3c', 'country', 'region']

            # Group by and calculate mean only for numeric columns
            dff = dff.groupby(groupby_cols)[numeric_cols].mean().reset_index()
        else:
            dff = dff[dff["year"] == years_chosen[0]]

    # Determine which value to show based on visualization type
    if viz_type == "sustainability":
        color_column = "Sustainability Score"
        title = "Sustainability Score Map"
        color_scale = "RdYlGn"  # Red-Yellow-Green scale
    else:
        color_column = indct_chosen
        title = f"{indct_chosen} by Country"
        # Set appropriate color scales based on indicator
        if "Mortality" in indct_chosen or "Fossil fuel" in indct_chosen:
            color_scale = "Reds"  # Higher values are worse
        else:
            color_scale = "Blues"  # Higher values are better/neutral

    fig = px.choropleth(
        data_frame=dff,
        locations="iso3c",
        color=color_column,
        scope="world",
        color_continuous_scale=color_scale,
        hover_data={
            "iso3c": False,
            "country": True,
            "region": True,
            color_column: True
        },
        title=title
    )

    # Customize the layout
    fig.update_layout(
        geo={"projection": {"type": "natural earth"}},
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_colorbar=dict(
            title=color_column,
            thickness=20,
            len=0.7,
        ),
        title={
            "font": {"size": 24},
            "x": 0.5,
            "xanchor": "center"
        },
        height=600,  # Set consistent height
    )

    # Add a text annotation for the animation year if active
    if animation_year is not None:
        fig.add_annotation(
            text=f"Year: {animation_year}",
            x=0.9,
            y=0.9,
            showarrow=False,
            font=dict(size=20, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )

    return fig


# Callback for secondary visualization when a country is clicked
@app.callback(
    [Output("country-detail", "figure"),
     Output("country-detail-title", "children")],
    [Input("selected-country", "data"),
     Input("storage", "data")],
    prevent_initial_call=True
)
def update_country_detail(selected_country, stored_dataframe):
    if selected_country is None:
        # Default empty figure
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(
                text="Click on a country in the map",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )],
            height=600  # Set consistent height
        )
        title_component = [
            html.H3("Regional Statistics", style={"textAlign": "center", "marginBottom": "15px"}),
            html.P("Click on a country to see regional details",
                   style={"textAlign": "center", "fontStyle": "italic", "marginBottom": "20px"})
        ]
        return fig, title_component

    # Load data
    dff = pd.DataFrame.from_records(stored_dataframe)

    # Filter for selected country
    country_data = dff[dff["iso3c"] == selected_country]
    country_name = country_data["country"].iloc[0] if not country_data.empty else "Unknown Country"

    # Create a subplot with 2 rows and 1 column
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Time Series Analysis", "Indicator Comparison"),
        vertical_spacing=0.22  # Increase vertical spacing between subplots
    )

    # 1. Time series for selected indicators
    for indicator in indicators.values():
        fig.add_trace(
            go.Scatter(
                x=country_data["year"],
                y=country_data[indicator],
                mode="lines+markers",
                name=indicator,
                hovertemplate="%{y:.2f}"
            ),
            row=1, col=1
        )

    # 2. Bar chart comparing the latest values of each indicator
    latest_year = country_data["year"].max()
    latest_data = country_data[country_data["year"] == latest_year]

    if not latest_data.empty:
        indicator_values = []
        indicator_names = []

        for indicator in indicators.values():
            if pd.notnull(latest_data[indicator].iloc[0]):
                indicator_values.append(latest_data[indicator].iloc[0])
                indicator_names.append(indicator)

        fig.add_trace(
            go.Bar(
                x=indicator_names,
                y=indicator_values,
                text=indicator_values,
                textposition="auto",
                hovertemplate="%{y:.2f}"
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=600,
        legend=dict(
            orientation="h",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)  # Smaller font for legend
        ),
        margin=dict(l=20, r=20, t=80, b=80),  # Increase margins
        hovermode="closest"
    )

    # Update axes
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    # Limit the number of ticks to avoid crowding
    fig.update_xaxes(nticks=10, row=1, col=1)

    # Create title component
    title_component = [
        html.H3(f"{country_name} Regional Statistics", style={"textAlign": "center", "marginBottom": "15px"}),
        html.P(f"Region: {country_data['region'].iloc[0] if not country_data.empty else 'Unknown'}",
               style={"textAlign": "center", "marginBottom": "15px"})
    ]

    return fig, title_component


# Callback for the scatter correlation plot
@app.callback(
    Output("scatter-correlation", "figure"),
    [Input("x-axis-indicator", "value"),
     Input("y-axis-indicator", "value"),
     Input("size-indicator", "value"),
     Input("storage", "data"),
     Input("years-range", "value")],
    prevent_initial_call=True
)
def update_correlation_scatter(x_indicator, y_indicator, size_indicator, stored_dataframe, years_chosen):
    dff = pd.DataFrame.from_records(stored_dataframe)

    # Filter for selected year range
    if years_chosen[0] != years_chosen[1]:
        dff = dff[dff.year.between(years_chosen[0], years_chosen[1])]

        # Select only numeric columns for mean calculation
        numeric_cols = dff.select_dtypes(include=['number']).columns

        # Ensure we include our indicators of interest if they're not in numeric_cols
        required_cols = []
        for col in [x_indicator, y_indicator]:
            if col in dff.columns and col not in numeric_cols:
                try:
                    dff[col] = pd.to_numeric(dff[col], errors='coerce')
                    required_cols.append(col)
                except:
                    pass

        if size_indicator != "none" and size_indicator in dff.columns and size_indicator not in numeric_cols:
            try:
                dff[size_indicator] = pd.to_numeric(dff[size_indicator], errors='coerce')
                required_cols.append(size_indicator)
            except:
                pass

        # Combine required columns with existing numeric columns
        calc_cols = list(set(list(numeric_cols) + required_cols))

        # Take the average for each country over the selected years
        dff = dff.groupby(["iso3c", "country", "region"])[calc_cols].mean().reset_index()
        year_text = f"{years_chosen[0]}-{years_chosen[1]} (average)"
    else:
        dff = dff[dff["year"] == years_chosen[0]]
        year_text = str(years_chosen[0])

    # Set up hover data
    hover_data = {
        "country": True,
        "region": True,
        x_indicator: ':.2f',
        y_indicator: ':.2f',
    }

    # Drop rows with NaN values in x or y indicators to avoid plotting issues
    dff = dff.dropna(subset=[x_indicator, y_indicator])

    # Set up the scatter plot
    if size_indicator != "none":
        # Add the size indicator to hover data
        hover_data[size_indicator] = ':.2f'

        # Drop rows with NaN in the size indicator
        valid_size_data = dff.dropna(subset=[size_indicator])

        # Only proceed with size parameter if we have valid data
        if not valid_size_data.empty:
            fig = px.scatter(
                valid_size_data,
                x=x_indicator,
                y=y_indicator,
                size=size_indicator,
                color="region",
                hover_name="country",
                hover_data=hover_data,
                size_max=30,
                title=f"Relationship between {x_indicator} and {y_indicator} ({year_text})",
                labels={
                    x_indicator: x_indicator,
                    y_indicator: y_indicator,
                    size_indicator: size_indicator
                }
            )
        else:
            # Fallback to regular scatter plot without size if no valid size data
            fig = px.scatter(
                dff,
                x=x_indicator,
                y=y_indicator,
                color="region",
                hover_name="country",
                hover_data=hover_data,
                title=f"Relationship between {x_indicator} and {y_indicator} ({year_text})",
                labels={
                    x_indicator: x_indicator,
                    y_indicator: y_indicator
                }
            )
    else:
        fig = px.scatter(
            dff,
            x=x_indicator,
            y=y_indicator,
            color="region",
            hover_name="country",
            hover_data=hover_data,
            title=f"Relationship between {x_indicator} and {y_indicator} ({year_text})",
            labels={
                x_indicator: x_indicator,
                y_indicator: y_indicator
            }
        )

    # Add a trend line
    # Calculate correlation
    valid_data = dff[[x_indicator, y_indicator]].dropna()
    if len(valid_data) > 1:  # Need at least 2 points for correlation
        correlation = np.corrcoef(valid_data[x_indicator], valid_data[y_indicator])[0, 1]

        # Add trendline
        fig.add_trace(
            go.Scatter(
                x=valid_data[x_indicator],
                y=np.poly1d(np.polyfit(valid_data[x_indicator], valid_data[y_indicator], 1))(valid_data[x_indicator]),
                mode='lines',
                name=f'Trend (r = {correlation:.2f})',
                line=dict(color='rgba(0,0,0,0.5)', dash='dash')
            )
        )

    # Update layout
    fig.update_layout(
        height=600,
        legend=dict(
            orientation="h",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=20, r=20, t=80, b=100),  # Increase margins
        xaxis=dict(title=dict(text=x_indicator, font=dict(size=14))),
        yaxis=dict(title=dict(text=y_indicator, font=dict(size=14))),
        title={
            "font": {"size": 22},
            "x": 0.5,
            "xanchor": "center",
            "y": 0.95
        }
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)