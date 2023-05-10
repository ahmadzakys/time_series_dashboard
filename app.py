######-----Import Dash-----#####
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import pandas as pd
import plotly.express as px

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric

#####-----Create a Dash app instance-----#####
app = dash.Dash(
    external_stylesheets=[dbc.themes.LITERA],
    name = 'CTS Performance'
    )
app.title = 'CTS Performance Dashboard'

##-----Navbar
navbar = dbc.NavbarSimple(
    brand="CTS Performance Dashboard",
    brand_href="#",
    color="#242947",
    dark=True,
)

##-----Import Data
##-- Java
java = pd.read_excel('18-22.xlsx', sheet_name='Bulk Java')
java['ATL'] = pd.to_datetime(java['ATL'], yearfirst=True)
java['ATC'] = pd.to_datetime(java['ATC'], yearfirst=True)
java.rename(columns={'Fuel Ratio (8.7)':'Fuel Ratio'}, inplace=True)
java.sort_values(by='ATC', inplace=True)
# Monthly
java_qty_monthly = java.groupby(pd.Grouper(key='ATC', freq='M'))['Quantity'].sum().reset_index()
java_qty_monthly = java_qty_monthly[java_qty_monthly['Quantity']>0]
java_monthly = java.groupby(pd.Grouper(key='ATC', freq='M'))\
                [['Gross Loading Rate', 'Net Loading Rate', 'Fuel Ratio']]\
                .mean().reset_index()
java_monthly = java_monthly.dropna()
java_monthly = pd.merge(java_qty_monthly, java_monthly)
# Yearly
java_qty_yearly = java.groupby(pd.Grouper(key='ATC', freq='Y'))['Quantity'].sum().reset_index()
java_qty_yearly = java_qty_yearly[java_qty_yearly['Quantity']>0]
java_yearly = java.groupby(pd.Grouper(key='ATC', freq='Y'))\
                [['Gross Loading Rate', 'Net Loading Rate', 'Fuel Ratio']]\
                .mean().reset_index()
java_yearly = java_yearly.dropna()
java_yearly = pd.merge(java_qty_yearly, java_yearly)
java_yearly['ATC'] = java_yearly['ATC'].dt.year
# List
list_cts = list(['Bulk Java', 'Bulk Sumatra'])
list_variable = list(['Quantity', 'Gross Loading Rate', 'Net Loading Rate', 'Fuel Ratio'])
list_year = list([2018, 2019, 2020, 2021, 2022])

##-- Sumatra
sumatra = pd.read_excel('18-22.xlsx', sheet_name='Bulk Sumatra')
sumatra['ATL'] = pd.to_datetime(sumatra['ATL'], yearfirst=True)
sumatra['ATC'] = pd.to_datetime(sumatra['ATC'], yearfirst=True)
sumatra.rename(columns={'Fuel Ratio (8.7)':'Fuel Ratio'}, inplace=True)
sumatra.sort_values(by='ATC', inplace=True)
# Monthly
sumatra_qty_monthly = sumatra.groupby(pd.Grouper(key='ATC', freq='M'))['Quantity'].sum().reset_index()
sumatra_qty_monthly = sumatra_qty_monthly[sumatra_qty_monthly['Quantity']>0]
sumatra_monthly = sumatra.groupby(pd.Grouper(key='ATC', freq='M'))\
                [['Gross Loading Rate', 'Net Loading Rate', 'Fuel Ratio']]\
                .mean().reset_index()
sumatra_monthly = sumatra_monthly.dropna()
sumatra_monthly = pd.merge(sumatra_qty_monthly, sumatra_monthly)
# Yearly
sumatra_qty_yearly = sumatra.groupby(pd.Grouper(key='ATC', freq='Y'))['Quantity'].sum().reset_index()
sumatra_qty_yearly = sumatra_qty_yearly[sumatra_qty_yearly['Quantity']>0]
sumatra_yearly = sumatra.groupby(pd.Grouper(key='ATC', freq='Y'))\
                [['Gross Loading Rate', 'Net Loading Rate', 'Fuel Ratio']]\
                .mean().reset_index()
sumatra_yearly = sumatra_yearly.dropna()
sumatra_yearly = pd.merge(sumatra_qty_yearly, sumatra_yearly)
sumatra_yearly['ATC'] = sumatra_yearly['ATC'].dt.year

##----- Prophet

# Java

# 1. Prepare dataframe contains ds and y columns
java_monthly_qty = java_monthly[['ATC', 'Quantity']].rename(columns={'ATC':'ds','Quantity':'y'})
java_monthly_grosslr = java_monthly[['ATC', 'Gross Loading Rate']].rename(columns={'ATC':'ds','Gross Loading Rate':'y'})
java_monthly_netlr = java_monthly[['ATC', 'Net Loading Rate']].rename(columns={'ATC':'ds','Net Loading Rate':'y'})
java_monthly_fr = java_monthly[['ATC', 'Fuel Ratio']].rename(columns={'ATC':'ds','Fuel Ratio':'y'})

# 2. Fitting Model
model_qty_monthly = Prophet()
model_qty_monthly.fit(java_monthly_qty)

model_grosslr_monthly = Prophet()
model_grosslr_monthly.fit(java_monthly_grosslr)

model_netlr_monthly = Prophet()
model_netlr_monthly.fit(java_monthly_netlr)

model_fr_monthly = Prophet()
model_fr_monthly.fit(java_monthly_fr)

# 3. Forecasting (make future dataframe and then predict)
future_qty_monthly = model_qty_monthly.make_future_dataframe(periods=12, freq='M')
future_grosslr_monthly = model_grosslr_monthly.make_future_dataframe(periods=12, freq='M')
future_netlr_monthly = model_netlr_monthly.make_future_dataframe(periods=12, freq='M')
future_fr_monthly = model_fr_monthly.make_future_dataframe(periods=12, freq='M')

forecast_qty_monthly = model_qty_monthly.predict(future_qty_monthly)
forecast_grosslr_monthly = model_grosslr_monthly.predict(future_grosslr_monthly)
forecast_netlr_monthly = model_netlr_monthly.predict(future_netlr_monthly)
forecast_fr_monthly = model_fr_monthly.predict(future_fr_monthly)

# 4. Visualize the model

plot_java_qty = plot_plotly(model_qty_monthly, forecast_qty_monthly)
plot_java_qty_components = plot_components_plotly(model_qty_monthly, forecast_qty_monthly)
plot_java_grosslr = plot_plotly(model_grosslr_monthly, forecast_grosslr_monthly)
plot_java_grosslr_components = plot_components_plotly(model_grosslr_monthly, forecast_grosslr_monthly)
plot_java_netlr = plot_plotly(model_netlr_monthly, forecast_netlr_monthly)
plot_java_netlr_components = plot_components_plotly(model_netlr_monthly, forecast_netlr_monthly)
plot_java_fr = plot_plotly(model_fr_monthly, forecast_fr_monthly)
plot_java_fr_components = plot_components_plotly(model_fr_monthly, forecast_fr_monthly)

# Sumatra

# 1. Prepare dataframe contains ds and y columns
sumatra_monthly_qty = sumatra_monthly[['ATC', 'Quantity']].rename(columns={'ATC':'ds','Quantity':'y'})
sumatra_monthly_grosslr = sumatra_monthly[['ATC', 'Gross Loading Rate']].rename(columns={'ATC':'ds','Gross Loading Rate':'y'})
sumatra_monthly_netlr = sumatra_monthly[['ATC', 'Net Loading Rate']].rename(columns={'ATC':'ds','Net Loading Rate':'y'})
sumatra_monthly_fr = sumatra_monthly[['ATC', 'Fuel Ratio']].rename(columns={'ATC':'ds','Fuel Ratio':'y'})

# 2. Fitting Model
model_qty_monthly = Prophet()
model_qty_monthly.fit(sumatra_monthly_qty)

model_grosslr_monthly = Prophet()
model_grosslr_monthly.fit(sumatra_monthly_grosslr)

model_netlr_monthly = Prophet()
model_netlr_monthly.fit(sumatra_monthly_netlr)

model_fr_monthly = Prophet()
model_fr_monthly.fit(sumatra_monthly_fr)

# 3. Forecasting (make future dataframe and then predict)
future_qty_monthly = model_qty_monthly.make_future_dataframe(periods=12, freq='M')
future_grosslr_monthly = model_grosslr_monthly.make_future_dataframe(periods=12, freq='M')
future_netlr_monthly = model_netlr_monthly.make_future_dataframe(periods=12, freq='M')
future_fr_monthly = model_fr_monthly.make_future_dataframe(periods=12, freq='M')

forecast_qty_monthly = model_qty_monthly.predict(future_qty_monthly)
forecast_grosslr_monthly = model_grosslr_monthly.predict(future_grosslr_monthly)
forecast_netlr_monthly = model_netlr_monthly.predict(future_netlr_monthly)
forecast_fr_monthly = model_fr_monthly.predict(future_fr_monthly)

# 4. Visualize the model
plot_sumatra_qty = plot_plotly(model_qty_monthly, forecast_qty_monthly)
plot_sumatra_qty_components = plot_components_plotly(model_qty_monthly, forecast_qty_monthly)
plot_sumatra_grosslr = plot_plotly(model_grosslr_monthly, forecast_grosslr_monthly)
plot_sumatra_grosslr_components = plot_components_plotly(model_grosslr_monthly, forecast_grosslr_monthly)
plot_sumatra_netlr = plot_plotly(model_netlr_monthly, forecast_netlr_monthly)
plot_sumatra_netlr_components = plot_components_plotly(model_netlr_monthly, forecast_netlr_monthly)
plot_sumatra_fr = plot_plotly(model_fr_monthly, forecast_fr_monthly)
plot_sumatra_fr_components = plot_components_plotly(model_fr_monthly, forecast_fr_monthly)

## -----LAYOUT-----
app.layout = html.Div(children=[
    navbar,
    
    html.Br(),

    ## --Component Main Page--
    html.Div([
        ## --ROW1--
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Select Year'),
                    dbc.CardBody(
                        dcc.Dropdown(
                            id = 'select_year',
                            options = list_year,
                            value = '2022',
                            style={'font-size' : '100%'}
                        )
                    )
                ]),
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Select CTS'),
                    dbc.CardBody(
                        dcc.Dropdown(
                            id = 'select_cts',
                            options = list_cts,
                            value = 'Bulk Java',
                            style={'font-size' : '100%'}
                        )
                    )
                ]),
            ], width=4),


            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Select Variable'),
                    dbc.CardBody(
                        dcc.Dropdown(
                            id = 'select_variable',
                            options = list_variable,
                            value = 'Quantity',
                            style={'font-size' : '100%'}
                        )
                    )
                ]),
            ], width=4)
        ]),

        html.Br(),

        ## --ROW2--
        dbc.Row([
            ### --COLUMN1--
            dbc.Col([
                dbc.Card(id='sum_quantity',color='#b8c1ec',className='text-center',inverse=True),
                html.Br(),
                dbc.Card(id='avg_grosslr',color='#b8c1ec',className='text-center',inverse=True),
            ], width=2),
            ### --COLUMN2--
            dbc.Col([
                dbc.Card(id='avg_netlr',color='#b8c1ec',className='text-center',inverse=True),
                html.Br(),
                dbc.Card(id='avg_fr',color='#b8c1ec',className='text-center',inverse=True),
            ], width=2),
            ### --COLUMN3--
            dbc.Col([
                dcc.Graph(id='plot_line')
            ], width=8)
        ]),

    html.P('Time Series Forecasting',
            style={'textAlign': 'center', 
                   'fontSize': 20, 
                   'background-color':'#242947',
                   'color':'white',
                   'font-family':'Arial'}
            ),

        ## --ROW3--
        dbc.Row([
            ### --COLUMN1--
            dbc.Col([

                dcc.Graph(id='plot_forecast')
     
            ],width=6),

            ### --COLUMN2--
            dbc.Col([

                dcc.Graph(id='plot_components')

            ],width=6,
            ),
        ]) 
    ], style={
        'paddingLeft':'30px',
        'paddingRight':'30px',
    }),

    html.Br(),
    html.Footer('ABL',
            style={'textAlign': 'center', 
                   'fontSize': 20, 
                   'background-color':'#242947',
                   'color':'white'})
])

##-- Callback Sum Quantity
@app.callback(
    Output(component_id='sum_quantity', component_property='children'),
    Input(component_id='select_cts', component_property='value'),
    Input(component_id='select_year', component_property='value')
)

def update_sum_quantity(cts, year):
    if cts == 'Bulk Java':
        df = java_yearly
    elif cts == 'Bulk Sumatra':
        df = sumatra_yearly
    num = df[df['ATC']==int(year)]['Quantity']
    sum_quantity = [
        dbc.CardHeader(f'Sum of Quantity on {cts} in {year}'),
        dbc.CardBody([
            html.H1(num)
            # html.H1(print(f"{num:,}"))
            ]),
    ]
    return sum_quantity

##-- Callback Avg Gross Loading Rate
@app.callback(
    Output(component_id='avg_grosslr', component_property='children'),
    Input(component_id='select_cts', component_property='value'),
    Input(component_id='select_year', component_property='value')
)

def update_avg_grosslr(cts, year):
    if cts == 'Bulk Java':
        df = java_yearly
    elif cts == 'Bulk Sumatra':
        df = sumatra_yearly
    num = df[df['ATC']==int(year)]['Gross Loading Rate']
    avg_grosslr = [
        dbc.CardHeader(f'Average of Gross LR on {cts} in {year} '),
        dbc.CardBody([
            html.H1((round(num,2)))
            # html.H1(print(f"{num:,}"))
            ]),
    ]
    return avg_grosslr

##-- Callback Avg Net Loading Rate
@app.callback(
    Output(component_id='avg_netlr', component_property='children'),
    Input(component_id='select_cts', component_property='value'),
    Input(component_id='select_year', component_property='value')
)

def update_avg_netlr(cts, year):
    if cts == 'Bulk Java':
        df = java_yearly
    elif cts == 'Bulk Sumatra':
        df = sumatra_yearly
    num = df[df['ATC']==int(year)]['Net Loading Rate']
    avg_netlr = [
        dbc.CardHeader(f'Average of Net LR on {cts} in {year}'),
        dbc.CardBody([
            html.H1(round(num,2))
            # html.H1(print(f"{num:,}"))
            ]),
    ]
    return avg_netlr

##-- Callback Avg Fuel Ratio
@app.callback(
    Output(component_id='avg_fr', component_property='children'),
    Input(component_id='select_cts', component_property='value'),
    Input(component_id='select_year', component_property='value')
)

def update_avg_fr(cts, year):
    if cts == 'Bulk Java':
        df = java_yearly
    elif cts == 'Bulk Sumatra':
        df = sumatra_yearly
    num = df[df['ATC']==int(year)]['Fuel Ratio']
    avg_fr = [
        dbc.CardHeader(f'Average of Fuel Ratio on {cts} in {year}'),
        dbc.CardBody([
            html.H1(round(num,2))
            ]),
    ]
    return avg_fr

##-- Callback Plot Line
@app.callback(
    Output(component_id='plot_line', component_property='figure'),
    Input(component_id='select_cts', component_property='value'),
    Input(component_id='select_variable', component_property='value')
)

def update_line(cts, variable):
    # Data aggregation
    if cts == 'Bulk Java':
        df_line = java_monthly
    elif cts == 'Bulk Sumatra':
        df_line = sumatra_monthly
    
    df_line['Year'] = df_line['ATC'].dt.year
    df_line['Month'] = df_line['ATC'].dt.month

    plot_line = px.line(df_line, x='Month', y=variable, color='Year')
    plot_line.update_layout(title=f'{variable} Line Plot on {cts}')
    return plot_line

##-- Callback Plot Forecast
@app.callback(
    Output(component_id='plot_forecast', component_property='figure'),
    Input(component_id='select_cts', component_property='value'),
    Input(component_id='select_variable', component_property='value')
)

def update_forecast(cts, variable):
    # Data aggregation
    if cts == 'Bulk Java' and variable == 'Quantity':
        plot_forecast = plot_java_qty
    elif cts == 'Bulk Java' and variable == 'Gross Loading Rate':
        plot_forecast = plot_java_grosslr
    elif cts == 'Bulk Java' and variable == 'Net Loading Rate':
        plot_forecast = plot_java_netlr
    elif cts == 'Bulk Java' and variable == 'Fuel Ratio':
        plot_forecast = plot_java_fr
    elif cts == 'Bulk Sumatra' and variable == 'Quantity':
        plot_forecast = plot_sumatra_qty
    elif cts == 'Bulk Sumatra' and variable == 'Gross Loading Rate':
        plot_forecast = plot_sumatra_grosslr
    elif cts == 'Bulk Sumatra' and variable == 'Net Loading Rate':
        plot_forecast = plot_sumatra_netlr
    elif cts == 'Bulk Sumatra' and variable == 'Fuel Ratio':
        plot_forecast = plot_sumatra_fr
    plot_forecast.update_layout(autosize=False, width=740, height=500,
                                title=f'{variable} Forecasting on {cts}')
    return plot_forecast


##-- Callback Plot Components
@app.callback(
    Output(component_id='plot_components', component_property='figure'),
    Input(component_id='select_cts', component_property='value'),
    Input(component_id='select_variable', component_property='value')
)

def update_components(cts, variable):
    # Data aggregation
    if cts == 'Bulk Java' and variable == 'Quantity':
        plot_components = plot_java_qty_components
    elif cts == 'Bulk Java' and variable == 'Gross Loading Rate':
        plot_components = plot_java_grosslr_components
    elif cts == 'Bulk Java' and variable == 'Net Loading Rate':
        plot_components = plot_java_netlr_components
    elif cts == 'Bulk Java' and variable == 'Fuel Ratio':
        plot_components = plot_java_fr_components
    elif cts == 'Bulk Sumatra' and variable == 'Quantity':
        plot_components = plot_sumatra_qty_components
    elif cts == 'Bulk Sumatra' and variable == 'Gross Loading Rate':
        plot_components = plot_sumatra_grosslr_components
    elif cts == 'Bulk Sumatra' and variable == 'Net Loading Rate':
        plot_components = plot_sumatra_netlr_components
    elif cts == 'Bulk Sumatra' and variable == 'Fuel Ratio':
        plot_components = plot_sumatra_fr_components
    plot_components.update_layout(autosize=False, width=740, height=500,
                                  title=f'{variable} Forecasting Components on {cts}')
    return plot_components

######-----Start the Dash server-----#####
if __name__ == "__main__":
    app.run_server()