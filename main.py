import dash
from dash_extensions.enrich import Input, Output, State, Trigger, FileSystemCache
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import re
import time

import statistics as st
import numpy as np
import pandas as pd
from pandas.core.indexes.multi import MultiIndex
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pL = dict()
pL['Work Income'] = st.NormalDist(75,5)
pL['Pension Income'] = st.NormalDist(10, 1)
pL['Fixed Expenses'] = st.NormalDist(30,2.5)
pL['Discretionary Expenses'] = st.NormalDist(0.1,0.05)
pL['Annual Raise'] = st.NormalDist(0.03, 0.005)
pL['Investment return'] = st.NormalDist(0.04, 0.005)
pL['Inflation'] = st.NormalDist(0.02, 0.01)
# Distributions for annual variations
pS = dict()
pS['Work Income'] = st.NormalDist(0, 10)
pS['Fixed Expenses'] = st.NormalDist(0, 0.05)
pS['Annual Raise'] = st.NormalDist(0, 0.005)
pS['Discretionary Expenses'] = st.NormalDist(0, 0.05)
pS['Inflation'] = st.NormalDist(0, 0.005)
pS['Investment return'] = st.NormalDist(0, 0.01)

app = dash.Dash('Retirement', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

baseOptions = [{'label': 'Statistics', 'value': 'Statistics'}, {'label': 'All cases', 'value': 'All cases'}]

fig = make_subplots(rows=4, cols=1)
fig.update_layout(template='none', xaxis4=dict(title='Age'), yaxis1=dict(title='Retired (%)', rangemode='tozero'), yaxis2=dict(title='Savings (000s)', rangemode='tozero'), yaxis3=dict(title='Cashflow (000s)', rangemode='tozero'), yaxis4=dict(title='Rates (%)', rangemode='tozero'))

taxRaw = pd.read_excel('tax-bc.xlsx')
taxProc = pd.Series(dtype=float)

for i in range(len(taxRaw)):
    bracketText = str(taxRaw.iloc[i,0])
    if 'first' in bracketText:
        floor = 0
    elif 'up to' in bracketText or 'over' in bracketText:
        floor = float(re.sub('[$,.]', '', re.search('\$\w+,?\w+', bracketText)[0]))/1000
    else:
        continue
    taxProc.loc[floor] = taxRaw.iloc[i,1]

# fsc = FileSystemCache('cache_dir')
# fsc.set('progress', None)
app.layout = html.Div(
    [
        html.H1('Retirement Monte Carlo'),
        html.P('(Need a catchier name...)'),
        html.Hr(),
        html.H2('Inputs'),
        html.P('Enter all dollar values in thousands of dollars, and all percentages as fractions'),

        html.H4('Initial conditions'),
        html.P('What''s your sitch?'),
        dash_table.DataTable(
            id='init',
            columns=[dict(name='Current savings', id='iSavings', type='numeric'), dict(name='Current age', id='iAge', type='numeric'), dict(name='Life expectancy', id='fAge', type='numeric')],
            data=[dict(iSavings=50, iAge=35, fAge=90)],
            editable=True,
            fill_width=False
        ),
        html.Ul([
            html.Li('Life expectancy should be the average value for your demographic (not the oldest possible age you can imagine)', style={'margin':'0'}),
            html.Li('Tax shelters like RRSPs are not currently modelled, so for now it''s best to reduce RRSP savings by your typical tax rate (e.g. include only 70%)', style={'margin':'0'}),
        ]),

        html.H4('Average case estimates'),
        html.P('What do you think the long term averages will be for these parameters?'),
        dash_table.DataTable(
            id='means',
            columns=[{'name': key, 'id': key, 'type':'numeric'} for key in pL.keys()],
            data=[{key: value.mean for key, value in pL.items()}],
            editable=True,
            fill_width=False
        ),
        html.Ul([
            html.Li('Dollar values should be given in current dollars (inflation and annual raises will be accounted for automatically)', style={'margin':'0'}),
            html.Li('Fixed expenses are what you will have to spend each year regardless of your income', style={'margin':'0'}),
            html.Li('Discretionary expenses are calculated as proprtion of your income', style={'margin':'0'}),
            html.Li('Discretionary expenses are calculated as proprtion of your income', style={'margin':'0'}),
        ]),

        html.H4('Uncertainty estimates'),
        html.P('How much variation could there might be from the average case?'),
        dash_table.DataTable(
            id='stds',
            columns=[{'name': key, 'id': key} for key in pL.keys()],
            data=[{key: 2*value.stdev for key, value in pL.items()}],
            editable=True,
            fill_width=False
        ),
        html.Ul([
            html.Li('95% of all possible outcomes should be within average +/- uncertainty', style={'margin':'0'}),
        ]),

        html.Hr(),
        html.H2('Run analysis'),
        html.P('Click RUN when you''re ready to see the results!'),
        html.Table(
            html.Tr([
                html.Td('Cases to run: ', style={'border-bottom':0}),
                html.Td(
                    dcc.Dropdown(
                        id='nCases', value=10, options=[dict(label=f'{n}', value=n) for n in [1, 10, 50, 100]],
                        multi=False, searchable=False, clearable=False,
                        style={'width':100}
                    ),
                    style={'border-bottom':0}),
                # html.Td(html.Button('Run', id='run'), style={'border-bottom':0}),
                html.Td(
                    html.Button('Run', id='run'),
                    style={'border-bottom':0}
                ),
                html.Td(
                    dcc.Loading(
                        id='loading-1',
                        type='default',
                        children=html.Div(id='loading')
                    ),
                    style={'border-bottom':0}
                ),
                                    
            ],
            ),
        ),
        html.Ul([
            html.Li('More cases give better accuracy, but take longer to compute', style={'margin':'0'}),
        ]),

        html.Hr(),
        html.H2('Results'),
        html.P('You can choose to view the overall statistics, all cases, or invidiual cases (ordered from best to worst)'),
        dcc.Dropdown(
            id='select-case', value='Statistics', options=baseOptions,
            multi=False, searchable=False, clearable=False,
            style=dict(width=200)
        ),

        dcc.Graph(id='graph', figure=fig, style=dict(height=1200)),

        dcc.Interval(id='interval', interval=500, disabled=True),
        # dcc.Store(id='progress'),
    ],
    style=dict(width=1024, height=4000, margin='auto')
)

# @app.callback(Output('interval', 'disabled'), Input('run', 'n_clicks'))
# def on_run(run):
#     raise PreventUpdate
#     print('test')
#     return False

# @app.callback(Output('interval', 'disabled'), Input('graph', 'figure'))
# def on_done(done):
#     return True

@app.callback([Output('graph', 'figure'), Output('select-case', 'options'), Output('loading', 'children')], [Input('run', 'value'), Input('run', 'n_clicks'), Input('select-case', 'value')], [State('nCases', 'value'), State('init', 'data'), State('means', 'data'), State('stds', 'data')])
def update_graph(run, clicks, select, nCases, init, means, stds):
    cases = range(0, nCases)
    trigger = dash.callback_context.triggered[0]['prop_id']
    if trigger == 'run.n_clicks' or trigger == '.':   
        # Initial conditions
        iAge = init[0]['iAge']
        fAge = init[0]['fAge']
        iSav = init[0]['iSavings']

        pL = {key: st.NormalDist(mean, std/2) for key, mean, std in zip(means[0].keys(), means[0].values(), stds[0].values())}

        data = pd.Series(dtype=float, index=pd.MultiIndex.from_product([cases, range(iAge, fAge), ['Savings', 'Income', 'IncomeTaxed', 'Interest', 'Expenses', 'Tax', 'Raise', 'Return', 'Inflation', 'Retired']], names=['Case', 'Age', 'Value']))
        rAges = pd.Series(dtype=float, index=cases)
        for c in cases:
            cpL = sample(pL, c+1)
            cpS = {age: sample(pS, (c+1)*age) for age in range(iAge, fAge)}
            # Binary search for earliest retirement age
            rAgeMin = iAge; rAgeMax = fAge
            while rAgeMax - rAgeMin > 1:
                rAge = np.floor((rAgeMin + rAgeMax)/2)
                success = sim(iAge, fAge, iSav, c, cpL, cpS, rAge)
                if success:
                    rAgeMax = rAge
                else:
                    rAgeMin = rAge
                # print(rAge, success)
            success = sim(iAge, fAge, iSav, c, cpL, cpS, rAgeMax, data)
            rAges[c] = rAgeMax
            print(f'Case: {c} | Age: {rAgeMax}')
            # fsc.set('progress', f'{(c+1)/nCases*100:.0f}%')

        # fsc.set('progress', f'')

        i = rAges.sort_values().index

        mean = data.unstack('Value').mean(level='Age')
        std = data.unstack('Value').std(level='Age')
        
        fig.data = []
        plot(fig, 'Statistics', mean, std, True)
        for c in cases:
            case = data[i[c]].unstack('Value')
            plot(fig, str(c), case)

    for trace in fig.data:
        if select == trace.uid:
            trace.visible = True
        elif select == 'All cases' and re.match('\d+', trace.uid):
            trace.visible = True
        else:
            trace.visible = False

    options = baseOptions + [dict(label=f'Case {c}', value=f'{c}') for c in cases]

    return fig, options, run

def sim(iAge, fAge, iSav, c, cpL, cpS, rAge, data=None):
    success = True
    sav = iSav
    inc = cpL['Work Income']
    pen = cpL['Pension Income']
    incMax = inc
    expMin = cpL['Fixed Expenses']
    ret = cpL['Investment return']
    inv = ret*sav
    inv =  0
    bpa = 10
    for age in range(iAge, fAge):
        dInc = cpL['Annual Raise'] + cpS[age]['Annual Raise']
        inf = cpL['Inflation'] + cpS[age]['Inflation']
        ret = cpL['Investment return'] + cpS[age]['Investment return']
        if age < rAge:
            rtd = False
            inc += cpS[age]['Work Income']
            incMax = max(inc, incMax)
            exp = expMin + inc*(cpL['Discretionary Expenses'] + cpS[age]['Discretionary Expenses'])
            inc *= 1+dInc
        else:
            rtd = True
            if age > 65:
                inc = pen
            else:
                inc = 0
            incMax *= 1+inf
            exp = expMin + incMax*(cpL['Discretionary Expenses'] + cpS[age]['Discretionary Expenses'])
        incTax = calcTax(inc+inv, bpa)
        net = inc - incTax - exp
        tax = incTax/(inc+inv) if (inc+inv) > 0 else 0
        if data is not None:
            data[c, age] = [sav, inc, inc-incTax, inv, exp, tax, dInc, ret, inf, rtd]
        inv = ret*sav
        sav += net + inv
        expMin *= 1+inf
        pen *= 1+inf
        bpa *= 1+inf
        if sav < 0:
            success = False
            break
    return success

def plot(fig, uid, data, conf=None, showlegend=False):

    color = 'rgba(0,100,80,1)'
    fillcolor = 'rgba(0,100,80,0.2)'

    fig.add_trace(go.Scatter(x=data.index, y=data['Retired']*100, line_color='red', name='Retired %', uid=uid, fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', showlegend=showlegend, legendgroup='Retirement'), row=1, col=1)

    if type(conf) == pd.DataFrame:
        fig.add_trace(go.Scatter(x=data.index, y=data['Savings']-conf['Savings'], line_color='green', line=dict(width=0), name='-Conf', uid=uid, showlegend=False, legendgroup='Savings'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['Savings']+conf['Savings'], line_color='green', line=dict(width=0), name='+Conf', uid=uid, fill='tonexty', fillcolor=fillcolor, showlegend=False, legendgroup='Savings'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Savings'], line_color='green', name='Savings', uid=uid, showlegend=showlegend, legendgroup='Savings'), row=2, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['Income'], line_color='black', name='Income', uid=uid, showlegend=showlegend, legendgroup='Cashflow'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['IncomeTaxed'], line_color='grey', name='After-tax', uid=uid, showlegend=showlegend, legendgroup='Cashflow'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Expenses'], line_color='red', name='Expenses', uid=uid, showlegend=showlegend, legendgroup='Cashflow'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Interest'], line_color='blue', name='Interest', uid=uid, showlegend=showlegend, legendgroup='Cashflow'), row=3, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['Raise']*100, line_color='green', name='Raise', uid=uid, showlegend=showlegend, legendgroup='Rates'), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Return']*100, line_color='black', name='Return', uid=uid, showlegend=showlegend, legendgroup='Rates'), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Inflation']*100, line_color='blue',  name='Inflation', uid=uid, showlegend=showlegend, legendgroup='Rates'), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Tax']*100, line_color='red', name='Tax', uid=uid, showlegend=showlegend, legendgroup='Rates'), row=4, col=1)        
    # fig.add_trace(go.Scatter(x=rAges, y=rSavings, mode='markers', line_color='green', name='Case retirement\nage/savings'), row=3, col=1)

# @app.callback(Output('progress', 'children'), Trigger('interval', 'n_intervals'))
# def update_progress(trigger):
#     value = fsc.get('progress')  # get progress
#     if value is None:
#         raise PreventUpdate
#     return fsc.get('progress')

def calcTax(inc, bpa):
    taxInc = inc - bpa
    tax = 0
    for i in reversed(range(0, len(taxProc))):
        tax += taxProc.iloc[i] * max(0, (taxInc - bpa - taxProc.index[i]))
        taxInc -= tax
    return tax

def sample(d, seed):
    dSamp = dict()
    for dk in d.keys():
        dSamp[dk] = d[dk].samples(1, seed=seed)[0]
        seed += 1
    return dSamp

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8080, debug=True, use_reloader=False)