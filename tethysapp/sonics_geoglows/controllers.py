from tethys_sdk.gizmos import *
from django.shortcuts import render
from tethys_sdk.gizmos import PlotlyView
from django.http import HttpResponse, JsonResponse

import io
import os
import sys
import json
import math
import requests
import geoglows
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import datetime as dt
from glob import glob
from lmoments3 import distr
import plotly.graph_objs as go
from .app import SonicsGeoglows as app

import time

def home(request):
    """
    Controller for the app home page.
    """

    folder = app.get_custom_setting('folder')
    forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")))

    dates_array = []

    for file in forecast_nc_list:
        dates_array.append(file[len(folder) + 1 + 23:-3])

    dates = []

    for date in dates_array:
        date_f = dt.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8])).strftime('%Y-%m-%d')
        dates.append([date_f, date])

    dates.append(['Select Date', dates[-1][1]])
    dates.reverse()

    date_picker = DatePicker(name='datesSelect',
                             display_text='Date',
                             autoclose=True,
                             format='yyyy-mm-dd',
                             start_date=dates[-1][0],
                             end_date=dates[1][0],
                             start_view='month',
                             today_button=True,
                             initial='')

    region_index = json.load(open(os.path.join(os.path.dirname(__file__), 'public', 'geojson', 'index.json')))
    regions = SelectInput(
        display_text='Zoom to a Region:',
        name='regions',
        multiple=False,
        original=True,
        options=[(region_index[opt]['name'], opt) for opt in region_index]
    )

    context = {
        "date_picker": date_picker,
        "regions": regions
    }

    return render(request, 'sonics_geoglows/home.html', context)


def get_popup_response(request):
    """
    Get simulated data from api
    """
    start_time = time.time()

    observed_data_path_file = os.path.join(app.get_app_workspace().path, 'observed_data.json')
    simulated_data_path_file = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
    corrected_data_path_file = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
    forecast_data_path_file = os.path.join(app.get_app_workspace().path, 'forecast_data.json')

    f = open(observed_data_path_file, 'w')
    f.close()
    f2 = open(simulated_data_path_file, 'w')
    f2.close()
    f3 = open(corrected_data_path_file, 'w')
    f3.close()
    f4 = open(forecast_data_path_file, 'w')
    f4.close()

    return_obj = {}

    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']

        '''Get Observed Data'''
        folder = app.get_custom_setting('folder')
        forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
        nc_file = forecast_nc_list[0]

        qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr
        time_dataset = qout_datasets.time

        historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Observed Streamflow'])
        historical_simulation_df.index.name = 'Datetime'

        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')

        historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
        historical_simulation_df.index = historical_simulation_df.index.to_series().dt.strftime("%Y-%m-%d")
        historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
        historical_simulation_df.index.name = 'datetime'
        historical_simulation_df.to_json(observed_data_file_path,orient='columns')


        '''Get Simulated Data'''
        simulated_df = geoglows.streamflow.historic_simulation(comid, forcing='era_5', return_format='csv')
        # Removing Negative Values
        simulated_df[simulated_df < 0] = 0
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.index = simulated_df.index.to_series().dt.strftime("%Y-%m-%d")
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df = pd.DataFrame(data=simulated_df.iloc[:, 0].values, index=simulated_df.index, columns=['Simulated Streamflow'])

        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df.reset_index(level=0, inplace=True)
        simulated_df['datetime'] = simulated_df['datetime'].dt.strftime('%Y-%m-%d')
        simulated_df.set_index('datetime', inplace=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.index.name = 'Datetime'
        simulated_df.to_json(simulated_data_file_path)

        print("finished get_popup_response")

        print("--- %s seconds getpopup ---" % (time.time() - start_time))

        return JsonResponse({})

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: "+ str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
                'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })


def gve_1(loc: float, scale: float, shape: float, rp: int or float) -> float:
    """
    Solves the Gumbel Type I probability distribution function (pdf) = exp(-exp(-b)) where b is the covariate. Provide
    the standard deviation and mean of the list of annual maximum flows. Compare scipy.stats.gumbel_r
    Args:
      std (float): the standard deviation of the series
      xbar (float): the mean of the series
      skew (float): the skewness of the series
      rp (int or float): the return period in years
    Returns:
      float, the flow corresponding to the return period specified
    """

    return ((scale / shape) * (1 - math.exp(shape * (math.log(-math.log(1 - (1 / rp))))))) + loc


def get_hydrographs(request):

    start_time = time.time()

    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        historical_simulation_df = pd.read_json(observed_data_file_path, convert_dates=True)
        historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
        historical_simulation_df.sort_index(inplace=True, ascending=True)

        '''Get Simulated Data'''
        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df = pd.read_json(simulated_data_file_path, convert_dates=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.sort_index(inplace=True, ascending=True)

        '''Correct the Bias in Simulation'''
        corrected_df = geoglows.bias.correct_historical(simulated_df, historical_simulation_df)
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df.reset_index(level=0, inplace=True)
        corrected_df['index'] = corrected_df['index'].dt.strftime('%Y-%m-%d')
        corrected_df.set_index('index', inplace=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.index.name = 'Datetime'
        corrected_df.to_json(corrected_data_file_path)

        '''Plotting hydrograph'''
        observed_Q = go.Scatter(x=historical_simulation_df.index, y=historical_simulation_df.iloc[:, 0].values, name='SONICS', )
        simulated_Q = go.Scatter(x=simulated_df.index, y=simulated_df.iloc[:, 0].values, name='GEOGloWS', )
        corrected_Q = go.Scatter(x=corrected_df.index, y=corrected_df.iloc[:, 0].values, name='Corrected GEOGloWS', )

        layout = go.Layout(
	        title='Simulated Streamflow at {0}'.format(comid), xaxis=dict(title='Dates',),
	        yaxis=dict(title='Streamflow (m<sup>3</sup>/s)', autorange=True), showlegend=True)

        chart_obj = PlotlyView(go.Figure(data=[observed_Q, simulated_Q, corrected_Q], layout=layout))

        context = {
            'gizmo_object': chart_obj,
        }

        return render(request, 'sonics_geoglows/gizmo_ajax.html', context)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })


def get_observed_discharge_csv(request):
    """
    Get historic simulations from ERA Interim
    """

    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        historical_simulation_df = pd.read_json(observed_data_file_path, convert_dates=True)
        historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
        historical_simulation_df.sort_index(inplace=True, ascending=True)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=simulated_sonics_{0}.csv'.format(comid)

        historical_simulation_df.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })


def get_simulated_discharge_csv(request):
    """
    Get historic simulations from ERA Interim
    """

    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']

        '''Get Simulated Data'''
        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df = pd.read_json(simulated_data_file_path, convert_dates=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.sort_index(inplace=True, ascending=True)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=simulated_geoglows_{0}.csv'.format(comid)

        simulated_df.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_simulated_bc_discharge_csv(request):
    """
    Get historic simulations from ERA Interim
    """

    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path, convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=corrected_geoglows_{0}.csv'.format(comid)

        corrected_df.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_time_series(request):
    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']
        startdate = get_data['startdate']

        '''Get Simulated Data'''
        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df = pd.read_json(simulated_data_file_path, convert_dates=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.sort_index(inplace=True, ascending=True)

        folder = app.get_custom_setting('folder')

        '''Getting Forecast Stats'''
        if startdate != '':
            nc_file = folder + '/PISCO_HyD_ARNOVIC_v1.0_' + startdate + '.nc'
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr
            time_dataset = qout_datasets.time
            historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            initial_condition = historical_simulation_df.loc[historical_simulation_df.index == pd.to_datetime(historical_simulation_df.index[-1])]
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
            historical_simulation_df.index = historical_simulation_df.index.to_series().dt.strftime("%Y-%m-%d")
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)

            '''ETA Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_eta
            forecast_eta_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_eta_df.index.name = 'Datetime'
            forecast_eta_df = forecast_eta_df.append(initial_condition)
            forecast_eta_df.sort_index(inplace=True)
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)
            forecast_eta_df.index = forecast_eta_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)

            '''GFS Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_gfs
            forecast_gfs_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_gfs_df.index.name = 'Datetime'
            forecast_gfs_df = forecast_gfs_df.append(initial_condition)
            forecast_gfs_df.sort_index(inplace=True)
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)
            forecast_gfs_df.index = forecast_gfs_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)

            '''GEOGloWs Forecast'''
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastStats/?reach_id=' + comid + '&date=' + startdate + '&return_format=csv', verify=False).content

        else:
            forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
            nc_file = forecast_nc_list[0]
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr
            time_dataset = qout_datasets.time
            historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            initial_condition = historical_simulation_df.loc[historical_simulation_df.index == pd.to_datetime(historical_simulation_df.index[-1])]
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
            historical_simulation_df.index = historical_simulation_df.index.to_series().dt.strftime("%Y-%m-%d")
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)

            '''ETA Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_eta
            forecast_eta_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_eta_df.index.name = 'Datetime'
            forecast_eta_df = forecast_eta_df.append(initial_condition)
            forecast_eta_df.sort_index(inplace=True)
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)
            forecast_eta_df.index = forecast_eta_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)

            '''GFS Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_gfs
            forecast_gfs_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_gfs_df.index.name = 'Datetime'
            forecast_gfs_df = forecast_gfs_df.append(initial_condition)
            forecast_gfs_df.sort_index(inplace=True)
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)
            forecast_gfs_df.index = forecast_gfs_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)

            '''GEOGloWs Forecast'''
            date = nc_file[len(folder) + 1 + 23:-3]
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastStats/?reach_id=' + comid + '&date=' + date + '&return_format=csv', verify=False).content

        '''GEOGloWs Forecast'''
        stats_df = pd.read_csv(io.StringIO(res.decode('utf-8')), index_col=0)
        stats_df.index = pd.to_datetime(stats_df.index)
        stats_df[stats_df < 0] = 0
        stats_df.index = stats_df.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        stats_df.index = pd.to_datetime(stats_df.index)

        records = geoglows.streamflow.forecast_records(comid)
        records.index = pd.to_datetime(records.index)
        records[records < 0] = 0
        records.index = records.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        records.index = pd.to_datetime(records.index)

        '''Return Periods SONICS'''
        max_annual_flow = historical_simulation_df.groupby(historical_simulation_df.index.strftime("%Y")).max()
        params = distr.gev.lmom_fit(max_annual_flow.iloc[:, 0].values.tolist())

        return_periods = [10, 5, 2.33]

        return_periods_values = []

        for rp in return_periods:
            return_periods_values.append(gve_1(params['loc'], params['scale'], params['c'], rp))

        d = {'rivid': [comid], 'return_period_10': [return_periods_values[0]],
             'return_period_5': [return_periods_values[1]], 'return_period_2_33': [return_periods_values[2]]}

        rperiods_sonics = pd.DataFrame(data=d)
        rperiods_sonics.set_index('rivid', inplace=True)

        '''Return Periods GEOGloWS'''
        max_annual_flow = simulated_df.groupby(simulated_df.index.strftime("%Y")).max()
        params = distr.gev.lmom_fit(max_annual_flow.iloc[:, 0].values.tolist())

        for rp in return_periods:
            return_periods_values.append(gve_1(params['loc'], params['scale'], params['c'], rp))

        d = {'rivid': [comid], 'return_period_10': [return_periods_values[0]],
             'return_period_5': [return_periods_values[1]], 'return_period_2_33': [return_periods_values[2]]}

        rperiods_geoglows = pd.DataFrame(data=d)
        rperiods_geoglows.set_index('rivid', inplace=True)

        '''Plotting Forecast'''

        titles = {'Reach ID': comid}
        hydroviewer_figure = geoglows.plots.forecast_stats(stats=stats_df, titles=titles)

        x_vals = (stats_df.index[0], stats_df.index[len(stats_df.index) - 1], stats_df.index[len(stats_df.index) - 1], stats_df.index[0])
        max_visible = max(stats_df.max())

        record_plot = records.copy()
        record_plot = record_plot.loc[record_plot.index >= pd.to_datetime(stats_df.index[0] - dt.timedelta(days=8))]
        record_plot = record_plot.loc[record_plot.index <= pd.to_datetime(stats_df.index[0] + dt.timedelta(days=2))]

        if len(records.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='1st days forecasts',
                x=record_plot.index,
                y=record_plot.iloc[:, 0].values,
                line=dict(
                    color='#FFA15A',
                )
            ))

            x_vals = (records.index[0], stats_df.index[len(stats_df.index) - 1], stats_df.index[len(stats_df.index) - 1], records.index[0])
            max_visible = max(max(records.max()), max_visible)

        '''Adding Recent Days'''
        records_df = historical_simulation_df.loc[historical_simulation_df.index >= pd.to_datetime(historical_simulation_df.index[-1] - dt.timedelta(days=8))]
        records_df = records_df.loc[records_df.index <= pd.to_datetime(historical_simulation_df.index[-1] + dt.timedelta(days=2))]

        if len(records_df.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='SONICS',
                x=records_df.index,
                y=records_df.iloc[:, 0].values,
                line=dict(color='green', )
            ))

            x_vals = (records_df.index[0], stats_df.index[len(stats_df.index) - 1], stats_df.index[len(stats_df.index) - 1],records_df.index[0])
            max_visible = max(max(records_df.max()), max_visible)

        '''SONICS Forecaast'''
        if len(forecast_gfs_df.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='GFS Forecast',
                x=forecast_gfs_df.index,
                y=forecast_gfs_df['Streamflow (m3/s)'],
                showlegend=True,
                line=dict(color='black', dash='dash')
            ))

            max_visible = max(max(forecast_gfs_df.max()), max_visible)

        if len(forecast_eta_df.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='ETA Forecast',
                x=forecast_eta_df.index,
                y=forecast_eta_df['Streamflow (m3/s)'],
                showlegend=True,
                line=dict(color='blue', dash='dash')
            ))

            max_visible = max(max(forecast_eta_df.max()), max_visible)

        '''Getting Return Periods'''
        r2_33 = int(rperiods_geoglows.iloc[0]['return_period_2_33'])

        colors = {
            '2.33 Year': 'rgba(243, 255, 0, .4)',
            '5 Year': 'rgba(255, 165, 0, .4)',
            '10 Year': 'rgba(255, 0, 0, .4)',
        }

        if max_visible > r2_33:
            visible = True
            hydroviewer_figure.for_each_trace(
                lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (), )
        else:
            visible = 'legendonly'
            hydroviewer_figure.for_each_trace(
                lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (), )

        def template(name, y, color, fill='toself'):
            return go.Scatter(
                name=name,
                x=x_vals,
                y=y,
                legendgroup='returnperiods',
                fill=fill,
                visible=visible,
                line=dict(color=color, width=0))

        r5 = int(rperiods_geoglows.iloc[0]['return_period_5'])
        r10 = int(rperiods_geoglows.iloc[0]['return_period_10'])

        hydroviewer_figure.add_trace(template('Return Periods', (r10 * 0.05, r10 * 0.05, r10 * 0.05, r10 * 0.05), 'rgba(0,0,0,0)', fill='none'))
        hydroviewer_figure.add_trace(template(f'2.33 Year: {r2_33}', (r2_33, r2_33, r5, r5), colors['2.33 Year']))
        hydroviewer_figure.add_trace(template(f'5 Year: {r5}', (r5, r5, r10, r10), colors['5 Year']))
        hydroviewer_figure.add_trace(template(f'10 Year: {r10}', (r10, r10, max(r10 + r10 * 0.05, max_visible), max(r10 + r10 * 0.05, max_visible)), colors['10 Year']))

        hydroviewer_figure['layout']['xaxis'].update(autorange=True)

        chart_obj = PlotlyView(hydroviewer_figure)

        context = {
            'gizmo_object': chart_obj,
        }

        return render(request, 'sonics_geoglows/gizmo_ajax.html', context)


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })


def get_sonics_forecast_data_csv(request):
    """""
    Returns Forecast data as csv
    """""

    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']
        startdate = get_data['startdate']

        folder = app.get_custom_setting('folder')

        '''Getting Forecast Stats'''
        if startdate != '':
            nc_file = folder + '/PISCO_HyD_ARNOVIC_v1.0_' + startdate + '.nc'
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr
            time_dataset = qout_datasets.time
            historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            initial_condition = historical_simulation_df.loc[historical_simulation_df.index == pd.to_datetime(historical_simulation_df.index[-1])]

            '''ETA Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_eta
            forecast_eta_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_eta_df.index.name = 'Datetime'
            forecast_eta_df = forecast_eta_df.append(initial_condition)
            forecast_eta_df.sort_index(inplace=True)
            forecast_eta_df.rename(columns={"Streamflow (m3/s)": "ETA Streamflow (m3/s)"}, inplace=True)

            '''GFS Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_gfs
            forecast_gfs_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_gfs_df.index.name = 'Datetime'
            forecast_gfs_df = forecast_gfs_df.append(initial_condition)
            forecast_gfs_df.sort_index(inplace=True)
            forecast_gfs_df.rename(columns={"Streamflow (m3/s)": "GFS Streamflow (m3/s)"}, inplace=True)

        else:

            forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
            nc_file = forecast_nc_list[0]
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr
            time_dataset = qout_datasets.time
            historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            initial_condition = historical_simulation_df.loc[historical_simulation_df.index == pd.to_datetime(historical_simulation_df.index[-1])]

            '''ETA Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_eta
            forecast_eta_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_eta_df.index.name = 'Datetime'
            forecast_eta_df = forecast_eta_df.append(initial_condition)
            forecast_eta_df.sort_index(inplace=True)
            forecast_eta_df.rename(columns={"Streamflow (m3/s)": "ETA Streamflow (m3/s)"}, inplace=True)

            '''GFS Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_gfs
            forecast_gfs_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_gfs_df.index.name = 'Datetime'
            forecast_gfs_df = forecast_gfs_df.append(initial_condition)
            forecast_gfs_df.sort_index(inplace=True)
            forecast_gfs_df.rename(columns={"Streamflow (m3/s)": "GFS Streamflow (m3/s)"}, inplace=True)

        forecast_df = pd.concat([forecast_eta_df, forecast_gfs_df], axis=1)

        # Writing CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=sonics_forecast_{0}_{1}.csv'.format(comid, startdate)

        forecast_df.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
                'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })


def get_geoglows_forecast_data_csv(request):
    """""
    Returns Forecast data as csv
    """""

    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']
        startdate = get_data['startdate']

        folder = app.get_custom_setting('folder')

        '''Getting Forecast Stats'''
        if startdate != '':
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id=' + comid + '&date=' + startdate + '&return_format=csv', verify=False).content
        else:
            forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
            nc_file = forecast_nc_list[0]
            date = nc_file[len(folder) + 1 + 23:-3]
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id=' + comid + '&date=' + startdate + '&return_format=csv',verify=False).content

        forecast_ens = pd.read_csv(io.StringIO(res.decode('utf-8')), index_col=0)
        forecast_ens.index = pd.to_datetime(forecast_ens.index)
        forecast_ens[forecast_ens < 0] = 0
        forecast_ens.index = forecast_ens.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast_ens.index = pd.to_datetime(forecast_ens.index)

        # Writing CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=geoglows_forecast_{0}_{1}.csv'.format(comid, startdate)

        forecast_ens.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
                'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_time_series_bc(request):
    try:
        get_data = request.GET
        # get stream attributes
        comid = get_data['comid']
        region = get_data['region']
        subbasin = get_data['subbasin']
        watershed = get_data['watershed']
        startdate = get_data['startdate']

        '''Get Observed Data'''
        observed_data_file_path = os.path.join(app.get_app_workspace().path, 'observed_data.json')
        observed_df = pd.read_json(observed_data_file_path,convert_dates=True)
        observed_df.index = pd.to_datetime(observed_df.index)
        observed_df.sort_index(inplace=True, ascending=True)

        '''Get Simulated Data'''
        simulated_data_file_path = os.path.join(app.get_app_workspace().path, 'simulated_data.json')
        simulated_df = pd.read_json(simulated_data_file_path, convert_dates=True)
        simulated_df.index = pd.to_datetime(simulated_df.index)
        simulated_df.sort_index(inplace=True, ascending=True)

        '''Get Bias Corrected Data'''
        corrected_data_file_path = os.path.join(app.get_app_workspace().path, 'corrected_data.json')
        corrected_df = pd.read_json(corrected_data_file_path,convert_dates=True)
        corrected_df.index = pd.to_datetime(corrected_df.index)
        corrected_df.sort_index(inplace=True, ascending=True)

        folder = app.get_custom_setting('folder')

        '''Getting Forecast Stats'''
        if startdate != '':
            nc_file = folder + '/PISCO_HyD_ARNOVIC_v1.0_' + startdate + '.nc'
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr
            time_dataset = qout_datasets.time
            historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            initial_condition = historical_simulation_df.loc[historical_simulation_df.index == pd.to_datetime(historical_simulation_df.index[-1])]
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
            historical_simulation_df.index = historical_simulation_df.index.to_series().dt.strftime("%Y-%m-%d")
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)

            '''ETA Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_eta
            forecast_eta_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_eta_df.index.name = 'Datetime'
            forecast_eta_df = forecast_eta_df.append(initial_condition)
            forecast_eta_df.sort_index(inplace=True)
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)
            forecast_eta_df.index = forecast_eta_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)

            '''GFS Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_gfs
            forecast_gfs_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_gfs_df.index.name = 'Datetime'
            forecast_gfs_df = forecast_gfs_df.append(initial_condition)
            forecast_gfs_df.sort_index(inplace=True)
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)
            forecast_gfs_df.index = forecast_gfs_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)

            '''GEOGloWs Forecast'''
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id=' + comid + '&date=' + startdate + '&return_format=csv',verify=False).content

        else:
            forecast_nc_list = sorted(glob(os.path.join(folder, "*.nc")), reverse=True)
            nc_file = forecast_nc_list[0]
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr
            time_dataset = qout_datasets.time
            historical_simulation_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            initial_condition = historical_simulation_df.loc[historical_simulation_df.index == pd.to_datetime(historical_simulation_df.index[-1])]
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)
            historical_simulation_df.index = historical_simulation_df.index.to_series().dt.strftime("%Y-%m-%d")
            historical_simulation_df.index = pd.to_datetime(historical_simulation_df.index)

            '''ETA Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_eta
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_eta
            forecast_eta_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_eta_df.index.name = 'Datetime'
            forecast_eta_df = forecast_eta_df.append(initial_condition)
            forecast_eta_df.sort_index(inplace=True)
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)
            forecast_eta_df.index = forecast_eta_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_eta_df.index = pd.to_datetime(forecast_eta_df.index)

            '''GFS Forecast'''
            qout_datasets = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).qr_gfs
            time_dataset = xr.open_dataset(nc_file, autoclose=True).sel(comid=comid).time_gfs
            forecast_gfs_df = pd.DataFrame(qout_datasets.values, index=time_dataset.values, columns=['Streamflow (m3/s)'])
            forecast_gfs_df.index.name = 'Datetime'
            forecast_gfs_df = forecast_gfs_df.append(initial_condition)
            forecast_gfs_df.sort_index(inplace=True)
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)
            forecast_gfs_df.index = forecast_gfs_df.index.to_series().dt.strftime("%Y-%m-%d")
            forecast_gfs_df.index = pd.to_datetime(forecast_gfs_df.index)

            '''GEOGloWs Forecast'''
            date = nc_file[len(folder) + 1 + 23:-3]
            res = requests.get('https://geoglows.ecmwf.int/api/ForecastEnsembles/?reach_id=' + comid + '&date=' + date + '&return_format=csv',verify=False).content

        '''GEOGloWs Forecast'''
        forecast_ens = pd.read_csv(io.StringIO(res.decode('utf-8')), index_col=0)
        forecast_ens.index = pd.to_datetime(forecast_ens.index)
        forecast_ens[forecast_ens < 0] = 0
        forecast_ens.index = forecast_ens.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        forecast_ens.index = pd.to_datetime(forecast_ens.index)

        forecast_ens_file_path = os.path.join(app.get_app_workspace().path, 'forecast_ens.json')
        forecast_ens.index.name = 'Datetime'
        forecast_ens.to_json(forecast_ens_file_path)

        '''Get Forecasts Records'''
        records = geoglows.streamflow.forecast_records(comid)
        records.index = pd.to_datetime(records.index)
        records[records < 0] = 0
        records.index = records.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S")
        records.index = pd.to_datetime(records.index)

        '''Correct Bias Forecasts'''
        monthly_simulated = simulated_df[simulated_df.index.month == (forecast_ens.index[0]).month].dropna()
        monthly_observed = observed_df[observed_df.index.month == (forecast_ens.index[0]).month].dropna()

        min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
        max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())

        min_factor_df = forecast_ens.copy()
        max_factor_df = forecast_ens.copy()
        forecast_ens_df = forecast_ens.copy()

        for column in forecast_ens.columns:
            tmp = forecast_ens[column].dropna().to_frame()
            min_factor = tmp.copy()
            max_factor = tmp.copy()
            min_factor.loc[min_factor[column] >= min_simulated, column] = 1
            min_index_value = min_factor[min_factor[column] != 1].index.tolist()

            for element in min_index_value:
                min_factor[column].loc[min_factor.index == element] = tmp[column].loc[tmp.index == element] / min_simulated

            max_factor.loc[max_factor[column] <= max_simulated, column] = 1
            max_index_value = max_factor[max_factor[column] != 1].index.tolist()

            for element in max_index_value:
                max_factor[column].loc[max_factor.index == element] = tmp[column].loc[tmp.index == element] / max_simulated

            tmp.loc[tmp[column] <= min_simulated, column] = min_simulated
            tmp.loc[tmp[column] >= max_simulated, column] = max_simulated
            forecast_ens_df.update(pd.DataFrame(tmp[column].values, index=tmp.index, columns=[column]))
            min_factor_df.update(pd.DataFrame(min_factor[column].values, index=min_factor.index, columns=[column]))
            max_factor_df.update(pd.DataFrame(max_factor[column].values, index=max_factor.index, columns=[column]))

        corrected_ensembles = geoglows.bias.correct_forecast(forecast_ens_df, simulated_df, observed_df)
        corrected_ensembles = corrected_ensembles.multiply(min_factor_df, axis=0)
        corrected_ensembles = corrected_ensembles.multiply(max_factor_df, axis=0)

        forecast_ens_bc_file_path = os.path.join(app.get_app_workspace().path, 'forecast_ens_bc.json')
        corrected_ensembles.index.name = 'Datetime'
        corrected_ensembles.to_json(forecast_ens_bc_file_path)

        ensemble = corrected_ensembles.copy()
        high_res_df = ensemble['ensemble_52_m^3/s'].to_frame()
        ensemble.drop(columns=['ensemble_52_m^3/s'], inplace=True)
        ensemble.dropna(inplace=True)
        high_res_df.dropna(inplace=True)

        max_df = ensemble.quantile(1.0, axis=1).to_frame()
        max_df.rename(columns={1.0: 'flow_max_m^3/s'}, inplace=True)

        p75_df = ensemble.quantile(0.75, axis=1).to_frame()
        p75_df.rename(columns={0.75: 'flow_75%_m^3/s'}, inplace=True)

        p25_df = ensemble.quantile(0.25, axis=1).to_frame()
        p25_df.rename(columns={0.25: 'flow_25%_m^3/s'}, inplace=True)

        min_df = ensemble.quantile(0, axis=1).to_frame()
        min_df.rename(columns={0.0: 'flow_min_m^3/s'}, inplace=True)

        mean_df = ensemble.mean(axis=1).to_frame()
        mean_df.rename(columns={0: 'flow_avg_m^3/s'}, inplace=True)

        high_res_df.rename(columns={'ensemble_52_m^3/s': 'high_res_m^3/s'}, inplace=True)

        fixed_stats = pd.concat([max_df, p75_df, mean_df, p25_df, min_df, high_res_df], axis=1)

        '''Correct Bias Forecasts Records'''

        date_ini = records.index[0]
        month_ini = date_ini.month

        date_end = records.index[-1]
        month_end = date_end.month

        if month_end < month_ini:
            meses1 = np.arange(month_ini, 13, 1)
            meses2 = np.arange(1, month_end + 1, 1)
            meses = np.concatenate([meses1, meses2])
        else:
            meses = np.arange(month_ini, month_end + 1, 1)

        fixed_records = pd.DataFrame()

        for mes in meses:
            values = records.loc[records.index.month == mes]

            monthly_simulated = simulated_df[simulated_df.index.month == mes].dropna()
            monthly_observed = observed_df[observed_df.index.month == mes].dropna()

            min_simulated = np.min(monthly_simulated.iloc[:, 0].to_list())
            max_simulated = np.max(monthly_simulated.iloc[:, 0].to_list())

            min_factor_records_df = values.copy()
            max_factor_records_df = values.copy()
            fixed_records_df = values.copy()

            column_records = values.columns[0]
            tmp = records[column_records].dropna().to_frame()
            min_factor = tmp.copy()
            max_factor = tmp.copy()
            min_factor.loc[min_factor[column_records] >= min_simulated, column_records] = 1
            min_index_value = min_factor[min_factor[column_records] != 1].index.tolist()

            for element in min_index_value:
                min_factor[column_records].loc[min_factor.index == element] = tmp[column_records].loc[tmp.index == element] / min_simulated

            max_factor.loc[max_factor[column_records] <= max_simulated, column_records] = 1
            max_index_value = max_factor[max_factor[column_records] != 1].index.tolist()

            for element in max_index_value:
                max_factor[column_records].loc[max_factor.index == element] = tmp[column_records].loc[tmp.index == element] / max_simulated

            tmp.loc[tmp[column_records] <= min_simulated, column_records] = min_simulated
            tmp.loc[tmp[column_records] >= max_simulated, column_records] = max_simulated
            fixed_records_df.update(pd.DataFrame(tmp[column_records].values, index=tmp.index, columns=[column_records]))
            min_factor_records_df.update(pd.DataFrame(min_factor[column_records].values, index=min_factor.index, columns=[column_records]))
            max_factor_records_df.update(pd.DataFrame(max_factor[column_records].values, index=max_factor.index, columns=[column_records]))

            corrected_values = geoglows.bias.correct_forecast(fixed_records_df, simulated_df, observed_df)
            corrected_values = corrected_values.multiply(min_factor_records_df, axis=0)
            corrected_values = corrected_values.multiply(max_factor_records_df, axis=0)
            fixed_records = fixed_records.append(corrected_values)

        fixed_records.sort_index(inplace=True)

        '''Return Periods SONICS'''
        max_annual_flow = corrected_df.groupby(corrected_df.index.strftime("%Y")).max()
        params = distr.gev.lmom_fit(max_annual_flow.iloc[:, 0].values.tolist())

        return_periods = [10, 5, 2.33]

        return_periods_values = []

        for rp in return_periods:
            return_periods_values.append(gve_1(params['loc'], params['scale'], params['c'], rp))

        d = {'rivid': [comid], 'return_period_10': [return_periods_values[0]],
             'return_period_5': [return_periods_values[1]], 'return_period_2_33': [return_periods_values[2]]}

        rperiods_sonics = pd.DataFrame(data=d)
        rperiods_sonics.set_index('rivid', inplace=True)

        '''Return Periods GEOGloWS'''
        max_annual_flow = simulated_df.groupby(simulated_df.index.strftime("%Y")).max()
        params = distr.gev.lmom_fit(max_annual_flow.iloc[:, 0].values.tolist())

        for rp in return_periods:
            return_periods_values.append(gve_1(params['loc'], params['scale'], params['c'], rp))

        d = {'rivid': [comid], 'return_period_10': [return_periods_values[0]],
             'return_period_5': [return_periods_values[1]], 'return_period_2_33': [return_periods_values[2]]}

        rperiods_geoglows = pd.DataFrame(data=d)
        rperiods_geoglows.set_index('rivid', inplace=True)

        '''Return Periods GEOGloWS'''
        max_annual_flow = simulated_df.groupby(simulated_df.index.strftime("%Y")).max()
        params = distr.gev.lmom_fit(max_annual_flow.iloc[:, 0].values.tolist())

        for rp in return_periods:
            return_periods_values.append(gve_1(params['loc'], params['scale'], params['c'], rp))

        d = {'rivid': [comid], 'return_period_10': [return_periods_values[0]],
             'return_period_5': [return_periods_values[1]], 'return_period_2_33': [return_periods_values[2]]}

        rperiods_geoglows = pd.DataFrame(data=d)
        rperiods_geoglows.set_index('rivid', inplace=True)

        '''Return Periods Corrected GEOGloWS'''
        max_annual_flow = corrected_df.groupby(corrected_df.index.strftime("%Y")).max()
        params = distr.gev.lmom_fit(max_annual_flow.iloc[:, 0].values.tolist())

        for rp in return_periods:
            return_periods_values.append(gve_1(params['loc'], params['scale'], params['c'], rp))

        d = {'rivid': [comid], 'return_period_10': [return_periods_values[0]],
             'return_period_5': [return_periods_values[1]], 'return_period_2_33': [return_periods_values[2]]}

        rperiods_bc_geoglows = pd.DataFrame(data=d)
        rperiods_bc_geoglows.set_index('rivid', inplace=True)

        '''Plotting Forecast'''

        titles = {'Reach ID': comid}
        hydroviewer_figure = geoglows.plots.forecast_stats(stats=fixed_stats, titles=titles)

        x_vals = (fixed_stats.index[0], fixed_stats.index[len(fixed_stats.index) - 1], fixed_stats.index[len(fixed_stats.index) - 1], fixed_stats.index[0])
        max_visible = max(fixed_stats.max())

        record_plot = fixed_records.copy()
        record_plot = record_plot.loc[record_plot.index >= pd.to_datetime(fixed_stats.index[0] - dt.timedelta(days=8))]
        record_plot = record_plot.loc[record_plot.index <= pd.to_datetime(fixed_stats.index[0] + dt.timedelta(days=2))]

        if len(records.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='1st days forecasts',
                x=record_plot.index,
                y=record_plot.iloc[:, 0].values,
                line=dict(
                    color='#FFA15A',
                )
            ))

            x_vals = (records.index[0], fixed_stats.index[len(fixed_stats.index) - 1], fixed_stats.index[len(fixed_stats.index) - 1], records.index[0])
            max_visible = max(max(records.max()), max_visible)

        '''Adding Recent Days'''
        records_df = historical_simulation_df.loc[historical_simulation_df.index >= pd.to_datetime(historical_simulation_df.index[-1] - dt.timedelta(days=8))]
        records_df = records_df.loc[records_df.index <= pd.to_datetime(historical_simulation_df.index[-1] + dt.timedelta(days=2))]

        if len(records_df.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='SONICS',
                x=records_df.index,
                y=records_df.iloc[:, 0].values,
                line=dict(color='green', )
            ))

            x_vals = (records_df.index[0], fixed_stats.index[len(fixed_stats.index) - 1], fixed_stats.index[len(fixed_stats.index) - 1],records_df.index[0])
            max_visible = max(max(records_df.max()), max_visible)

        '''SONICS Forecaast'''
        if len(forecast_gfs_df.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='GFS Forecast',
                x=forecast_gfs_df.index,
                y=forecast_gfs_df['Streamflow (m3/s)'],
                showlegend=True,
                line=dict(color='black', dash='dash')
            ))

            max_visible = max(max(forecast_gfs_df.max()), max_visible)

        if len(forecast_eta_df.index) > 0:
            hydroviewer_figure.add_trace(go.Scatter(
                name='ETA Forecast',
                x=forecast_eta_df.index,
                y=forecast_eta_df['Streamflow (m3/s)'],
                showlegend=True,
                line=dict(color='blue', dash='dash')
            ))

            max_visible = max(max(forecast_eta_df.max()), max_visible)

        '''Getting Return Periods'''
        r2_33 = int(rperiods_bc_geoglows.iloc[0]['return_period_2_33'])

        colors = {
            '2.33 Year': 'rgba(243, 255, 0, .4)',
            '5 Year': 'rgba(255, 165, 0, .4)',
            '10 Year': 'rgba(255, 0, 0, .4)',
        }

        if max_visible > r2_33:
            visible = True
            hydroviewer_figure.for_each_trace(
                lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (), )
        else:
            visible = 'legendonly'
            hydroviewer_figure.for_each_trace(
                lambda trace: trace.update(visible=True) if trace.name == "Maximum & Minimum Flow" else (), )

        def template(name, y, color, fill='toself'):
            return go.Scatter(
                name=name,
                x=x_vals,
                y=y,
                legendgroup='returnperiods',
                fill=fill,
                visible=visible,
                line=dict(color=color, width=0))

        r5 = int(rperiods_bc_geoglows.iloc[0]['return_period_5'])
        r10 = int(rperiods_bc_geoglows.iloc[0]['return_period_10'])

        hydroviewer_figure.add_trace(template('Return Periods', (r10 * 0.05, r10 * 0.05, r10 * 0.05, r10 * 0.05), 'rgba(0,0,0,0)', fill='none'))
        hydroviewer_figure.add_trace(template(f'2.33 Year: {r2_33}', (r2_33, r2_33, r5, r5), colors['2.33 Year']))
        hydroviewer_figure.add_trace(template(f'5 Year: {r5}', (r5, r5, r10, r10), colors['5 Year']))
        hydroviewer_figure.add_trace(template(f'10 Year: {r10}', (r10, r10, max(r10 + r10 * 0.05, max_visible), max(r10 + r10 * 0.05, max_visible)), colors['10 Year']))

        hydroviewer_figure['layout']['xaxis'].update(autorange=True)

        chart_obj = PlotlyView(hydroviewer_figure)

        context = {
            'gizmo_object': chart_obj,
        }

        return render(request, 'sonics_geoglows/gizmo_ajax.html', context)


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
            'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })

def get_geoglows_forecast_bc_data_csv(request):
    """""
    Returns Forecast data as csv
    """""

    get_data = request.GET

    startdate = get_data['startdate']

    try:
        #get station attributes
        watershed = get_data['watershed']
        subbasin = get_data['subbasin']
        comid = get_data['comid']
        startdate = get_data['startdate']

        '''Get Forecast Ensemble Data'''
        forecast_ens_bc_file_path = os.path.join(app.get_app_workspace().path, 'forecast_ens_bc.json')
        corrected_ensembles = pd.read_json(forecast_ens_bc_file_path, convert_dates=True)
        corrected_ensembles.index = pd.to_datetime(corrected_ensembles.index)
        corrected_ensembles.sort_index(inplace=True, ascending=True)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=corrected_geoglows_forecast_{0}_{1}.csv'.format(comid, startdate)

        corrected_ensembles.to_csv(encoding='utf-8', header=True, path_or_buf=response)

        return response

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("error: " + str(e))
        print("line: " + str(exc_tb.tb_lineno))

        return JsonResponse({
                'error': f'{"error: " + str(e), "line: " + str(exc_tb.tb_lineno)}',
        })
