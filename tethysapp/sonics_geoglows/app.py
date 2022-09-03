from tethys_sdk.base import TethysAppBase, url_map_maker
from tethys_sdk.app_settings import CustomSetting, SpatialDatasetServiceSetting

class SonicsGeoglows(TethysAppBase):
    """
    Tethys app class for SONICS Hydroviewer.
    """

    name = 'SONICS GEOGloWS'
    index = 'sonics_geoglows:home'
    icon = 'sonics_geoglows/images/sonics_geoglows_logo.png'
    package = 'sonics_geoglows'
    root_url = 'sonics-geoglows'
    color = '#292b47'
    description = ''
    tags = '"Hydrology", "SONICS", "GEOGloWS", "Peru"'
    enable_feedback = False
    feedback_emails = []

    def spatial_dataset_service_settings(self):
        """
		Spatial_dataset_service_settings method.
		"""
        return (
            SpatialDatasetServiceSetting(
                name='main_geoserver',
                description='spatial dataset service for app to use (https://tethys2.byu.edu/geoserver/rest/)',
                engine=SpatialDatasetServiceSetting.GEOSERVER,
                required=True,
            ),
        )

    def url_maps(self):
        """
        Add controllers
        """
        UrlMap = url_map_maker(self.root_url)

        url_maps = (
            UrlMap(
                name='home',
                url='sonics-geoglows',
                controller='sonics_geoglows.controllers.home'
            ),
            UrlMap(
                name='get_popup_response',
                url='get-request-data',
                controller='sonics_geoglows.controllers.get_popup_response'
            ),
            UrlMap(
                name='get_hydrographs',
                url='get-hydrographs',
                controller='sonics_geoglows.controllers.get_hydrographs'
            ),
            UrlMap(
                name='get_observed_discharge_csv',
                url='get-observed-discharge-csv',
                controller='sonics_geoglows.controllers.get_observed_discharge_csv'
            ),
            UrlMap(
                name='get_simulated_discharge_csv',
                url='get-simulated-discharge-csv',
                controller='sonics_geoglows.controllers.get_simulated_discharge_csv'
            ),
            UrlMap(
                name='get_simulated_bc_discharge_csv',
                url='get-simulated-bc-discharge-csv',
                controller='sonics_geoglows.controllers.get_simulated_bc_discharge_csv'
            ),
            UrlMap(
                name='get-time-series',
                url='get-time-series',
                controller='sonics_geoglows.controllers.get_time_series'
            ),
            UrlMap(
                name='get-time-series-bc',
                url='get-time-series-bc',
                controller='sonics_geoglows.controllers.get_time_series_bc'
            ),
            UrlMap(
                name='get_sonics_forecast_data_csv',
                url='get-sonics-forecast-data-csv',
                controller='sonics_geoglows.controllers.get_sonics_forecast_data_csv'
            ),
            UrlMap(
                name='get_geoglows_forecast_data_csv',
                url='get-geoglows-forecast-data-csv',
                controller='sonics_geoglows.controllers.get_geoglows_forecast_data_csv'
            ),
            UrlMap(
                name='get_geoglows_forecast_bc_data_csv',
                url='get-bc-geoglows-forecast-data-csv',
                controller='sonics_geoglows.controllers.get_geoglows_forecast_bc_data_csv'
            ),
        )

        return url_maps

    def custom_settings(self):
        return (
            CustomSetting(
                name='folder',
                type=CustomSetting.TYPE_STRING,
                description="Floder where the SONICS Forecast are stored",
                required=True,
                default='/home/tethys/sonics',
            ),
        )