import pandas as pd
import os


class Data():
    """ Класс, выкачивающий данные

    df_money_start - датафрейм с изначаьным объемом денег в банкоматах
    df_money_in - датафрейм с пополнениями банкоматов
    df_geo_TIDS - датафрейм с  гео координатами банкоматов
    df_distance_matrix - датафрейм с временными расстояниями между банкоматами
    """
    def __init__(self, data_folder: str='../data'):
        self.__data_folder = data_folder
        self.df_money = None
        self.df_money_start = None
        self.df_money_in = None
        self.df_geo_TIDS = None
        self.df_distance_matrix = None
        self.params_dict = None
        self.clustered_tids = None
    
    def get_distance_matrix(self):
        if self.df_distance_matrix is None:
            folder = os.path.join(self.__data_folder, 'times v4.csv')
            self.df_distance_matrix = pd.read_csv(folder)
        return self.df_distance_matrix

    def get_money_all(self):
        if self.df_money is None:
            money_folder = os.path.join(self.__data_folder, 'terminal_data_hackathon v4.xlsx')
            self.df_money = pd.read_excel(money_folder, sheet_name='Incomes')
        return self.df_money    
    
    def get_money_start(self):
        if self.df_money_start is None:
            if self.df_money is None:
                self.get_money_all()
            df_money_start = self.df_money[['TID','остаток на 31.08.2022 (входящий)']]
            self.df_money_start = df_money_start.rename(columns = {'остаток на 31.08.2022 (входящий)':'money'})
        return self.df_money_start

    def get_money_in(self):
        if self.df_money_in is None:
            if self.df_money is None:
                self.get_money_all()
            df_money = self.df_money.copy()    
            df_money.index = df_money['TID']
            df_money.drop(['TID','остаток на 31.08.2022 (входящий)'], axis=1, inplace=True)
            df_money_in = df_money.unstack(level=0)
            df_money_in = df_money_in.reset_index().rename(columns={'level_0':'date', 0:'money_in'})
            #df_money_in['money_in'] = df_money_in['money_in'].where(df_money_in['money_in'] <= params_dict['max_money'], other=params_dict['max_money'] )
            df_money_in['date'] = pd.to_datetime(df_money_in['date'])
            self.df_money_in = df_money_in
        return self.df_money_in


    def get_geo_TIDS(self):
        if self.df_geo_TIDS is None:
            geo_TIDS_folder = os.path.join(self.__data_folder, 'terminal_data_hackathon v4.xlsx')
            self.df_geo_TIDS = pd.read_excel(geo_TIDS_folder, sheet_name='TIDS')
        return self.df_geo_TIDS
    
    def get_params_dict(self):
        if self.params_dict is None:
            params_folder = os.path.join(self.__data_folder, 'Params.xlsx')
            df_params= pd.read_excel(params_folder)
            self.params_dict = df_params.set_index('param')['value'].to_dict()
        return self.params_dict
    
