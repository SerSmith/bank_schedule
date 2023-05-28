from bank_schedule import cluster
from bank_schedule.data import Data


class DataClaster(Data):

    def __init__(self, data_folder: str='../data'):
        super().__init__(data_folder)

    def get_distance_matrix(self, cluster_num=None):

        super().get_distance_matrix()

        if cluster_num is None:
            out = self.df_distance_matrix
        else:
            tids_chosen = self.get_tids_by_claster(cluster_num)
            out = self.df_distance_matrix[(self.df_distance_matrix["Origin_tid"].isin(tids_chosen)) & (self.df_distance_matrix["Destination_tid"].isin(tids_chosen))]
        return out

    def get_money_in(self, cluster_num=None):

        super().get_money_in()

        if cluster_num is None:
            out = self.df_money_in
        else:
            tids_chosen = self.get_tids_by_claster(cluster_num)
            out = self.df_money_in[self.df_money_in['TID'].isin(tids_chosen)]

        return out

    def run_cluster(self, allowed_percent, n_clusters):
        self.clustered_tids = cluster.clusterize_atm(self, allowed_percent=allowed_percent, n_clusters=n_clusters)
        return self
    
    def get_tids_by_claster(self, cluster_num):

        assert self.clustered_tids is not None, "run run_cluster first"
        return self.clustered_tids[self.clustered_tids["label"] == cluster_num]['TID'].unique()
    
    def get_money_start(self, cluster_num=None):
        super().get_money_start()

        if cluster_num is None:
            out = self.df_money_start
        else:
            tids_chosen = self.get_tids_by_claster(cluster_num)
            out = self.df_money_start[self.df_money_start['TID'].isin(tids_chosen)]

        return out