import pandas as pd

class DataLoader:

    def __init__(self, ):
        pass


    def get_data_as_pd(self, data_file_path):
        return pd.read_parquet(data_file_path)