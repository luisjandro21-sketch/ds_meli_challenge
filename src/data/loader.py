import pandas as pd


class DataLoader:
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_json(path, lines=True)
