import pandas as pd
import numpy as np

def load_and_clean_data():
    df = pd.read_csv("googleplaystore.csv")


    df = df[df['Rating'].notnull()]
    df = df[df['Rating'] <= 5]  

    df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)

    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    df = df[df['Installs'].notnull()]  
    df['Installs'] = df['Installs'].astype(int)

    df['Price'] = df['Price'].str.replace('$', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)

  
    def size_to_mb(size):
        if size == 'Varies with device' or size is np.nan:
            return np.nan
        if size.endswith('M'):
            return float(size[:-1])
        elif size.endswith('k'):
            return float(size[:-1]) / 1024
        else:
            return np.nan
    df['Size_MB'] = df['Size'].apply(size_to_mb)


    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_and_clean_data()
    print("Initial Data Preview:")
    print(df.head())
    print("\nCleaned Data Info:")
    print(df.info())
    print("\nCleaned Data Sample:")
    print(df[['App', 'Category', 'Rating', 'Installs', 'Price', 'Size_MB']].head())
