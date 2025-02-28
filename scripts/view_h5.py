import pandas as pd

store = pd.HDFStore('data/processed_data.h5')
df = store["d"]
print(df)
store.close()