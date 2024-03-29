import pandas as pd
import matplotlib.pyplot as plt

with open('./suv_data.csv') as f:
    data = pd.read_csv(f)

print(data)