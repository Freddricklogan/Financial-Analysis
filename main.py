# Financial Data Analysis Tool
import pandas as pd
import matplotlib.pyplot as plt

def analyze_financial_data(file_path):
    '''Analyzes financial data to identify trends and patterns'''
    data = pd.read_csv(file_path)
    return data.describe()
