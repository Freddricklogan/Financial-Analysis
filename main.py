# Financial-Analysis/financial_analysis.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Analysis Tool

This script performs comprehensive financial analysis on business records from a CSV file.
It calculates key metrics including:
- Total revenue and expenses
- Monthly profit/loss analysis
- Percentage changes and growth rates
- Largest increases and decreases
- Statistical analysis of financial patterns

Author: Freddrick Logan
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from pathlib import Path

def load_financial_data(filepath):
    """
    Load financial data from CSV file.
    
    Parameters:
        filepath (str): Path to the financial data CSV file
    
    Returns:
        pandas.DataFrame: DataFrame containing the financial records
    """
    print(f"Loading financial data from: {filepath}")
    
    try:
        # Read data with pandas for easier manipulation
        df = pd.read_csv(filepath, parse_dates=['Date'], infer_datetime_format=True)
        print(f"Successfully loaded {len(df)} financial records.")
        
        # Basic data validation
        required_columns = ['Date', 'Profit/Losses']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the CSV file")
        
        return df
    
    except Exception as e:
        print(f"Error loading financial data: {e}")
        return None

def perform_financial_analysis(data):
    """
    Perform comprehensive financial analysis on the provided data.
    
    Parameters:
        data (pandas.DataFrame): DataFrame containing financial records
    
    Returns:
        dict: Dictionary containing analysis results
    """
    # Check if data is valid
    if data is None or len(data) == 0:
        print("No data to analyze.")
        return None
    
    # Initialize results dictionary
    results = {}
    
    # Calculate basic financial metrics
    results['total_months'] = len(data)
    results['total'] = data['Profit/Losses'].sum()
    
    # Calculate changes between months
    data['Previous'] = data['Profit/Losses'].shift(1)
    data['Change'] = data['Profit/Losses'] - data['Previous']
    
    # Skip first row when calculating changes (no previous month to compare to)
    valid_changes = data['Change'].dropna()
    
    # Calculate average change
    results['average_change'] = valid_changes.mean()
    
    # Find greatest increase and decrease
    max_increase_row = data.loc[data['Change'].idxmax()]
    max_decrease_row = data.loc[data['Change'].idxmin()]
    
    results['greatest_increase'] = {
        'date': max_increase_row['Date'].strftime('%b-%Y'),
        'amount': max_increase_row['Change']
    }
    
    results['greatest_decrease'] = {
        'date': max_decrease_row['Date'].strftime('%b-%Y'),
        'amount': max_decrease_row['Change']
    }
    
    # Additional analysis: Monthly statistics
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    monthly_profits = data.groupby(['Year', 'Month'])['Profit/Losses'].sum()
    results['monthly_profits'] = monthly_profits
    
    # Calculate year-over-year growth rates
    yearly_profits = data.groupby('Year')['Profit/Losses'].sum()
    yearly_profits_prev = yearly_profits.shift(1)
    yearly_growth = (yearly_profits - yearly_profits_prev) / yearly_profits_prev * 100
    results['yearly_growth'] = yearly_growth.dropna()
    
    # Calculate volatility (standard deviation of changes)
    results['volatility'] = valid_changes.std()
    
    # Determine profit vs loss months
    results['profit_months'] = (data['Profit/Losses'] > 0).sum()
    results['loss_months'] = (data['Profit/Losses'] <= 0).sum()
    
    # Calculate moving averages for trend analysis
    data['3M_Rolling_Avg'] = data['Profit/Losses'].rolling(window=3).mean()
    data['6M_Rolling_Avg'] = data['Profit/Losses'].rolling(window=6).mean()
    results['moving_averages'] = data[['Date', '3M_Rolling_Avg', '6M_Rolling_Avg']]
    
    return results, data

def generate_visualizations(data, results, output_dir):
    """
    Generate visualizations of financial analysis.
    
    Parameters:
        data (pandas.DataFrame): Original data with added analysis columns
        results (dict): Dictionary containing analysis results
        output_dir (str): Directory to save visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for plots
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # 1. Monthly Profit/Loss Plot
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Profit/Losses'], color='blue', marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.title('Monthly Profits and Losses Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Profit/Loss ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_profit_loss.png'))
    plt.close()
    
    # 2. Monthly Changes Plot
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'][1:], data['Change'][1:], color='green', marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Monthly Changes in Profits and Losses', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Change in Profit/Loss ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_changes.png'))
    plt.close()
    
    # 3. Rolling Averages Plot
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Profit/Losses'], color='blue', alpha=0.5, label='Monthly Profit/Loss')
    plt.plot(data['Date'], data['3M_Rolling_Avg'], color='red', linewidth=2, label='3-Month Rolling Average')
    plt.plot(data['Date'], data['6M_Rolling_Avg'], color='green', linewidth=2, label='6-Month Rolling Average')
    plt.title('Profit/Loss Trends with Moving Averages', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Profit/Loss ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rolling_averages.png'))
    plt.close()
    
    # 4. Yearly Profits Bar Chart
    yearly_profits = data.groupby('Year')['Profit/Losses'].sum()
    plt.figure(figsize=(12, 6))
    bars = plt.bar(yearly_profits.index, yearly_profits.values, color='skyblue')
    plt.title('Total Profit/Loss by Year', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Profit/Loss ($)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add data labels to bars
    for bar in bars:
        height = bar.get_height()
        label_text = f"${height:,.0f}"
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + (0.1 * height if height > 0 else -0.1 * abs(height)), 
            label_text,
            ha='center', va='bottom' if height > 0 else 'top',
            rotation=0
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yearly_profits.png'))
    plt.close()
    
    # 5. Monthly Profit/Loss Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Profit/Losses'], kde=True, bins=20)
    plt.title('Distribution of Monthly Profits/Losses', fontsize=16)
    plt.xlabel('Profit/Loss ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_loss_distribution.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_financial_report(results, output_path):
    """
    Generate a textual financial analysis report.
    
    Parameters:
        results (dict): Dictionary containing analysis results
        output_path (str): Path to save the report
    """
    report_lines = [
        "FINANCIAL ANALYSIS REPORT",
        "------------------------\n",
        f"Total Months: {results['total_months']}",
        f"Total: ${results['total']:,.2f}",
        f"Average Change: ${results['average_change']:,.2f}",
        f"Greatest Increase in Profits: {results['greatest_increase']['date']} (${results['greatest_increase']['amount']:,.2f})",
        f"Greatest Decrease in Profits: {results['greatest_decrease']['date']} (${results['greatest_decrease']['amount']:,.2f})",
        "\nADDITIONAL METRICS",
        "-----------------",
        f"Profit Months: {results['profit_months']}",
        f"Loss Months: {results['loss_months']}",
        f"Volatility (Std Dev of Changes): ${results['volatility']:,.2f}",
        "\nYEARLY GROWTH RATES",
        "------------------"
    ]
    
    # Add yearly growth rates to report
    for year, growth in results['yearly_growth'].items():
        report_lines.append(f"{year}: {growth:.2f}%")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Financial report saved to {output_path}")

def main():
    """
    Main function to execute the financial analysis.
    """
    print("Financial Analysis Tool Starting...")
    
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Resources', 'budget_data.csv')
    output_dir = os.path.join(current_dir, 'Analysis')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    financial_data = load_financial_data(data_path)
    
    if financial_data is None:
        print("Failed to load financial data. Exiting program.")
        return
    
    # Perform analysis
    results, enhanced_data = perform_financial_analysis(financial_data)
    
    if results is None:
        print("Failed to analyze financial data. Exiting program.")
        return
    
    # Generate report
    report_path = os.path.join(output_dir, 'financial_analysis_report.txt')
    generate_financial_report(results, report_path)
    
    # Generate visualizations
    vis_output_dir = os.path.join(output_dir, 'visualizations')
    generate_visualizations(enhanced_data, results, vis_output_dir)
    
    print("Financial analysis completed successfully!")

if __name__ == "__main__":
    main()