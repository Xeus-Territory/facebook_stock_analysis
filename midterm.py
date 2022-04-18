from cProfile import label
from datetime import datetime
from enum import auto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

# Step 1: Load data set from directory and choice features to use
dataSet = pd.read_csv('D:\Code\Project\PythonTutorial\midterm\FB_stock_history.csv')

data_features = dataSet['Open'].where(dataSet['Date'] > '2018-01-01').dropna()

# Step 2: Create a func to calculate and plot by bootstrap method
def bootstrap_func(data_features, n_bootstraps_sample_size, M_number_of_bootstraps_sampling, x_confidence_interval):
    boot_IQR = []
    
    for i in range(M_number_of_bootstraps_sampling):
        boot_sample = np.random.choice(data_features, size = n_bootstraps_sample_size)
        q3, q1 = np.percentile(boot_sample, [75, 25])
        iqr = q3 - q1
        boot_IQR.append(iqr)
    
    boot_IQR = np.array(boot_IQR)
    
    range_reduce_of_each_side = (100 - x_confidence_interval) / 2
    
    range_confidence_interval = np.percentile(boot_IQR, [0 + range_reduce_of_each_side, 100 - range_reduce_of_each_side])
    
    plt.figure(figsize=(10,8))
    plt.title("Bootstrap Method to Sampling and Get Confidence Interval of IQR" , fontsize = 12.0)
    plt.ylim(0 , 160)
    sns.histplot(boot_IQR)
    plt.axvline(x =range_confidence_interval[0], ymin = 0, ymax = .95, color = 'r', linewidth = 2)
    plt.text(x = range_confidence_interval[0] - 1, y = 153, s = '{:.2f}'.format(range_confidence_interval[0]), color = 'r', fontsize = 12.0)
    plt.axvline(x = range_confidence_interval[1], ymin= 0, ymax = .95, color = 'r', linewidth = 2, label = range_confidence_interval[1])
    plt.text(x = range_confidence_interval[1] - 1, y = 153, s = '{:.2f}'.format(range_confidence_interval[1]), color = 'r', fontsize = 12.0)
    plt.hlines(y = 140, xmin= range_confidence_interval[0], xmax = range_confidence_interval[1], color = 'r', linewidth = 2, linestyles='--')
    plt.text(x = (range_confidence_interval[0] + range_confidence_interval[1]) / 3 + 13, y = 150, s = f'Confidence Interval: {x_confidence_interval}%', color = 'r', fontsize = 12.0)
    
# Step 3: Draw plot hist and create IQR table
histogram_Image = bootstrap_func(data_features, 50, 1000, 95)

    
    
# Problem 2:
# Processing data to create the new col : Growth

# Get the data from the 4 month last year and calculate the growth rate
data_features = dataSet[['Date', 'Volume', 'Open', 'Close', 'High', 'Low']].where(dataSet['Date'] > '2021-06-01').dropna()

# Set the value for colum 'Growth' based on the formula of Growth
conditions = [(data_features['Open'] - data_features['Close']) < 0, (data_features['Open'] - data_features['Close']) > 0]
values = ['Growth Up', 'Growth Down']
data_features['Growth'] = np.select(conditions, values, default = 'No Growth')

# Calculate the SMA (Simple Moving Average) of the stock in 4 month last year
data_features['SMA5'] = data_features['Close'].transform(lambda x: x.rolling(window=5).mean())
data_features['SMA20'] = data_features['Close'].transform(lambda x: x.rolling(window=20).mean())


# Plot Bar chart and line chart to show the growth rate of stock in the last year base on Volume, Price and SMA values
plt.figure(figsize=(20,10))
ax0 = plt.subplot2grid((6,4), (0,0), rowspan=4, colspan=4)
ax0.plot(data_features['Date'], data_features['SMA5'], data_features['Date'], data_features['SMA20'], data_features['Date'], data_features['Close'])
ax0.set_ylabel('Price')
ax0.legend(['SMA5','SMA20','Close'],ncol=3, loc = 'upper left', fontsize = 15)
plt.xticks(rotation=90)
plt.xticks(np.arange(0, len(data_features['Date']), step = 5))
plt.title('Facebook Stock Price, Slow and Fast Moving Average', fontsize = 12.0)
ax1 = plt.subplot2grid((6,4), (4,0), rowspan=2, colspan=4, sharex = ax0)
sns.barplot(x = data_features['Date'], y = data_features['Volume'], hue=data_features['Growth'])
plt.xticks(rotation=90)
plt.xticks(np.arange(0, len(data_features['Date']), step = 5))

plt.figure(figsize=(20,8))
# Create the col on daily change percentage for the stock 
data_features['daily_change_percentage'] = data_features['Close'].pct_change() * 100
data_features['returns'] = data_features['daily_change_percentage'] / data_features['Close']
data_features['daily_change_percentage'] = data_features['daily_change_percentage'].fillna(0)
ax2 = plt.subplot(1, 3, 1)
sns.histplot(data_features['daily_change_percentage'])
sns.kdeplot(data_features['daily_change_percentage'])
ax2.set_title("Histogram of Daily Change Percentage", fontsize = 12.0)
ax2.set_xlabel("Daily Change Percentage", fontsize = 12.0)
ax2.set_ylabel("Frequency", fontsize = 12.0)

# Create the pie chart and bar chart about trend of the stock in 4 month last year
def daily_trend(x):
    if x > -0.5 and x < 0.5:
        return 'No change'
    if x > 0.5 and x < 2.0:
        return 'Up to 2% Increase'
    if x > -2.0 and x < -0.5:
        return 'Up to 2% Decrease'
    if x > 2.0 and x < 5.0:
        return '2-5% Increase'
    if x > -5.0 and x < -2.0:
        return '2-5% Decrease'
    if x > -10.0 and x < -5.0:
        return '5-10% Decrease'
    if x > 5.0 and x < 10.0:
        return '5-10% Increase'
    if x > 10.0:
        return '>10% Increase'
    if x < -10.0:
        return '>10% Decrease'
    
data_features['daily_trend'] = data_features['daily_change_percentage'].apply(lambda x: daily_trend(x))
data_features['daily_trend'] = data_features['daily_trend'].replace('None', 'No change')

data_pie_data = data_features.groupby('daily_trend')
ax3 = plt.subplot(1, 3, 2)
colors = sns.color_palette('pastel')[0:9]
plt.pie(data_pie_data['daily_trend'].count(), autopct= '%.2f%%', radius= 1.6, colors= colors)
ax3.legend(title = 'Trend of the stock', loc = 'upper right', bbox_to_anchor=(0.0, 1), prop = {'size': 8}
           , labels = ['%s' % x for x in data_pie_data['daily_trend'].unique()])

ax4 = plt.subplot(1, 3, 3)
data_pie_data['daily_trend'].count().sort_values(ascending = False).plot.bar(rot = 60, fontsize = 8.0)
ax4.set_title("Detail of Trend of the stock", fontsize = 12.0)

plt.show()