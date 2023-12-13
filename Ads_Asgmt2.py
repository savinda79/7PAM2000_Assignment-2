# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:13:55 2023

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew



# Read the CSV file into a DataFrame
def read_file(filename):
    #Read and return the data frame
    climate_df = pd.read_csv( filename, skiprows=4)
    
    return climate_df

# This Function used to clean,preporcess
def process_and_transpose(df,xyz,year,country):
    
    df1=df[(df['Indicator Name']==xyz)]
    df1 = df1.reset_index()
    df1.set_index('Country Name',inplace=True)
    df1=df1.loc[:,year]
    df1=df1.loc[country,:]
    
    df2=df1.reset_index().rename(columns={"index":"Country"})
    
    #Trnapose the data frame
    df3=df1.T
    
    return df1,df2,df3

    

def draw_barplot(val,title):
    
    val.plot.bar(x='Country Name', figsize=(50,30),fontsize=60)
    plt.legend(fontsize=50)
    plt.title(title.upper(),fontsize=75)
    plt.xlabel("Countries",fontsize=65)
    plt.ylabel(title,fontsize=65)
    plt.savefig(title +'.png')
    plt.show()
    
    return

def draw_linplot(val1,title):
    
    val1.plot.line(figsize=(50,30),fontsize=60,linewidth=6.0,linestyle='--')
    plt.title(title.upper(),fontsize=60)
    plt.xlabel("Year",fontsize=60)
    plt.ylabel(title,fontsize=60)
    plt.legend(fontsize=50)
    plt.savefig(title +'.png')
    plt.show()
    
    return



def draw_heatmap(df,cnty):
    heat_df=df
    
    heat_df=heat_df[['Country Name','Indicator Name','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]

    heat_df=heat_df.fillna(0)
    heat_df1=heat_df[(heat_df['Country Name']==cnty)]
    
    

    multiple_values = ['Electricity production from coal sources (% of total)','Urban population growth (annual %)', 'Population growth (annual %)',
                   'Access to electricity (% of population)','CO2 emissions from solid fuel consumption (kt)',
                   'Electricity production from renewable sources, excluding hydroelectric (% of total)',
                  'Electric power consumption (kWh per capita)',
                  'Electricity production from oil sources (% of total)',
                  'Electricity production from nuclear sources (% of total)',
                  'Electricity production from hydroelectric sources (% of total)',
                  'Electricity production from coal sources (% of total)']

    filtered_data = heat_df1[heat_df1['Indicator Name'].isin(multiple_values)]
    
    filtered_data = filtered_data.reset_index()
    filtered_data = filtered_data.set_index('Indicator Name')
    filtered_data_new = filtered_data[['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    
    #Transpose data
    filtered_data_new_t=filtered_data_new.T
    filtered_data_new_t
    
    #Calling the heat Map
    correlation_matrix = filtered_data_new_t.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(12,10) )
    sns.heatmap(correlation_matrix, annot=True,
            cmap='flare', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap for '+cnty)
    #plt.savefig('heatmap.png')
    plt.savefig('heatmap of '+cnty +'.png')
    plt.show()

    return filtered_data_new_t


#calculate skwenes value 
def skew(dist):
    """
    Calculates the centralised and normalised skewness of dist.
    """

    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)

    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / (len(dist) - 1)

    return value


#Calling functions

df = read_file("wb_climatechange_data.csv")

#print(df.head(10))


year=['1990','1995','2000','2005','2010','2015','2020']
year2=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
country=['Pakistan','Colombia','Canada','China','Ecuador','Germany','India','Ethiopia','South Africa','United States']

indicator1='CO2 emissions from solid fuel consumption (kt)'
#indicator2='Electric power consumption (kWh per capita)'
indicator2='Electricity production from coal sources (% of total)'

indicator3='Urban population growth (annual %)'
indicator4='Electric power consumption (kWh per capita)'

#df1 = df[df['Indicator Code']==Indicator]

df1new,df2new,df3new = process_and_transpose(df,indicator1,year,country)
df1new,df2new2,df3new2 = process_and_transpose(df,indicator2,year,country)
df1new,df2new3,df3new3 = process_and_transpose(df,indicator3,year2,country)
df1new,df2new4,df3new4 = process_and_transpose(df,indicator4,year2,country)

draw_linplot(df3new3,indicator3)
draw_linplot(df3new4,indicator4)

draw_barplot(df2new,indicator1)
draw_barplot(df2new2,indicator2)


#darw heat maps 

#filtered_data=draw_heatmap(df,'China')
filtered_data_new_t=draw_heatmap(df,'South Africa')

# Call the function
skewness_value = skew(filtered_data_new_t)
rounded_skewness_value = np.round(skewness_value, 3)
print(f"Skewness: {rounded_skewness_value}")


