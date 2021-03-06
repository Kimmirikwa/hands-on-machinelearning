# this involves reading the data, preparing it, visualizing, training a model and using it to 
# make a prediction
# the model will be simple and will only use one training feature.

# the necessary imports
import matplotlib  # for plotting 2D figures
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def prepare_country_stats(oecd_bli, gdp_per_capita):
    '''
        merges the 2 dataframes
        return one dataset with "Country" as the index and only 2 columns i.e "GDP per capita" and "Life satisfaction"
    '''
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]  # removes duplicates to allow pivoting
    # pivoting the table will reshape it refer here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
    # the pivoted dataframe will have the following columns: "Air pollution", "Assualt rate", "Life satisfaction"...etc. These are
    # the values that originally belonged to "Indicator" column of the original dataset
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    # now both dataframes have "Country" as the index
    # merge the dataframes on the Index - "Country" i.e, rows belonging to the
    # same country from the 2 dataframes will form 1 row in the new merge dataframe
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    return full_country_stats[["GDP per capita", 'Life satisfaction']]

# load the data. The data will be held in pandas dataframes
# for more information on pandas dataframes: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
oecd_bli = pd.read_csv('datasets/oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('datasets/gdp_per_capita.csv', thousands=',', delimiter='\t',
	encoding='latin1', na_values='n/a')

# merge the 2 dataframes for each row to have data of a particular country
# the dataframe will be indexed by country and will have 2 columns
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

X = country_stats[["GDP per capita"]]  # training feature, using only one feature for investigation purposes
y = country_stats[["Life satisfaction"]]  # target variable

# scatter plot of X and y
# there is a rough trend of increase in life satisfaction with increase in per capita gdp
country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
plt.show()

# use linearregression model to train and predict
# the model is very basic
linear_reg = LinearRegression()

linear_reg.fit(X, y)

# predict satisfaction for cyprus with 22587 as per capita GDP
print(linear_reg.predict([[22587]]))
