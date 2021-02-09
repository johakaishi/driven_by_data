<img src="/readme_assets/1_driven_by_data_banner.jpg">

<a id='back_to_top'></a>

# Driven by Data
#### Joh Akaishi

This repository contains notebooks for my machine learning project, “Driven by Data”.<br>
Notebooks can be found below:

- [Web Scraping](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_1_scraping_cleaning.ipynb)
- [EDA](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_2_eda.ipynb)<br>
- [Modelling: linear regression](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_3_linreg.ipynb) (Including regularised models.)
- [Modelling: tree based regressors](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_4_dtr_rfr_etr.ipynb) (Decision Tree, Random Forest, Extra Trees Regressor.)
- [Modelling: boosting regressors](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_5_adaboost_gradientboost_histgbr_lightgbm.ipynb) (AdaBoost, GradientBoost, HistGradientBoost, LightGBM.)
- [Functions script](https://github.com/johakaishi/driven_by_data/blob/master/myfunctions.py) (Functions have been defined within myfunctions.py, which are called upon in the notebooks above.)



## Table of Contents
---
[1. Problem Statement](#problem_statement)<br>
[2. Project Flow](#projectflow)<br>
[3. Web Scraping, Feature Extraction and Cleaning](#webscraping)<br>
[4. Exploratory Data Analysis](#eda)<br>
[5. Modelling](#modelling)<br>
[6. Conclusions and Future Expansion](#conclusions)

<a id='problem_statement'></a>

## 1. Problem Statement
---
“What is my car worth?”<br><br>

There are many factors (or “features”…) that affect the value of a car. This project aims to dive deep (or should I say "drive" deep?) into this very question, and to use a data-driven approach to accurately predict the listed price of cars.<br><br>

Autotrader is one of the largest automotive classified advertising sites in the UK, listing both new and second hand cars sold by private sellers and trade dealers. I extracted the data for this project by scraping through the Autotrader website, resulting in approximately 15,000 unique (non-duplicated) observations, on which to perform exploratory analysis and predict prices using regression models.

[(back to top)](#back_to_top)

<a id='projectflow'></a>

## 2. Project Flow
---

<img src="/readme_assets/2_project_flowchart.jpg">

[(back to top)](#back_to_top)

<a id='webscraping'></a>

## 3. Web Scraping, Feature Extraction and Cleaning
---
[Link to notebook](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_1_scraping_cleaning.ipynb)

### Web Scraping
The data for this project was scraped from the Autotrader website using Beautiful Soup, Soup Strainer and Cloudscraper.

<img src="/readme_assets/3_autotrader_scraper_screenshot_2.png">

**Scraped fields:**

| Scraped Fields | Description |
| :- | :- |
| scr_price | Listed car price |
| scr_header | Header of the advert |
| scr_attngrab | "Attention grabbing" one liner! |
| scr_descr | Description of the vehicle <br>(incl. year, bodytype, mileage, engine size, horsepower, transmission, fuel type, ULEZ compliance, number of previous owners) |
| scr_sellerrating | Rating of the seller (5-star scale) |
| scr_sellertown | Location of the seller |
| scr_sellertype | Type of the seller (either private or trade seller) |
| scr_url | URL of the unique advert |


<br><br>
The scraped dataframe resulted in 62,082 observations, across 8 columns:

<img src="/readme_assets/4_scraped_df_screenshot.png">

### Dropping duplicates and null values
Below is a visual representation of null values, using the [missingno](https://github.com/ResidentMario/missingno) package. 
<br>White space represents null values in the dataset.

<img src="/readme_assets/5_scraped_df_msno_plot_62082.png">

- As the data was gathered through web scraping, there were duplicate rows within the dataset. 
- Firstly, these duplicates were dropped, and then rows containing null values were also dropped.

```python
# original df_master
print('before dropping duplicates:', df_master.shape)

# drop duplicates
df = df_master.drop_duplicates(ignore_index=True)
print('after dropping duplicates:', df.shape)
#df.head()

# drop null values
df = df.dropna().reset_index()
df.drop(columns='index', inplace=True)

print('after dropping null values:', df.shape)
display(df.head())
```

<img src="/readme_assets/6_df_after_dropping_nulls.png">

### Feature Extraction and Cleaning
The following features were extracted from the scr_descr column, using regex:

| Feature | Description | Extracted from |
| :- | :- | :- |
| make | Make (manufacturer) of the vehicle | 'scr_header' |
| model | Model of the vehicle | 'scr_header' |
| year | Year of manufacture | 'scr_descr' |
| reg_num | Registration number (related to year) | 'scr_descr' |
| body | Body type of vehicle ('hatchback', 'estate', 'SUV', etc) | 'scr_descr' |
| mileage | Mileage on the vehicle odometer | 'scr_descr' |
| engine_size | Size of the engine (litres) | 'scr_descr' |
| horsepower | Breaking horsepower | 'scr_descr' |
| transmission | Transmission type (AT or MT) | 'scr_descr' |
| fuel | Fuel type of the vehicle (petrol or diesel) | 'scr_descr' |
| owners | Number of previous owners of the vehicle | 'scr_descr' |
| ulez | Whether the vehicle is ULEZ compliant or not ([Ultra Low Emission Zone](https://tfl.gov.uk/modes/driving/ultra-low-emission-zone); a charge for polluting vehicles in central London)     | 'scr_descr' |
| seller_type | Type of seller ('trade seller' or 'approved dealer') | 'scr_sellertype' |
| price | Listed price of the vehicle advert in integer type | 'scr_price' |

Example code to extract features:

```python
# extract info from description column, to generate new columns
def year(x):
    pattern = re.compile(r'(\d{4})')
    element = re.findall(pattern, x)
    try:
        return int(element[0])
    except:
        return np.nan

def reg_num(x):
    pattern = re.compile(r'(\d\d reg)')
    element = re.findall(pattern, x)
    try:
        return int(element[0].split()[0])
    except:
        return np.nan

df['year'] = df.scr_descr.apply(lambda x: year(x))
df['reg_num'] = df.scr_descr.apply(lambda x: reg_num(x))
```

There were some entries where the engine size was picked up for the 'body' column:

<img src="/readme_assets/7_engine_size_in_body_col.png">

Corrected as below:

```python
# correct body style from engine size (eg 1.5l, 2.2l) to actual body style
def body_style_corrected(x):
    list_x = ast.literal_eval(x)
    return list_x[0].lower()

df.loc[df.body.str.contains('(\d.\dl)'), 'body'] = df.scr_descr.apply(lambda x: body_style_corrected(x))

# df = df where 'body' does not contain digits (ie delete all rows with digits in body column)
df = df[~df.body.str.contains('(\d)')]

# reset index inplace
df.reset_index(drop=True, inplace=True)
```

During extraction of 'make', it was found that those observations with multi-word manufacturer names only registered the first word.

- 'land' instead of 'land rover'
- 'aston' instead of 'aston martin'
- 'alfa' instead of 'alfa romeo'
- etc.

This was corrected using index replacement, as below:

```python
# correct make names for those with more than one word as the make name
double_car_names = ['land', 'aston', 'alfa', 'great']

indexer = df[df.make == 'land'].index
df.iloc[indexer, -2] = 'landrover'

indexer = df[df.make == 'aston'].index
df.iloc[indexer, -2] = 'astonmartin'

indexer = df[df.make == 'alfa'].index
df.iloc[indexer, -2] = 'alfaromeo'

indexer = df[df.make == 'great'].index
df.iloc[indexer, -2] = 'greatwall'
```

The resulting dataframe null values after the cleaning process:

<img src="/readme_assets/8_msno_after_cleaning_16199.png">

- Many of the 'owners' column had missing values, so this column was dropped (although there is a risk that the number of previous owners does have some correlation with the target: price).
- After dropping 'owners', all remaining rows with null values were dropped.
<br><br>
- The final dataframe with 14,996 observations (all unique):

<img src="/readme_assets/9_msno_final_df_14996.png">

[(back to top)](#back_to_top)

<a id='eda'></a>

## 4. Exploratory Data Analysis
---
[Link to notebook](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_2_eda.ipynb)

#### Distribution of Price

<img src="/readme_assets/10_distribution_price.png">

- As expected (for car prices), there is a heavy right skew showing the presence of a few outliers that are very expensive, some even reaching GBP 600,000!

<img src="/readme_assets/11_distribution%20log_price.png">

- Taking the logarithm of the price, results in an approximately normal distribution.

#### Correlation Heatmap and Scatterplots

<img src="/readme_assets/12_%20correlation_heatmap.png">

- Perhaps expectedly, with respect to price:
    - strong positive correlation with engine_size and horsepower.
    - negative correlation with mileage (cars with higher mileage on the odometer will sell for less).

<img src="/readme_assets/13_scatter_price_enginesize.png">

- Larger variance in price, when engine size increases, but smaller engine sizes will almost certainly result in a lower price.
- "Vertical lines" can be seen in the plot, also as engine sizes are stated down to one decimal place (eg 2.8L engine).

<img src="/readme_assets/14_scatter_price_horsepower.png">

- As horsepower (BHP) increases, the price also increases. 
- It seems there is an almost exponential relationship, which could be addressed using polynomial features, or a power of the horsepower to achieve a more linear relationship with price.

<img src="/readme_assets/15_scatter_price_horsepower_log.png">

<img src="/readme_assets/16_scatter_price_mileage.png">

- As mileage increases, there is a depreciation in price (resembling almost an exponential decay).

<img src="/readme_assets/17_regplot_horesepower_enginesize.png">

- Also understandably, there is a correlation between BHP and engine_size.

<img src="/readme_assets/18_scatter_year_regnum.png">

- Registration numbers correspond with the year, as can be seen in the plot above.
- For this reason, reg_num was dropped from the dataframe, to avoid issues with strong multicollinearity.

[(back to top)](#back_to_top)

<a id='modelling'></a>

## 5. Modelling
---
[Link to notebook: Linear regression with Ridge and Lasso regularisation](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_3_linreg.ipynb)<br>

[Link to notebook: Tree based regressors (Decision Tree, Random Forest, Extra Trees)](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_4_dtr_rfr_etr.ipynb)<br>

[Link to notebook: Boosting regressors (AdaBoost, GradientBoost, HistGradientBoost, LightGBM)](https://github.com/johakaishi/driven_by_data/blob/master/Autotrader_5_adaboost_gradientboost_histgbr_lightgbm.ipynb)<br>

[Link to 'myfunctions.py'](https://github.com/johakaishi/driven_by_data/blob/master/myfunctions.py) Functions have been defined within myfunctions.py, which are called upon in the notebooks above.

### Preprocessing

#### Train Test Split
- Dataframe split to have 20% set aside as the test set.

#### Pipeline
- A "pipeline construction" function was defined with the following transformers:
    - Standardisation for continuous variables
    - One Hot Encoding for categorical variables
    
```python
def pipe_construct(features_cont=features_cont, features_cat=features_cat, model=LinearRegression()):
    '''
    Input: continuous features, categorical features, model of choice. 
    Returns: pipeline.
    '''
    t = [('cont', StandardScaler(), features_cont), 
         ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), features_cat)]

    transformer = ColumnTransformer(transformers=t, remainder='drop')

    pipe = Pipeline(steps=[('transformer', transformer), ('model', model)])
    return pipe
```
<br>

- The pipe was then called upon in the following way (input features were selected in different combinations).

```python
# Define input features 
features_cont = ['year', 'mileage', 'engine_size', 'horsepower', 'bhp_per_litre']
features_cat = [i for i in df.columns if i not in features_cont]
features_cat.remove('price') # remove target variable

# fit pipe
pipe = pipe_construct(features_cont, features_cat, LinearRegression())
pipe.fit(X_train, y_train)
```

### Ridge Regression

The (5-fold) cross validated mean <img src="https://render.githubusercontent.com/render/math?math=R^{2}"> score (hereafter "CV mean") was used to assess model performance and generalisability to new, unknown data.

Ridge Regression (<img src="https://render.githubusercontent.com/render/math?math=\alpha"> = 0.3594) resulted in the highest initial CV mean of 0.9219

<img src="/readme_assets/19_plot_ridge_actual_predicted.png">

- The plots above show the actual against predicted values for the train and test set.


- There's already a relatively good fit, but it can be seen there are some negatively predicted values, as a result of linear relationships being inferred with the features.

<img src="/readme_assets/20_plot_ridge_residuals.png">

- Distribution of residuals show an approximately normal distribution, albeit with a few extreme residuals (approx. -300,000 and +150,000).


- Aside from this, the standardised residuals also show a relatively even pattern, with the exception of a few residuals at very high predicted prices (>400,000) which deviate quite far from zero.


- Homoscedasticity therefore not quite attained.

### Extra Trees Regressor
After fitting various ensemble models, the Extra Trees Regressor resulted in the highest CV mean of 0.9586

- The RMSE and MAE were also computed.


- RMSE showed relatively high values, perhaps owing to the presence of outliers (very expensive cars) within the dataset.


- MAE of the train set and test set were 22.69 and 1726.36 (in GBP) respectively.

<img src="/readme_assets/21_plot_etr_actual_predicted.png">

- As seen from the plot above, the model shows signs of heavy overfitting on the training set.
- However as the CV mean was also scoring highly, it was the model of choice for generalisability.

<img src="/readme_assets/22_plot_etr_residuals.png">

- The plots of residuals above, shows again that they follow an approximately normal distribution with the exception of a few extreme outliers.


- The standardised residuals plot shows some deviation of predictions when the predicted price was below GBP 100,000 and some large deviations at the GBP 300,000 mark (train set).


- The standardised residuals for the test set show relatively tight deviations at predicted prices below GBP 50,000 (test set).

<img src="/readme_assets/23_plot_etr_featimp.png">

- The feature importances above show relative feature importances, with the sum of all features adding to 1.0


- 'Horsepower' is of highest importance, closely followed by 'engine_size'.
    - In the future, another feature could be computed from horsepower and engine size (eg 'BHP per litre'), and used instead, as these two features are correlated with each other.
    
    
- It can also be seen that the make of the car (eg 'Ferrari', 'Bentley', and 'Rolls-Royce') are of relatively high importance. 
    - 'Martin' here is suspected to be mistaken as the the model name, from the make 'Aston Martin'.

[(back to top)](#back_to_top)

<a id='conclusions'></a>

## 6. Conclusions
---

### Highest scoring models:


| Model | CV mean (5 fold) | MAE (train) | MAE (test) | RMSE (train) | RMSE (test) |
| :- | :-: | :-: | :-: | :-: | :-: |
| Extra Trees Regressor | 0.9586 | 22.69 | 1726.36 | 1094.07 | 5286.36 |
| Random Forest Regressor | 0.9441 | 745.38 | 2048.36 | 2637.77 | 7179.35 |

<br>

- RMSE showed relatively high values, perhaps owing to the presence of outliers (very expensive cars) within the dataset.


- Extra Trees Regressor: MAE of the train set and test set were 22.69 and 1726.36 (in GBP) respectively.


### Features:

- Horsepower seems to be the most important feature for predicting the price of a vehicle, closely followed by other features such as 'engine_size', 'mileage' and 'year'.


- The make of the car, understandably, also ranks highly for predictive power on sales price.


### Limitations:

- This project is somewhat limited due to the fact that the number of observations (~15,000) was relatively small compared to the 400,000 listed cars on Autotrader.
    - This was a limitation borne from the web scraping process.
    
    
### Future Expansion:

- Further feature engineering such as implementing polynomial features may assist to improve model accuracy.
    - This is suggested from the trends found within the initial EDA.
    - Also implementing cross features in place of two features (such as 'BHP per litre', to replace 'horsepower' and 'engine_size') may simplify the model.
    
    
    
[(back to top)](#back_to_top)
