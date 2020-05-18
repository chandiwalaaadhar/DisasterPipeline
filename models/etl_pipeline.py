# import libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine



# load messages dataset
messages = pd.read_csv('messages.csv')
messages.head()



# load categories dataset
categories = pd.read_csv('categories.csv')
categories.head()



# merge datasets
df = messages.merge(categories)
df.head()




# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand=True)
categories.head()



# select the first row of the categories dataframe
row = categories.iloc[0,:]

# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x: x[:-2])
print(category_colnames)


# rename the columns of `categories`
categories.columns = category_colnames
categories.head()



for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype('str').str.split('-').str[1]
    
    # convert column from string to numeric
    categories[column]= pd.to_numeric(categories[column])
categories.head()




# drop the original categories column from `df`
df= df.join(categories)
df.drop(columns=['categories'],inplace=True)
df.head()








# check number of duplicates
df.duplicated().value_counts()





# drop duplicates
df.drop_duplicates(inplace=True)



# check number of duplicates
df.duplicated().value_counts()





engine = create_engine('sqlite:///data.db')
df.to_sql('df', engine, index=False)