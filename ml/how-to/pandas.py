# read from CSV
df = pd.read_csv("file.csv")


# read from CSV without headers, tab separator
df = pd.read_csv("file.csv", header=False, sep='\t')


# write df to CSV
df.to_csv('file.csv', sep=',', header=True, index=None)


# df column types
df.dtypes()


# create DataFrame
data=[[1, 'no', False], 
      [3, 'yes', False], 
      [3, 'no', True]]
df = pd.DataFrame(data, columns=['position', 'factor', 'is_normal'])


# df summary
df.info()


# drop rows with NaN values
df.dropna(inplace=True)


# id there any NaN values?
df.isnull().values.any()


# fill NaN values with some value
df.fillna('some_value', inplace=True)
df.fillna(df.mean(), inplace=True) # fill with each column's mean


# fill NaN in some column based on other column

df['age'] = df.apply(lambda row: row['age'] if row['has_no_age']==0 else -1, axis=1)


# add row to df
row = {'Name': 'Stan', 'Age': 75, 'Height': 186}
df.append(row, ignore_index=True, inplace=True)


# add column to df
df['is_student'] = [False]*5 + [True]*2


# drop rows from df
df.drop([5,6], axis=0, inplace=True)


# drop columns from df
df.drop(['is_student'], axis=1, inplace=True)


# change column type
df['dob'] = df['dob'].apply(pd.to_datetime)


# pase datetimes on read
df = pd.read_csv('file.csv', parse_dates=['begin', 'end'])

# get new df by rows
df[10:200]


# get df slice by row/cols subsets
df.loc[[1:200], ['Name', 'Age']]


# get df slice by row/cols subsets
df.iloc[[1:200], [0, 2]]


# get row by inner frame index
df.ix[14]   # inner frame index is a hidden column


# rows filtering
df[(df['bob'] > pd.datetime(1985, 1, 1)) & (df['height'] > 167)]


# map categorical values to some another
sex_dict = { 'male': 0, 'female': 1 }
df = df.replace({'Sex': sex_dict})


# Some column hist, mean, std, median: 
df['Age'].value_counts()
df['Age'].mean()
df['Age'].std()
df['Age'].median()


#Iterate through rows
for index, row in df.iterrows():
    print row['c1'], row['c2']


# Correlation (Pearson or other) between two columns:
df['Age'].corr(df['Sex'])


# Correlation between column subsets

focus_cols = ['name', 'age']
df.corr().filter(focus_cols).drop(focus_cols)



# Read from CSV without header (otherwise first row will be deleted)
df = pd.read_csv("file.csv", header=None)


# Plot column histogram

df.plot(y='Height', kind='hist', color='red', title='Height (inch.) distribution')


# Scatter graph - dependence of one column of another (just points on a plane)

data.plot(x="Weight", y="Height", kind="scatter", title="Height-Weight dependence")


# Scatter pair plots

from pandas.plotting import scatter_matrix

scatter_matrix(data_numeric['height', 'width'], alpha=0.5, figsize=(10, 10))
plt.show()


# Cross table - кросс-таблица между двумя или более признаками

pd.crosstab


# Get categorial columns

cat_columns = df.select_dtypes(include='object').columns
df[cat_columns] = df[cat_columns].astype(np.str)