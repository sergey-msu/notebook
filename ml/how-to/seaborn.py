import seaborm as sns

# Scatter pair graph - pairs of columns distribution

sns.pairplot(df)


# Pairplot with classes

sns.pairplot(df, hue='target')


# Boxplot - dependence of some float feature of categorical one

sns.boxplot(x="price_category", y="height", data=data[["price_category", "height"]])


# Correlation matrix

sns.heatmap(data[some_features].corr(), square=True)


# Count plot (histogram) with target classes

sns.countplot(x='feature_1', data=df, hue='target')


# Tagret distribution (balance if target class etc)

sns.countplot(df['target'])


# Heat map to visualize any 2D array

sns.heatmap(X[:, :100], square=True)


# Visualize missing values

sns.heatmap(X_train.isnull(), cbar=False)


