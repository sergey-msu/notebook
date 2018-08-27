# Plot simple x-y graph

from matplotlib import pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = x**2
plt.plot(x, y)
plt.show()


# Plot many graphs

plt.plot(x1, y1, 'o', x2, y2, '-')


# Plot histogram

plt.hist(sample, bins=3, normed=True)


# Plot distributions: features vs classes

n_features  = len(feature_names)
n_classes   = len(classes)
plot_number = 0
for feature_name in feature_names:
    for cls in classes:
        plot_number += 1
        plt.subplot(n_features, n_classes, plot_number)
        plt.hist(df[df['target'] == cls][feature_name])
        plt.title('Class mark: '+str(cls))
        plt.xlabel('X')
        plt.ylabel(feature_name)


# Plot algorithm surfaces with train objects scatter

from matplotlib.colors import ListedColormap

colors = ListedColormap(['red', 'blue', 'yellow'])
light_colors = ListedColormap(['lightcoral', 'lightblue', 'lightyellow'])
    
x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    
predictions = alg.predict(np.c_[xx.ravel(), yy.ravel()])
mesh_predictions = np.array(predictions).reshape(xx.shape)
plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, s = 100, cmap = colors)

plt.show()


# Plot many lines with gradient color depending upon some feature

from matplotlib.lines import Line2D

min_price = full_df['Стоимость покупки за кэш'].min()
max_price = full_df['Стоимость покупки за кэш'].max()

fig, axes = plt.subplots(2, 1, figsize=(12,6))
axes[0].set_title('Отказы с цветовым градиентом по цене')
legend_lines = [Line2D([0], [0], color=plt.cm.bwr(0), lw=4),
                Line2D([0], [0], color=plt.cm.bwr(700), lw=4)]
axes[0].legend(legend_lines, [min_price, max_price])

for i, row in full_df.iterrows():
    price = row['Стоимость покупки за кэш']
    price_color_idx = int(700*(price - min_price)/(max_price - min_price))
    axes[0].plot(lvl_cols, row[lvl_cols].values, color=plt.cm.bwr(price_color_idx))

plt.show()