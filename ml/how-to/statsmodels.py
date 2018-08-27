# Доверительный z-интервал

from statsmodels.stats.weightstats import zconfint

x = [1, 2, 1, 1, 1]
zint = zconfint(x, alpha=0.05, alternative='two-sided', ddof=1.0) # (0.8080072030919891, 1.5919927969080108)


# Доверительный t-интервал для среднего

from statsmodels.stats.weightstats import _tconfint_generic

x = [1, 2, 0, 3, 1, 1, 2, 4, 5, 6]
n = len(x)
mean = x.mean()
sigma = x.std(ddof=1)/math.sqrt(n)
_tconfint_generic(mean, sigma, n-1, 0.05, 'two-sided')  # (1.0994, 3.9006)) - 95% доверительный интервал для среднего


# Доверительный интервал для доли

from statsmodels.stats.proportion import proportion_confint

normal_interval = proportion_confint(n_positive, n_all, alpha=0.05 method = 'normal')  # 95% confident interval


# Размер выборки для интервала заданной ширины

from statsmodels.stats.proportion import samplesize_confint_proportion

n_samples = samplesize_confint_proportion(random_sample.mean(), half_length=0.01, alpha=0.05) # 95% confident interval
n_samples = int(np.ceil(n_samples)) # интервал ширины 0.02


# Сравнить наблюденную выборку с выборкой из нек сл.в. (фактически, проверить гипотезу о некотором распределении наблюдаемых данных)

observed_freqs = np.array([1, 2, 1, 1, 2, 3, 1, 1, 1])
expected_freqs = observed_freqs.mean()*np.ones_like(observed_freqs)  # проверка на равномерность исходного распределения
cs = stats.chisquare(observed_freqs, expected_freqs, ddof = 1) # statistic=2.923076923076923, pvalue=0.8920258540224891 > 0.05 - гипотезу не отвергаем


# ку-ку график (для проверки предположения о нормальности)

stats.probplot(df['day_calls'].values, dist="norm", plot=plt)
plt.show()