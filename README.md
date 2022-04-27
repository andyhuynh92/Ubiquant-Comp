# Ubiquant-Comp

These are the Jupyter notebooks I used for the [Ubiquant market prediction](https://www.kaggle.com/competitions/ubiquant-market-prediction/) competition on Kaggle.

1. `Finding the distribution of the target.ipynb`

Here, I was trying to find the distribution of the target column with the `time_id` fixed. I found out that the target follows a $t$-distribution, and not a normal distribution.

2. `Getting stock data.ipynb`

Using the reverse engineered data, I grabbed the stock closing price using the Yahoo finance API.

3. `Computing correlation with pandas.ipynb` and `Computing correlation with polars.ipynb`

After grabbing the stock data, I computed the correlation of the stock price with each and every one of the features, both ignoring the `investment_id`, and also fixing the `investment_id`. I also used [Polars](https://www.pola.rs/) as well over Pandas. This was because when I first attempted to compute the correlation, it was going to take 2-3 days using an inefficient algorithm. This was when I learned about Polars and I decided to try it. It turns out that I had to rewrite the way I implemented splitting a dataframe to make the algorithm much much faster. 

4. `LGBM.ipynb`

Here I did a simple model using LightGBM. I wanted to reverse engineer the features in order to feature engineer new ones, but this never happened due to the processing that Ubiquant did to the data. This made it extremely difficult to properly introduce new features into the data. The cross validation method is combinatorial purged group $K$ fold, gotten [here](https://www.kaggle.com/code/gogo827jz/combinatorial-purged-group-k-fold/notebook), which was proposed in [__Advances in Financial Machine Learning__](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089). The trained models are pickled to be used on Kaggle to prevent crashing from extremely high memory usage.
