import numpy as np
import pandas as pd


from surprise.model_selection import train_test_split
from surprise import Reader, Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV


class Classifier:
    def __init__(self, name, algo, param_grid):
        self.name = name
        self.algo = algo
        self.param_grid = param_grid

    def split_data(self,percent):
        print('started split')
        movie_df = pd.read_csv('/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/data.csv')
        req_data = movie_df[['cust_id', 'movie_id', 'rating']]
        reader = Reader(rating_scale=(1.0, 5.0))
        self.data = Dataset.load_from_df(req_data, reader)
        self.tr, self.te = train_test_split(self.data, test_size=percent)
        print('completed split')


    def train(self, train):
        self.algo.fit(train)

    def test(self, test):
        self.predictions = self.algo.test(test)

    def accuracy_rmse(self):
        return accuracy.rmse(self.predictions)

    def predict(self, user_id, restaurant_id):
        return self.algo.predict(uid=user_id, iid=restaurant_id, clip=True)


    def persist_prediction(self):
        recommendations = pd.DataFrame(columns=['cust_id', 'movie_id', 'actual_rating', 'recommended_rating'])

        for prediction in self.predictions:
            recommendations = recommendations.append(
                {'cust_id': prediction.uid, 'movie_id': prediction.iid, 'actual_rating': prediction.r_ui,
                 'recommended_rating': prediction.est}, ignore_index=True)
        recommendations.to_csv('recommendations-'+self.name+'.csv')

    def tune(self):
        print('tune called')
        self.split_data(0.25)
        print('algo')
        print(self.algo)
        res = not self.param_grid
        if res:
            return {}
        gs = GridSearchCV(self.algo, self.param_grid, measures=['rmse'], cv=2, verbose=10)
        gs.fit(self.tr)
        params = gs.best_params['rmse']
        return params

    def process(self):
        self.train(self.tr)
        self.test(self.te)
        print(self.accuracy_rmse())
        self.persist_prediction()

