
import multiprocessing
from multiprocessing import Semaphore
import pandas as pd
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset
from com.wday.ml.svd_algo import SVDMatrixFactorization
from com.wday.ml.nmf import NMFMatrixFactorization
from com.wday.ml.knn_base import KNNBase
from com.wday.ml.pmf import ProbabilisticMatrixFactorization
from com.wday.ml.slope_one import SlopeOneMatrixFactorization
from com.wday.ml.baseline_als import BaseLineALS
from com.wday.ml.basline_sgd import BaseLineSGD


class CrossValidationMapReduce:

    def __init__(self, city, algo_list, init=True):

        self.concurrency = 4;
        self.sema = Semaphore(self.concurrency)

        manager = multiprocessing.Manager()
        self.return_dict = manager.dict()

        self.algo_list = algo_list
        self.jobs = []
        self.city = city.replace(' ', '')
        self.algo_map = {}
        if init:
            self.algo_map[SVDMatrixFactorization().name] = SVDMatrixFactorization()
            self.algo_map[NMFMatrixFactorization().name] = NMFMatrixFactorization()
            self.algo_map[KNNBase().name] = KNNBase()
            self.algo_map[BaseLineALS().name] = BaseLineALS()
            self.algo_map[BaseLineSGD().name] = BaseLineSGD()
            self.algo_map[SlopeOneMatrixFactorization().name] = SlopeOneMatrixFactorization()
            self.algo_map[ProbabilisticMatrixFactorization().name] = ProbabilisticMatrixFactorization()

    def worker(self, algo, sema):
        sema.acquire()
        print("worker :: "+algo)
        A = self.algo_map.get(algo)
        data_full = pd.read_csv('data.csv')
        data_required = data_full[['cust_id', 'movie_id', 'rating']]
        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(data_required, reader)
        cv_results = cross_validate(A.algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs=1, verbose=False)
        res_df = pd.DataFrame.from_dict(cv_results).mean(axis=0)
        self.return_dict[algo]=res_df
        sema.release()


    def mapper(self):
        for algo in self.algo_list:
            print(algo)
            j = multiprocessing.Process(target=self.worker, name=algo, args=(algo, self.sema))

            self.jobs.append(j)
            j.start()

    def reducer(self):
        all_results = []
        for job in self.jobs:
            job.join()
            print(self.return_dict)
            res_df = self.return_dict[job.name]

            res_df = res_df.append(pd.Series([job.name], index=['algo']))
            all_results.append(res_df)
        results = pd.DataFrame(all_results).set_index('algo').sort_values('test_rmse')
        results.to_csv(self.city+'-cv-algos.csv')
        return results

    def get_best_algo(self):
        df = pd.read_csv(self.city + '-cv-algos.csv')
        return df['algo'][0]



def main():
    algo_list = [SVDMatrixFactorization().name, ProbabilisticMatrixFactorization().name,
                 SlopeOneMatrixFactorization().name, NMFMatrixFactorization().name,
                 BaseLineALS().name, BaseLineSGD().name, KNNBase().name]

    # algo_list = [KNNBase().name]


    mr = CrossValidationMapReduce('Netflix', algo_list)


    mr.mapper()
    mr.reducer()

    best_algo = mr.get_best_algo()


    print(best_algo)





if __name__ == '__main__':
    main()