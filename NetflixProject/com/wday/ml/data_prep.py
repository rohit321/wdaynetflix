import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from datetime import datetime
import os


class DataPrep:
    def __init__(self):
        pass


    def prepare(self):
        pass

    def combine_data(self):
        start = datetime.now()
        if not os.path.isfile('/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/data.csv'):
            print("here")
            data = open('/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/data.csv', mode='w')
            row = list()
            files = ['/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/combined_data_1.txt',
                     '/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/combined_data_2.txt',
                     '/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/combined_data_3.txt',
                     '/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/combined_data_4.txt']
            for file in files:
                print("reading from {}...".format(file))
                with open(file) as f:
                    for line in f:

                        line = line.strip()  # trimming
                        if line.endswith(':'):
                            movie_id = line.replace(':', '')
                        else:
                            row = [x for x in line.split(',')]
                            row.insert(0, movie_id)
                            data.write(','.join(row))
                            data.write('\n')
            data.close()
        print('Time taken: ', datetime.now() - start)
        df = pd.read_csv('/Users/rohit.pegallapati/PycharmProjects/NetflixProject/data/data.csv')
        df.columns = ['movie_id', 'cust_id', 'rating', 'date']
        return df





def main():
    prep = DataPrep()
    df = prep.combine_data()
    print(df)

if __name__ == '__main__':
    main()



