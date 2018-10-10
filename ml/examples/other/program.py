import time as t
import datetime as dt
import utils
#from tests.pipeline_test import main as m, title
#from tests.example_titanic import main as m, title
#from course1_math_and_python.root import main as m, title
#from course2_supervised_learning.root import main as m, title
#from course3_structure_in_data.root import main as m, title
#from course4_data_conclusions.root import main as m, title
#from course5_applications.root import main as m, title
from course6_final_project.root import main as m, title


def main(args=None):

    job()
    return




    utils.PRINT.HEADER(title())
    print('STARTED ', dt.datetime.now())
    start = t.time()
    m()
    end = t.time()
    utils.PRINT.HEADER('DONE in {}s ({}m)'.format(round(end-start, 2), round((end-start)/60, 2)))

    return





import numpy as np
import pandas as pd

def job():

    df, stats = get_data()
    df['date'] = pd.to_datetime(df['date'])
    stats['min_date'] = pd.to_datetime(stats['min_date'])
    stats['max_date'] = pd.to_datetime(stats['max_date'])

    feats = []

    user_ids = df['userid'].unique()
    for user_id in user_ids:
        user_df = df[df['userid'] == user_id]
        user = stats[stats['userid'] == user_id].iloc[0, :]
        fs = [user_id]

        # f1: user lifetime
        L = (user['max_date'] - user['min_date']).days + 1
        fs.append(L)

        # f2: delta_1
        d1 = (user_df.iloc[0]['date'] - user['min_date']).days
        fs.append(d1)

        # f3: delta_2
        d2 = (user['max_date'] - user_df.iloc[-1]['date']).days
        fs.append(d2)

        # f4: pct
        # TODO
        fs.append(-1)

        # f5: m1
        m1 = user_df[user_df['type'] == 0]['value'].mean()
        fs.append(m1)

        # f6: m2
        m2 = user_df[user_df['type'] == 1]['value'].mean()
        fs.append(m2)

        # f7: m3
        m3 = user_df[user_df['type'] == 0]['value'].max()
        fs.append(m3)

        # f8: m4
        m4 = user_df[user_df['type'] == 1]['value'].max()
        fs.append(m4)

        # f9: chaotic (0, 1 - not chaotic, 0.5 - chaotic)
        ch = np.abs(np.diff(user_df['type'].values)).sum()
        fs.append(ch)

        ## f9,10: D1, D2
        #ts = []
        #rs = []
        #ts_min, ts_max = None, None
        #ctype = 0
        #for i, row in user_df.iterrows():
        #    date  = row['date']
        #    type  = row['type']

        #    if ts_min is None:
        #        ts_min = date
        #    elif type == ctype:
        #        ts_max = date
        #    else:
        #        p = (ts_max - ts_min).days
        #        (ts if ctype == 0 else rs).append(p)
        #        ctype = type
        #D1 = np.array(ts).mean()
        #D2 = np.array(rs).mean()

        #fs.append(D1)
        #fs.append(D2)

        feats.append(fs)

    feat_df = pd.DataFrame(feats, columns=['userid',
                                           'L',
                                           'd1', 'd2',
                                           'pct',
                                           'm1', 'm2', 'm3', 'm4',
                                           'ch'])
    print(feat_df.head())

    return



def get_data():
    df = pd.DataFrame(
         {
             'userid': [
                 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
                 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102],
             'date': [
                 '2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-02-01', '2017-02-02', '2017-02-03', '2017-03-11', '2017-04-01', '2017-04-11', '2017-04-12',
                 '2017-05-01', '2017-05-02', '2017-05-03', '2017-06-04', '2017-06-05', '2017-06-06', '2017-06-07', '2017-06-08', '2017-07-01', '2017-07-11', '2017-07-15',
                 '2017-05-01', '2017-05-02', '2017-05-06', '2017-05-07', '2017-05-08', '2017-07-02', '2017-07-03', '2017-07-04', '2017-08-01', '2017-08-05', '2017-08-09',
                 ],
             'value': [
                     1.0, 0.5, 1.0, 2.5, 1.0, 1.0, 3.0, 5.6, 1.0, 1.0, 2.0,
                     10.0, 5.0, 10.0, 25.0, 10.0, 10.0, 30.0, 56.0, 10.0, 10.0, 20.0,
                     0.5, 0.5, 1.0, 1.5, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 3.0,
                 ],
             'type': [
                 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
                 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
                 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                 ]
         })

    stats = pd.DataFrame(
        {
            'userid':   [100, 101, 102],
            'min_date': ['2017-01-01', '2017-01-01', '2017-04-30'],
            'max_date': ['2017-04-12', '2017-09-01', '2018-01-01']
            })

    return df, stats






if __name__=='__main__':
    main()