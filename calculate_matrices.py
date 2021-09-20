import pickle
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

filenames = []
data = {}
# Evaluate Cardinalities
cardinalities_path = 'baselines/cardinality_estimation/results/deepDB/'
for item in glob.glob(cardinalities_path + 'imdb_light_model_based_budget_5_katerina*'):
    data[item.rsplit('/')[-1]] = pd.read_csv(item)
#
cardinality_error = {}
for key in data:
    cardinality_error[key] = sum(data[key]['latency_ms'])

# Evaluate AQP Queries
# aqp_path = 'baselines/aqp/results/deepDB/'
# for item in glob.glob(aqp_path + 'ssb_*'):
#     data[item.rsplit('/')[-1]] = pd.read_csv(item)
#
# cardinality_error = {}
# for key in data:
#     # overall = sum(data[key]['latency'])
#     overall = sum(data[key]['average_relative_error'])
#     cardinality_error[key] = overall

# i=13
# # Query by Query
#
# query = data[key].iloc[i]['query']
# cardinality_error = {}
# for key in data:
#     cardinality_error[key] = data[key].iloc[i]['latency']

# Confidence Intervals

# intervals_path = 'baselines/aqp/results/deepDB/'
# for item in glob.glob(intervals_path + '*intervals.csv'):
#     data[item.rsplit('/')[-1]] = pd.read_csv(item)
#
# cardinality_error = {}
# for key in data:
#     # overall = sum(data[key]['latency'])
#     overall = sum(data[key]['relative_confidence_interval_error'])
#     cardinality_error[key] = overall

clrs = ['rosybrown', 'lightcoral', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'salmon', 'tomato',
        'darksalmon', 'orangered', 'coral']
bars = range(1, len(cardinality_error) + 1)
plt.figure(figsize=(10, 5))
barlist = plt.bar(range(0, len(cardinality_error)), cardinality_error.values(), 0.5, color=clrs)
plt.xticks(range(0, len(cardinality_error)), bars)
labels = list(cardinality_error.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=clrs[label]) for label in range(len(labels))]
plt.legend(handles, labels)
plt.title('Cardinalities Latency - IMDB')
# plt.show()
plt.savefig('/home/kate/Documents/University/Database Systems/cardinalities_latency.png')