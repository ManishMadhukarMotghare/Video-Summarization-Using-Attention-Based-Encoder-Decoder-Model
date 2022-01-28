import matplotlib.pyplot as plt
import pandas as pd

# Performance metrics of our model
model_perf_dict = {'score' : ['precision', 'recall', 'f1-score'],
                   'average' : [54.6, 72., 62.0],
                   'can get upto' : [55.5,72.9,63.0]}
model_perf_dict = pd.DataFrame(model_perf_dict)


# Visualizing performance metrics of our model
plt.figure(figsize = (15,6))
plt.barh(model_perf_dict['score'],model_perf_dict['average'],color = 'blue')
plt.barh(model_perf_dict['score'],model_perf_dict['can get upto'],color = 'blue', alpha = 0.5)
for i, v in enumerate(model_perf_dict['can get upto']):
     plt.text(v + 0.5, i, str(v), color='green', fontsize = 15)

for i, v in enumerate(model_perf_dict['average']):
     plt.text(v - 2.7, i, str(v), color='white', fontsize = 15)

plt.xlabel('Score')
plt.ylabel('Score Type')
plt.title('Model Performance metrics', fontsize = 20)


