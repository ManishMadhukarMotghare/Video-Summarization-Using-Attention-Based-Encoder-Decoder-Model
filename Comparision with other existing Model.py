import matplotlib.pyplot as plt
import pandas as pd

# F1-scores of different models
performance_dict = {'model_name' : ['TVSum unsupervised','SUM-GANdpp unsupervised','vsLSTM supervised','dppLSTM supervised','SUM-GANsup supervised',
                                    'Li et al. supervised','DR-DSN','HSA-RNN Supervised','DySeqDPP Supervised','CSNet','A-AVS supervised','AC-SUM-GAN','M-AVS supervised',
                                    'M-AVS(with extra layers) supervised(OURS)'],
                    'F1-score' : [51.3,51.7,54.2,54.7,56.3,52.7,57.6,59.8,58.4,58.8,59.4,60.6,61.0,62.0]}

performance_df = pd.DataFrame(performance_dict)
performance_df = performance_df.sort_values(['F1-score'])



# Visualizing F1-scores of different models
plt.figure(figsize = (15,9))
plt.barh(performance_df['model_name'],performance_df['F1-score'])
for i, v in enumerate(performance_df['F1-score']):
    plt.text(v + 0.5, i-0.1, str(v), color='blue')

plt.xlabel('F1-score')
plt.ylabel('Model Name')
plt.title('Comparison of F1-scores of various models', fontsize = 20)
