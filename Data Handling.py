# Importing essential libraries
import h5py
import numpy as np
import tensorflow.keras as Keras
from sklearn.model_selection import train_test_split


# Mount your drive if using google colab
from google.colab import drive
drive.mount('/content/drive')


# Defining a generator object to handle dataset
import h5py
import numpy as np
import tensorflow.keras as Keras
from sklearn.model_selection import train_test_split

class DataGenerator(Keras.utils.Sequence):
    def __init__(self,dataset,batch_size=5,shuffle=False):
        '''Initialise the dataset'''
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '''Returns length of the dataset'''
        
        return int(np.floor(len(self.dataset)/self.batch_size))

    def __getitem__(self,index):
        '''Returns items of given index'''
        
        indexes = self.indices[index * self.batch_size : (index+1) * self.batch_size]
        feature, label = self.__data_generation(indexes)
        
        return feature, label

    def __data_generation(self,indexes):
        '''Generates data from given indices'''
        
        feature = np.empty((self.batch_size,320,1024))
        label = np.empty((self.batch_size,320,1))
        for i in range(len(indexes)):
            feature[i,] = np.array(self.dataset[indexes[i]][0])
            label[i,] = np.array(self.dataset[indexes[i]][1]).reshape(-1,1)
            
        return feature,label

    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indices)



# Defining a class to read the h5 dataset
class DatasetMaker(object):

    def __init__(self,data_path):
        '''Read file from defined path'''

        self.data_file = h5py.File(data_path)

    def __len__(self):
        '''Returns length of the file'''

        return len(self.data_file)

    def __getitem__(self,index):
        '''Returns feature, label and index of varoius keys'''

        index += 1
        video = self.data_file['video_'+str(index)]
        feature = np.array(video['feature'][:])
        label = np.array(video['label'][:])

        return feature,label,index


# Defining a function to read and split the dataset into train and test
def get_loader(path, batch_size=5):
    '''Takes file path as argument and returns train and test set'''

    dataset = DatasetMaker(path)
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = 123)
    train_loader = DataGenerator(train_dataset)

    return train_loader, test_dataset


# Loading and splitting the dataset
train_loader, test_dataset = get_loader('fcsn_tvsum.h5')

