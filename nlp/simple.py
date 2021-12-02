from pycaret.datasets import get_data
data = get_data('kiva')

#check the shape of data
data.shape

# sampling the data to select only 1000 documents
data = data.sample(1000, random_state=786).reset_index(drop=True)
data.shape