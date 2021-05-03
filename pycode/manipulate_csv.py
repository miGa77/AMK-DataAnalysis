import pandas as pd

filename = 'image_data_training'
filein = rf'../handwritten_digits/{filename}.csv'
fileout = rf'../handwritten_digits/{filename}_cutted.csv'

data = pd.read_csv(filein).astype('float32')
data_cutted = data.iloc[:, 2:]

label = data_cutted.iloc[:, -1]
label_list = label.tolist()
new_label_list = [x + 26 for x in label_list]

features = data_cutted.iloc[:, 0:-1]

frames = [pd.Series(new_label_list), features]

result = pd.concat(frames, axis=1, join='inner')
result = result.astype(int)

result.to_csv(fileout, index=False, header=False)
