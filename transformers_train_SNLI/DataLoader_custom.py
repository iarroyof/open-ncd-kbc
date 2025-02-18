# Esta funcion crea un Data Loader para cargar los datos de SNLI, atomic y conceptnet
from torch.utils.data import Dataset
import json

class DataLoader_custom(Dataset):
    def __init__(self, data_dir, split):
        # Abriendo un archivo json
        with open(data_dir, 'r') as file:
            dic_data = json.load(file)
        self.X, self.y = self.separar(dic_data, split)
        self.samples = len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.samples
    
    def separar(self, data, split):
        X=[]
        y=[]
        for i in range(len(data[split])):
            X.append(data[split][i][0])
            y.append(data[split][i][1])
        return X, y
        
data_dir = ["SNLI_procesados_tuplas.json","Datos_pocesados_atomic_v2.json"]
split = "test"      # Seleccion de test, dev o train
data_SNLI = DataLoader_custom(data_dir[0], split) # Base de datos con la cual se entrenara el modelo y particion

print(data_SNLI.__len__())
print(data_SNLI.__getitem__(8))
