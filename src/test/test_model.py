import sys
import os

sys.path.append(os.path.abspath('./src'))  
from data_module.data_module_static import DataModuleStatic
from model.room_model import RaumModell

DataModule = DataModuleStatic()

DataModule.load_csv(r"data\training\240331_Dataset_01.csv")

# Zeitbereich abrufen
#start, end = DataModule.get_time_range()

#DataModule.get_timespan(start, end)

Raum = RaumModell(dt = 1)

tmp = Raum.run_model(DataModule.df)

print(tmp)