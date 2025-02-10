import sys
import os

sys.path.append(os.path.abspath('./src'))  
from data_module.data_module_static import DataModuleStatic
from model.room_model import RaumModell

DataModule = DataModuleStatic()

DataModule.load_csv(r"data\training\240331_Dataset.csv")

# Zeitbereich abrufen
start, end = DataModule.get_time_range()

print( DataModule.get_timespan(start, end))

Raum = RaumModell()

Raum
