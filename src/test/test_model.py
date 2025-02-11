import sys
import os

sys.path.append(os.path.abspath('./src'))  
from data_module.data_module_static import DataModuleStatic
from model.room_model import RaumModell

DataModule = DataModuleStatic()

DataModule.load_csv(r"data\training\240331_Dataset.csv")

# Zeitbereich abrufen
#start, end = DataModule.get_time_range()

#DataModule.get_timespan(start, end)

Raum = RaumModell(dt = 1)

tmp = Raum.raumtemperatur_model(tmp_0          = 21.5357487923,
                                tmp_aussen     = DataModule.df["tmpAmbient1"],
                                sonnenleistung = DataModule.df["SunPow"],
                                orthogonalität = DataModule.df["sunOrtho"]
                                )

print(tmp)