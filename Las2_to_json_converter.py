from FMI_Las2_Reader import  FMI_Las2_Reader
from FMI_Las2_Reader import FMi_Las2_cleaner
from timeit import timeit as timer
#======================= prep the data ========================
fname = 'CRC2_1167_1169-5.las'   # insert the las file name (exported from Geolog here)
st = timer()
lines = FMI_Las2_Reader(fname)
end = timer()

Depth, image = FMi_Las2_cleaner(lines)
print(end-st)
#================write to json ==================================
import json
aDict = {'image':image.tolist(),'Depth':Depth}
jsonstring = json.dumps(aDict)
with open('CRC2_1167_1169-5.json','w') as f:  # insert the output user defined json file name here
    f.write(jsonstring)
f.close()