import numpy as np
import pandas as pd
from PIL import Image
#===================================================
def find_closest_value(vector,value):
    temp = [abs(i-value) for i in vector]
    indx = temp.index(min(temp))
    return vector[indx], indx

#===================================================
def read_excel_col(file_name):
    wb = pd.ExcelFile(file_name)
    Sheet1 = wb.parse('Sheet1')
    Depth = Sheet1['Depth']
    facies = Sheet1['New Code']
    return [Depth, facies]
#===================================================
def Image_Generator(facies,path,dir):
    j=0
    count = 1
    while j + 192 < len(facies):
        temp_img = facies[j:j+192]
        im = Image.fromarray(temp_img.astype(np.uint8))
        cc = im.save(path + dir + '\\image_'+str(count)+'.png')
        j = j + 4
        count +=1
    return 'end'
#===================================================
def FMI_LAS2_Generator(Depth,logs,Well_Name):
    LAS_text = []
    #---- header-------
    start = Depth[0]
    stop = Depth[-1:]
    step = Depth[2]-Depth[1]
    LAS_Header = ['# LAS format log file generated in Python\n'
                  '# Units are always meters in this version of the generator\n '
                  '# ============================================================\n'
                  '~Version information\n'
                  'VERS.   2.0: \n'
                  'WRAP.   NO: \n'
                  '# ============================================================\n'
                  '~Well\n'
                  'STRT.m      ' + str(start)+':\n'
                  'STOP.m      ' + str(stop)+':\n'
                  'STEP.m       ' + str(step)+':\n'
                  'NULL.  - 999.250000:\n'
                  'COMP.            : COMPANY\n'
                  'WELL.  ' + str(Well_Name)+'   : WELL\n'
                  'FLD.             : FIELD\n'
                  'LOC.             : LOCATION\n'
                  'SRVC.            : SERVICE COMPANY\n'
                  'DATE.   2022 - 09 - 20  16: 16:16: Log Export Date    {yyyy - MM - dd  HH: mm:ss}\n'
                  'PROV.            : PROVINCE\n'
                  'UWI.             : UNIQUE WELL ID\n'
                  'API.             : API NUMBER\n'
                  '# ============================================================\n'
                  '~Curve\n'
                  'DEPT.m           : DEPTH\n'
                  'Python_Facies_Associations_log._: Python_Facies_Associations_log\n'
                  '~Parameter\n'
                  '# ==================================================================\n'
                  '~Ascii\n' ]
    for i in range(len(Depth)):
        LAS_Header.append(str(Depth[i])+ ' '+str(logs[i])+'\n')
    # ---- write to file -------
    with open('F:\\My Files\\project_temp\\Python_facies.LAS','w') as f:
        f.writelines(LAS_Header)

    return

