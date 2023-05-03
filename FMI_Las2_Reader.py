# this function reads the exported Las2 files from Geolog
import numpy as np
#=================================================================
def FMI_Las2_Reader(fname):
    f= open(fname,'r')
    lines = f.readlines()
    f.close()
    return lines
def isfloat(string):
    try:
        float(string)
        return True
    except:
        return False

def get_numeric_values(string_list):
    index = []
    for i, s in enumerate(string_list):
        if isfloat(s):
            index.append(s)
    return index

def FMi_Las2_cleaner(lines):
    version_key = []
    ascii_key = []
    for i in range(len(lines)):
        if '~Version' in lines[i]:
            version_key.append(i)
        elif '~Ascii' in lines[i]:
            ascii_key.append(i)

    for j in range(len(version_key)):
        temp=lines[(version_key[j])- 1].strip().split(' ')
        if isfloat(temp[0]):
            lines[version_key[j]:] = []
            break

    for j in range(len(ascii_key)):
        if ascii_key[j] < len(lines):
            temp = lines[ascii_key[j]+1].strip().split(' ')
            if isfloat(temp[0]):
                lines[:ascii_key[j]+1] = []

   # read depth values and keep an index
    Depth = []
    for i in range (len(lines)):
        temp = lines[i].strip().split(' ')
        if len(temp) == 1:
            Depth.append([i,temp[0]])


    single_line_192 = []
    for j, d in enumerate(Depth):

        for k in range(1,40):
            temp= lines[d[0]+k].strip().split(' ')
            temp = get_numeric_values(temp)
            for i in temp:
                single_line_192.append(float(i))
    image = np.reshape(single_line_192,(-1,192))

    return Depth, image




