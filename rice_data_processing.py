# %%
import cv2
import pandas as pd
from spectral.io import envi
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

# %%
# Creating dataframe out of index csv file
pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"
df = pd.read_csv(pwd + "index.csv")
# %%
# Example for reading one file

hdr_path = pwd + "Data-VIS-20170111-1-room-light-off/CH12-01.hdr"
bin_path = pwd + "Data-VIS-20170111-1-room-light-off/CH12-01.raw"
img = envi.open(hdr_path, bin_path)
band = img[500:550,25:325,120]
plt.imshow(band, interpolation='nearest')
plt.show()
# %%
for slide in range(5):
    print(len(img[1,1,:]))
    unit = img[500:550,25+slide*50:75+slide*50,120]
    band = img[500:550,25:325,100]
    plt.imshow(unit, interpolation='nearest')
    plt.show()
# %%
subset_1 = ['HS1','CH12','AH1000','SVNI','91RH','DT8']
subset_2 = ['TB14','N54','NKB19','HQ15','BT6','NC7']
subset_3 = ['KB6','AH1000','HQ15','TQ14','KL25','NHN']
subset_4 = ['TC10','DTL2','KB16','BT6','KB27','CNC12']
subset_5 = ['CL61','NKB19','VH8','TX1','MT15','HL']
subset_6 = ['NBK','DT52','NBT1','NPT1','TB13','KB16']
# %%
species = df["Species Full Name"]
species_short = df["Species Short Name"]
folders = df["Folder"]
file_names = df["File Name"]
bundle_no = df["Bundle Number"]
subset = np.zeros(len(species))
make_df = []
for i in tqdm(range(len(species))):
    if species_short[i] in subset_1:
        subset[i] = 1
    elif species_short[i] in subset_2:
        subset[i] = 2
    elif species_short[i] in subset_3:
        subset[i] = 3
    elif species_short[i] in subset_4:
        subset[i] = 4
    elif species_short[i] in subset_5:
        subset[i] = 5
    elif species_short[i] in subset_6:
        subset[i] = 6
    make_df.append((species[i], species_short[i], bundle_no[i], folders[i], file_names[i], int(subset[i])))
df_bp = pd.DataFrame(make_df, columns =["Species Full Name","Species Short Name", "Bundle Number","Folder", "File Name","Subset"])
df_bp.to_csv(pwd + "subset_index.csv")
# %%
pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"
df = pd.read_csv(pwd + "subset_index.csv")
import time
def visualise_grain_row(df):
    species = df["Species Full Name"]
    folders = df["Folder"]
    file_names = df["File Name"]
    subsets = df["Subset"]
    pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"

    for i in range(len(species)):
        if subsets[i] != 0 and folders[i] == "Data-VIS-20170111-2-room-light-off":
            #print(species[i])
            hdr_path = pwd + folders[i] + '/' + file_names[i] + '.hdr'
            raw_path = pwd + folders[i] + '/' + file_names[i] + '.raw'
            img = envi.open(hdr_path, raw_path)

            print(subsets[i])
            band = img[500:550,25:325,100]
            #band = img[440:540,25:325,100]
            plt.clf()
            plt.imshow(band, interpolation='nearest')
            plt.show()
            time.sleep(0.1)
# %%
visualise_grain_row(df)
# %%
def open_hdr_rice(df):
    make_data = []
    species = df["Species Full Name"]
    folders = df["Folder"]
    file_names = df["File Name"]
    subsets = df["Subset"]
    pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"

    for i in range(len(species)):
        if subsets[i] != 0:
            hdr_path = pwd + folders[i] + '/' + file_names[i] + '.hdr'
            raw_path = pwd + folders[i] + '/' + file_names[i] + '.raw'
            img = envi.open(hdr_path, raw_path)

            for slide in range(5):
                unit = img[500:550,25+slide*50:75+slide*50,:]
                make_data.append((unit,subsets[i]))
                print()
    return make_data
            
            
# %%
make_data = open_hdr_rice(df)
# %%
df_sixclass = pd.DataFrame(make_data, columns =["Data","Label"])
df_sixclass.to_csv(pwd + "rice_data_sixclass.csv")
# %%
'''
Next steps:
1) Figure out image segmentation
2) Figure out how to label data: Tensor? DataLoader?

'''