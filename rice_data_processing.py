# %%
import cv2
import pandas as pd
from spectral.io import envi
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
# %%
'''
0. How to read original HDR file 
NOTE: Differs slightly from reading segmented files
'''
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
'''
1. Organizing rice samples into subsets
'''

# %% Old subsets based on paper
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
# New subsets based on consistent gridding
subset_1 = ["BC15","CT286","NepCoTien","NepThomBacHai","ND9","NV1"]
subset_1_extrafile = ["NM14-2-01","DTH155-01","HoangLong-01","N54-01"]
subset_2 = ["R068","R99KBL","91RH","CL61","DTL2","HongQuang15"]
subset_2_extrafile = ["KimCuong111-01","H229-01","HaNa39-01","TQ14-01"]
subset_3 = ["LDA8","N98","NBP","NTP","A128","BT6"]
subset_3_extrafile = ["DA1-01","DaiThom8-01","TC10-01","VietThom8-02"]
subset_4 = ["KB6","SHPT1","SWN","TB13","ThuanViet2","BacThomSo7"]
subset_4_extrafile = ["HaPhat28-02","LocTroi183-02","NC7-02","NDC1-02"]
subset_5 = ["CTX30","HS1","KB16","NBT1","NBT3","TB14"]
subset_5_extrafile = ["NepKB19-02","VietHuong8-02","VS5-02","PD211-02"]
subset_6 = ["TQ36","NepThomHungYen","NN4B","PC10","GS55R","NepDacSanLienHoa"]
subset_6_extrafile = ["MT15-1-02","DMV58-02","N97-02","NBK-02"]
# %%
# Creating dataframe out of index csv file, adding subset info for new csv file
pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"
df = pd.read_csv(pwd + "index.csv")

species = df["Species Full Name"]
species_short = df["Species Short Name"]
folders = df["Folder"]
file_names = df["File Name"]
bundle_no = df["Bundle Number"]
subset = np.zeros(len(species))
make_df = []
for i in tqdm(range(len(species))):
    if species[i] in subset_1 or file_names[i] in subset_1_extrafile:
        subset[i] = 1
    elif species[i] in subset_2 or file_names[i] in subset_2_extrafile:
        subset[i] = 2
    elif species[i] in subset_3 or file_names[i] in subset_3_extrafile:
        subset[i] = 3
    elif species[i] in subset_4 or file_names[i] in subset_4_extrafile:
        subset[i] = 4
    elif species[i] in subset_5 or file_names[i] in subset_5_extrafile:
        subset[i] = 5
    elif species[i] in subset_6 or file_names[i] in subset_6_extrafile:
        subset[i] = 6

    make_df.append((species[i], species_short[i], bundle_no[i], folders[i], file_names[i], int(subset[i])))
df_bp = pd.DataFrame(make_df, columns =["Species Full Name","Species Short Name", "Bundle Number","Folder", "File Name","Subset"])
df_bp.to_csv(pwd + "new_subset_index.csv")
# %%
'''
2. Code for visualising rows, units
'''
# %%
pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"
df = pd.read_csv(pwd + "subset_index.csv")
# Visualising last row of all arrays
def visualise_grain_row(df):
    species = df["Species Full Name"]
    folders = df["Folder"]
    file_names = df["File Name"]
    subsets = df["Subset"]
    pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"

    for i in range(len(species)):
        print(file_names[i])
        hdr_path = pwd + folders[i] + '/' + file_names[i] + '.hdr'
        raw_path = pwd + folders[i] + '/' + file_names[i] + '.raw'
        img = envi.open(hdr_path, raw_path)

        band = img[480:575,15:330,100]
        plt.imshow(band, interpolation='nearest')
        plt.show()
        time.sleep(0.1)

# %%
visualise_grain_row(df)
# %% Visualising segmented units for a single example row
def visualise_grain_unit(df):
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

            for slide in range(5):
                unit = img[480:575,15+slide*50:65+slide*50,120]
                plt.imshow(unit, interpolation='nearest')
                plt.show()

# %%
visualise_grain_unit(df)
# %% 
'''
3. Segmenting images, writing to HDR files in labelled folders 
''' 
def open_hdr_rice(df):
    make_data = []
    species = df["Species Full Name"]
    folders = df["Folder"]
    file_names = df["File Name"]
    subsets = df["Subset"]
    pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"

    for i in tqdm(range(len(species))):
        if subsets[i] != 0:
            hdr_path = pwd + folders[i] + '/' + file_names[i] + '.hdr'
            raw_path = pwd + folders[i] + '/' + file_names[i] + '.raw'
            img = envi.open(hdr_path, raw_path)

            for slide in range(6):
                unit = img[480:575,15+slide*50:65+slide*50,:]
                save_path = pwd + "segmented-data-newclass/" + str(subsets[i]) + "/" + file_names[i] + "_" + str(slide) + ".hdr"
                envi.save_image(save_path, unit, dtype=np.float32)
# %%
pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"
df = pd.read_csv(pwd + "new_subset_index.csv")
open_hdr_rice(df)
# %% 
'''
4. Code for reading segmented HDR file 
'''
hdr_path = pwd + "segmented-data-newclass/1/NV1-02_3.hdr"
img = envi.open(hdr_path) # opens as array
band = img[:,:,120] # Reading one band. For all bands, ':' instead of 120
plt.imshow(band, interpolation='nearest')
plt.show()
# %%