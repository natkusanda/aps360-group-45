# %%
import cv2
import pandas as pd
from spectral.io import envi
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import time
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
#band = img[500:550,25:325,120]
band = img[:,:600,120]
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
subset_2 = ["R068","R998KBL","91RH","CL61","DTL2","HongQuang15"]
subset_2_extrafile = ["KimCuong111-01","H229-01","HaNa39-01","TQ14-01"]
subset_3 = ["LDA8","N98","NBP","NTP","A128","BT6"]
subset_3_extrafile = ["DA1-01","DaiThom8-01","TC10-01","VietThom8-02"]
subset_4 = ["KB6","SHPT1","SVN1","TB13","ThuanViet2","BacThomSo7"]
subset_4_extrafile = ["HaPhat28-02","LocTroi183-02","NC7-02","NDC1-02"]
subset_5 = ["CTX30","HS1","KB16","NPT1","NPT3","TB14"]
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
#hdr_path = pwd + "segmented-data-newclass/1/NV1-02_3.hdr"
#hdr_path = pwd + "segmented-data-newclass/6/NepDacSanLienHoa-01_2.hdr"
hdr_path = pwd + "segmented-data-newclass/6/PC10-02_0.hdr"
img_path = pwd + "segmented-data-newclass/6/PC10-02_0.img"
img = envi.open(hdr_path,img_path) # opens as array
band = img[:,:,120] # Reading one band. For all bands, ':' instead of 120
plt.imshow(band, interpolation='nearest')
plt.show()
# %%
'''
23/3/23
5. 
'''
# %% Visualising segmented units for a single example row
def visualise_all_grain_units(df, species_name, batch_no, index_array,class_num):
    species = df["Species Full Name"]
    folders = df["Folder"]
    file_names = df["File Name"]
    subsets = df["Subset"]
    pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"

    spec_file = species_name + '-' + batch_no

    for i in range(len(species)):
        if file_names[i] == spec_file:
            hdr_path = pwd + folders[i] + '/' + file_names[i] + '.hdr'
            raw_path = pwd + folders[i] + '/' + file_names[i] + '.raw'
            img = envi.open(hdr_path, raw_path)

            index_num = 0
            for indices in index_array:
                x_start = indices[0]
                y_start = indices[1]
                unit = img[y_start:y_start+60,x_start:x_start+60,:]
                #plt.imshow(unit, interpolation='nearest')
                #plt.show()

                save_path = pwd + "segmented-six-rice-type/" + str(class_num) + "/" + file_names[i] + "-" + str(index_num) + ".hdr"
                envi.save_image(save_path, unit, dtype=np.float32)
                index_num = index_num + 1



# %%
hdr_path = pwd + "Data-VIS-20170203-1-room-light-off/DT66-01.hdr"
bin_path = pwd + "Data-VIS-20170203-1-room-light-off/DT66-01.raw"
img = envi.open(hdr_path, bin_path)
# %%
x_start = 250
y_start = 470
band = img[y_start:y_start+60,x_start:x_start+60,150]
#band = img[y_start:y_start+60,:,120]
#band = img[:,:,120]
plt.imshow(band, interpolation='nearest')
plt.show()
# %%
select_species = ["CT286","NepCoTien","NepThomBacHai", "NV1", "VietHuong8", "DT66"]
# %%
DT66_01_index = [[25,40],[70,40],[110,40],[155,30],[195,35],[245,35],
    [30,105],[70,100],[115,105],[155,95],[195,95],[245,100],
    [35,180],[75,175],[110,170],[150,170],[195,165],[245,160],
    [30,240],[70,240],[110,235],[150,230],[195,225],[245,225],
    [30,300],[70,300],[110,300],[155,290],[195,280],[250,285],
    [35,365],[70,360],[115,355],[155,350],[195,340],[245,345],
    [30,420],[75,415],[115,415],[155,405],[200,400],[240,405],
    [35,475],[72,475],[115,475],[155,465],[200,470],[250,470]
]
visualise_all_grain_units(df,'DT66','01',DT66_01_index,0)
# %%
VietHuong8_01_index = [[5,85],[50,80],[90,80],[135,70],[175,80],[225,75],
    [5,140],[55,140],[90,140],[132,135],[170,135],[225,130],
    [5,200],[45,200],[90,195],[130,195],[175,195],[225,190],
    [10,270],[50,265],[90,260],[130,260],[175,260],[225,255],
    [5,325],[50,325],[88,325],[130,325],[170,325],[225,315],
    [4,390],[45,390],[85,385],[125,390],[170,380],[220,375],
    [5,455],[45,450],[85,445],[135,450],[175,440],[220,440],
    [5,510],[45,505],[90,510],[130,505],[175,500],[220,510]
]
visualise_all_grain_units(df,'VietHuong8','01',VietHuong8_01_index,1)
# %%
NV1_01_index = [[15,85],[70,85],[115,85],[170,80],[215,75],[260,75],
    [15,140],[70,135],[120,140],[165,135],[210,130],[255,130],
    [15,200],[70,195],[115,200],[160,195],[210,190],[255,190],
    [15,260],[70,260],[115,255],[165,255],[205,245],[255,245],
    [15,320],[70,320],[115,315],[165,320],[215,310],[255,310],
    [15,380],[65,380],[115,380],[160,375],[210,375],[255,370],
    [10,440],[65,440],[110,440],[165,440],[205,435],[255,435],
    [10,505],[65,500],[110,500],[160,505],[205,505],[250,500]
]
visualise_all_grain_units(df,'NV1','01',NV1_01_index,2)
# %%
NepThomBacHai_01_index = [[15,45],[70,50],[115,50],[165,55],[210,50],[250,50],
    [15,110],[65,110],[115,105],[165,105],[210,105],[255,105],
    [15,180],[65,180],[120,175],[165,175],[210,170],[255,170],
    [10,250],[60,245],[110,245],[155,240],[200,240],[245,235],
    [15,310],[65,310],[105,310],[155,310],[200,305],[245,305],
    [5,380],[50,385],[100,375],[150,380],[200,375],[245,370],
    [10,450],[50,440],[100,440],[145,440],[190,435],[240,430],
    [10,505],[60,500],[100,500],[145,495],[195,495],[240,490]
]
visualise_all_grain_units(df,'NepThomBacHai','01',NepThomBacHai_01_index,3)
# %%
CT286_01_index = [[15,50],[60,50],[105,45],[155,50],[205,35],[255,35],
    [15,115],[65,110],[110,110],[155,110],[200,100],[250,95],
    [15,190],[70,175],[110,170],[150,165],[200,160],[245,155],
    [15,255],[65,235],[110,235],[155,230],[200,230],[250,230],
    [10,315],[65,300],[105,300],[150,300],[200,300],[250,305],
    [10,375],[60,365],[105,370],[145,370],[205,365],[255,370],
    [10,430],[55,430],[100,435],[150,435],[200,430],[255,425],
    [10,490],[50,505],[95,505],[145,490],[205,490],[250,495]]
visualise_all_grain_units(df,'CT286','01',CT286_01_index,4)
# %%
NepCoTien_01_index = [[10,60],[60,50],[110,45],[165,50],[215,60],[270,60],
    [10,110],[60,105],[110,110],[160,110],[210,115],[265,120],
    [10,175],[55,170],[110,175],[160,175],[210,180],[265,185],
    [10,245],[55,240],[105,240],[155,245],[210,245],[260,240],
    [10,300],[60,290],[105,290],[155,295],[210,295],[265,295],
    [10,365],[50,360],[105,360],[155,360],[205,360],[265,360],
    [10,430],[55,425],[110,425],[155,420],[210,420],[265,415],
    [10,490],[60,490],[110,490],[165,485],[210,485],[265,485]]
visualise_all_grain_units(df,'NepCoTien','01',NepCoTien_01_index,5)
# %%
