# %%
import cv2
import pandas as pd
from spectral.io import envi
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
# Creating dataframe out of index csv file
pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"
df = pd.read_csv(pwd + "index.csv")
# %%
# Example for reading one file
#hdr_path = pwd + "Data-VIS-20170111-1-room-light-off/CH12-02.hdr"
#bin_path = pwd + "Data-VIS-20170111-1-room-light-off/CH12-02.raw"
#hdr_path = pwd + "Data-VIS-20170111-1-room-light-off/BC15-01.hdr"
#bin_path = pwd + "Data-VIS-20170111-1-room-light-off/BC15-01.raw"

hdr_path = pwd + "Data-VIS-20170113-1-room-light-off/HS1-01.hdr"
bin_path = pwd + "Data-VIS-20170113-1-room-light-off/HS1-01.raw"
img = envi.open(hdr_path, bin_path)
band = img[500:600,:,100]
plt.imshow(band, interpolation='nearest')
plt.show()
# %%
# Attempt at segmenting image 
for i in range(6):
    for j in range(8):
        band = img[35+j:35+j+50,i:50+i,100]
        plt.imshow(band, interpolation='nearest')
        plt.show()
# %%
subset_1 = ['HS1','CH12','AH1000','SVNI','91RH','DT8']
subset_2 = ['TB14','N54','NKB19','HQ15','BT6','NC7']
subset_3 = ['KB6','AH1000','HQ15','TQ14','KL25','NHN']
subset_4 = ['TC10','DTL2','KB16','BT6','KB27','CNC12']
subset_5 = ['CL61','NKB19','VH8','TX1','MT15','HL']
subset_6 = ['NBK','DT52','NBT1','NPT1','TB13','KB16']

def open_hdr_rice(df):
    species = df["Species Full Name"]
    folders = df["folder"]
    file_names = df["File Name"]
    pwd = "/Users/nkusanda/Desktop/THIRD_YEAR_WINTER/APS360/rgb-vis-nir-rice/"

    for i in tqdm(range(len(species))):
        hdr_path = pwd + folders[i] + '/' + file_names[i] + '.hdr'
        raw_path = pwd + folders[i] + '/' + file_names[i] + '.raw'
        img = envi.open(hdr_path, bin_path)

        for i in range(6):
            for j in range(8):
                band = img[35+j:35+j+50,i:50+i,100]
                plt.imshow(band, interpolation='nearest')
                plt.show()

# %%
species = df["Species Full Name"]
species_short = df["Species Short Name"]
folders = df["Folder"]
file_names = df["File Name"]
bundle_no = df["Bundle Number"]
subset = zeros(len(species))
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
    make_df.append((species[i], species_short[i], bundle_no[i], folders[i], file_names[i], subset[i]))
df_bp = pd.DataFrame(make_df, columns =["Species Full Name","Species Short Name", "Folder", "File Name","Subset"])
df_bp.to_csv(pwd + "subset_index.csv")

# %%
'''
Next steps:
1) Figure out image segmentation
2) Figure out how to label data: Tensor? DataLoader?

'''