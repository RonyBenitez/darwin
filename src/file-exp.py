import glob
import cv2
path="/home/rony/Downloads/animal_dataset_v1_clean_check/*.jpg"
files=glob.glob(path)
dat=[]
base="./cows"
print(">>FILES",len(files))
indexes=['cattle_','bull_','beef','buffalo_']
for file in files:
    fg=file.split("/")[-1].split(".")[0]
    fg=fg.lower()
    for ind in indexes:
        if(ind in fg):
            print(fg)
            dat.append(file)
            cv2.imwrite(f'{base}/{file.split("/")[-1]}',cv2.imread(file))

print(">>FILES",len(dat))
    
        