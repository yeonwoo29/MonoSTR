import os
import shutil


output_dir = "./dataset/label_2"
val_file = "./dataset/ImageSets/train.txt"
save_dir = "./dataset/label_2_train"   

os.makedirs(save_dir, exist_ok=True)

# val.txt 안의 id 불러오기
with open(val_file, "r") as f:
    val_ids = set(line.strip().replace(".png","") for line in f)

# output 안의 모든 txt 파일 순회
for fname in os.listdir(output_dir):
    if fname.endswith(".txt"):
        img_id = os.path.splitext(fname)[0]
        if img_id in val_ids:
            shutil.copy(os.path.join(output_dir, fname),
                        os.path.join(save_dir, fname))