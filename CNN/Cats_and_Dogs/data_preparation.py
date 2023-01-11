import os, shutil

original_dataset_dir = r"E:/DeepLearning/CNN/dogs-vs-cats/train/train"

#1
base_dir = r"E:/DeepLearning/CNN/dogs-vs-cats/cats_and_dogs_small" #儲存少量資料集的目錄位置
if not os.path.isdir(base_dir): os.mkdir(base_dir) #若該目錄不存在，則建立該目錄

#2
train_dir = os.path.join(base_dir, "train") #訓練集
if not os.path.isdir(train_dir): os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, "validation") #驗證集
if not os.path.isdir(validation_dir): os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, "test") #測試集
if not os.path.isdir(test_dir): os.mkdir(test_dir)

#3
train_cats_dir = os.path.join(train_dir, "cats") #訓練的貓圖片
if not os.path.isdir(train_cats_dir): os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, "dogs") #訓練的狗圖片
if not os.path.isdir(train_dogs_dir): os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, "cats") #驗證的貓圖片
if not os.path.isdir(validation_cats_dir): os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, "dogs") #驗證的狗圖片
if not os.path.isdir(validation_dogs_dir): os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, "cats") #測試的貓圖片
if not os.path.isdir(test_cats_dir): os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, "dogs") #測試的狗圖片
if not os.path.isdir(test_dogs_dir): os.mkdir(test_dogs_dir)

#複製:1000貓圖片至train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

#複製1000:1500貓圖片至validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

#複製1500:2000貓圖片至test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

#複製:1000狗圖片至train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#複製1000:1500狗圖片至validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

#複製1500:2000狗圖片至test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('訓練用貓圖片張數:', len(os.listdir(train_cats_dir)))
print('訓練用狗圖片張數:', len(os.listdir(train_dogs_dir)))
print('驗證用貓圖片張數:', len(os.listdir(validation_cats_dir)))
print('驗證用狗圖片張數:', len(os.listdir(validation_dogs_dir)))
print('測試用貓圖片張數:', len(os.listdir(test_cats_dir)))
print('測試用狗圖片張數:', len(os.listdir(test_dogs_dir)))