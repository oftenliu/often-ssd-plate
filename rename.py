import os
def traverse(f,imageset):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            ext = os.path.splitext(tmp_path)[-1][1:]
            if ext == "tfrecord" or ext == "png":
                #print('文件: %s'%tmp_path)
                imageset.append(tmp_path)
        else:
            print('文件夹：%s'%tmp_path)
            traverse(tmp_path,imageset)

path = './ccpd_test_tfrecord'
#path = '../test'
imageset = []
traverse(path,imageset)
for file in imageset:
    newfile = file.replace("voc_2012", "ccpd")
    os.rename(file, newfile)