import os
import shutil
def copy_rename(old_file_name, new_file_name):
        src_dir= os.curdir
        print src_dir
        dst_dir= os.path.join(os.curdir , "subfolder")
        print dst_dir
        src_file = os.path.join(src_dir, old_file_name)
        print src_file
        shutil.copy(src_file,dst_dir)
        
        dst_file = os.path.join(dst_dir, old_file_name)
        print dst_file
        new_dst_file_name = os.path.join(dst_dir, new_file_name)
        print new_dst_file_name
        os.rename(dst_file, new_dst_file_name)


def main():
    indexes = [217, 218, 221, 222, 223, 225, 226, 230, 231, 234]

    for i in range(0, 10):
        # load an image to search for faces
        idx = '0' + str(indexes[i])
        if indexes[i] < 10:
            idx = '00' + idx
        elif (indexes[i] >= 10 and indexes[i] < 100):
            idx = '0' + idx
        face = "image_" + idx + ".jpg"
        new_face = '10_0' + str(i) + ".jpg"
        copy_rename(face,new_face)

if __name__ == "__main__":
    main()