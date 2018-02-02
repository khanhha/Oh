import pcl
import os

basepath = 'D:\\Projects\\Oh\data\\test_data\\'
name = 'normal_lucy_none-Slice-55_center_vn.obj'
filepath = basepath + name

if os.path.exists(filepath) == False:
    print('Error file does not exist: ' + filepath)
    exit(1)

reader = pcl.util.objreader.read(filepath)
verts = reader.vv
faces = reader._fv
for f in faces:
    assert len(f) == 3
    for i in range(3):
        f[i] = f[i] -1

measure = pcl.util.SlicePerimeter(verts, faces)
prm = measure.calc_perimeter()
print('perimeter = ' + str(prm))