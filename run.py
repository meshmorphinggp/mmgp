import os

print(">> untar datasets")
os.system("tar -xf data/datasets.tar.xz")

print(">> pretreat data")
os.system("python pretreatData.py")

print(">> train")
os.system("python train.py")

print(">> predictFields")
os.system("python predictFields.py")

print(">> plotFields")
os.system("python plotFields.py")

print(">> predictScalars")
os.system("python predictScalars.py")

print(">> plotScalars")
os.system("python plotScalars.py")

print(">> computeAndPlotHorsParamFields")
os.system("python computeAndPlotHorsParamFields.py")
