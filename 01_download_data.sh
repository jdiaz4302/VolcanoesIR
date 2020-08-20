# Downloads all ASTER imagery from beginning of data record
# to May 1, 2020 for specified volcano
# THINGS TO SPECIFY
# 	1. coord_1 - lon of approx geographic midpoint determined by Google Maps satellite imagery
#	2. coord_2 - lat of approx geographic midpoint determined by Google Maps satellite imagery
#	3. volcano_name - directory within the data folder where data will be stored
# MAKE SURE YOU ARE IN THE MAIN PROJECT DIRECTORY (i.e. VolcanoesIR)
./Daac2Disk_ubuntu18 --shortname AST_L1T --versionid 003 --browse --begin 1999-12-18 --end 2020-05-1 --bbox coord_1 coord2 coord1 coord2 --outputdir data/volcano_name/
