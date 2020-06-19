




################# THE BIG ASSUMPTIONS
# 1. We are in the volcano-specific data folder, which has
#     i. the "good" csv
#    ii. the processed data in a /processed/ folder
# 2. That you want the image/matrix data saved in the current directory

# PRIOR TO THIS
# run: "mkdir chips" within the volcano-named directory (within the data directory)


library(raster)
library(MASS) # for saving the matrix

good_df <- read.csv('good_df.csv')


name = ""
coord_x =  # lon
coord_y =  # lat
  
  
# Establish band conversion parameters
UCC <- matrix(nrow = 1, ncol = 14)
K1 <- matrix(nrow = 1, ncol = 14)
K2 <- matrix(nrow = 1, ncol = 14)

UCC[10] <- 0.006822 
K1[10] <- 3040.136402 
K2[10] <- 1735.337945
UCC[11] <- 0.006780 
K1[11] <- 2482.375199 
K2[11] <- 1666.398761
UCC[12] <- 0.006590
K1[12] <- 1935.060183 
K2[12] <- 1585.420044
UCC[13] <- 0.005693
K1[13] <- 866.468575
K2[13] <- 1350.069147
UCC[14] <- 0.005225 
K1[14] <- 641.326517 
K2[14] <- 1271.221673

#count = 1
par(mfrow = c(1, 2), asp = 1)
for (file in good_df$nighttime_volcano_files) {
  for (i in 10:14) {
    # Going over all the TIR bands
    split_file_name <- strsplit(file, '_')[[1]]
    split_file_name[6] <- paste0(substr(split_file_name[6],
                                        start = 1,
                                        stop = 9),
                                 as.character(i))
    file <- paste(split_file_name, collapse = '_')
    
    # Reading and ignoring the out of extent areas
    r <- raster(file)
    r[r == 0] <- NA
    # Reprojecting into CRS with known reference points
    r <- projectRaster(r,
                       crs = CRS("+proj=longlat +datum=WGS84"))
    # Performing temperature conversions
    r_rad <- (r - 1)*UCC[i]
    r_T <- (K2[i] / (log((K1[i] / r_rad) + 1))) - 273.15
    
    # Viewing the temperature image with the vent/summit indicated
    plot(r_T,
         col = grey(1:1000/1000), colNA="black")
    points(coord_x, coord_y, cex = 3, col = 'red')
    
    # Finding the index of that vent/summit
    row_col <- rowColFromCell(r_T, cellFromXY(r_T, cbind(coord_x, coord_y)))
    row <- row_col[1]
    col <- row_col[2]
    # Adding a buffer around that index
    r_T_chip <- as.matrix(r_T)[(row-47):(row+48), (col-47):(col+48)]
    # Viewing the 48x48 pixel subset
    image(r_T_chip, col = grey(1:1000/1000))
    
    # Formatting the RData file to save and saving the 48x48 subset
    chip_file <- sub(x = sub(pattern = "processed/",
                             replacement = "chips/",
                             x = file),
                     pattern = ".tif",
                     replacement = ".RData")
    write.matrix(r_T_chip, file = chip_file)
  }
}




