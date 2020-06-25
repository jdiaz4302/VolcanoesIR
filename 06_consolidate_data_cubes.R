




# Make sure that
#   1. you are in the directory volcano-specific data folder
#      (data/volcano/)

library(reshape2)

good_df <- read.csv('good_df.csv')

# Changing strings from original data to subset data 
chip_files <- sub(x = sub(x = good_df$nighttime_volcano_files,
                  pattern = 'processed/', replacement = 'chips/'),
                  pattern = '.tif', replacement = '.RData')

# Loading each band at each time and storing it into an array
image_cube_series <- array(dim = c(length(chip_files), 5, 96, 96))
count <- 1
for (chip in chip_files) {
  base_file <- chip
  for (band in 10:14) {
    band_chip <- sub(x = base_file,
                     pattern = 'ImageData10',
                     replacement = paste0("ImageData",
                                          as.character(band)))
    data <- as.matrix(read.table(band_chip, as.is = TRUE))
    image_cube_series[count, band-9, ,] <- data
  }
  count <- count + 1
}

# Deconstructs the 4-D array into a table for easy export to Python
deconstructed_image_series <- melt(image_cube_series)
write.csv(deconstructed_image_series, 'image_series.csv')




