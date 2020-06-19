



# Make sure that
#   1. destination folder (temp_gif_image_holder) is empty and
#      not containing scenes from other volcanoes
#   2. you are in the directory volcano-specific data folder
#      (data/volcano/)

good_df <- read.csv('good_df.csv')

chip_files <- sub(x = sub(x = sub(x = good_df$nighttime_volcano_files,
                  pattern = 'processed/', replacement = 'chips/'),
                  pattern = '.tif', replacement = '.RData'),
                  pattern = 'ImageData10', replacement = 'ImageData12')

image_cube_series <- array(dim = c(length(chip_files), 5, 96, 96))

count <- 1

for (chip in chip_files) {
  base_file <- chip
  for (band in 10:14) {
    band_chip <- sub(x = base_file,
                     pattern = 'ImageData12',
                     replacement = paste0("ImageData",
                                          as.character(band)))
    data <- as.matrix(read.table(chip, as.is = TRUE))
    image_cube_series[count, band-9, ,] <- data
  }
  count <- count + 1
}

for (i in 1:length(chip_files)) {
  png(paste0("../../temp_gif_image_holder/",
             sprintf("%04d", i), ".png"),
      units = 'in', height = 8, width = 8, res = 300)
  date = substr(strsplit(chip_files[i], '_')[[1]][3], start = 4, stop = 11)
  image(image_cube_series[i, 3, ,], col = grey(1:10000/10000),
        xaxt = 'n', yaxt = 'n',
        zlim = c(min(image_cube_series, na.rm = TRUE),
                 max(image_cube_series, na.rm = TRUE)),
        main = paste0('MM-DD-YYYY\n',
                      substr(date, start = 1, stop = 2), "-",
                      substr(date, start = 3, stop = 4), "-",
                      substr(date, start = 5, stop = 8)))
  dev.off()
}

