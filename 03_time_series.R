




################# THE BIG ASSUMPTIONS
# 1. We are in the volcano-specific data folder, which has
#     i. the data quality csv
#    ii. the processed data in a /processed/ folder
# 2. That you want the figures saved in a folder
#    that is two cd's back ('../../') - i.e. the VolcanoesIR folder


# Import quality indicators
quality_df <- read.csv('quality_df.csv')
# I accidently labeled an image where Puuoo is out of the extent
# as good, so this is correcting that
#quality_df[grep("00302242012084238", quality_df$nighttime_volcano_files),]$quality <- "n"
#write.csv(quality_df, "quality_df.csv")
#quality_df <- read.csv('quality_df.csv')

library(raster)
library(fractal)

# Input parameters
inspection_index = # used for determining volcano extent and background extent
name = # used for Plots
coord_x = # lon
coord_y = # lat

# Remove data entry inconsistencies (lower versus upper case)
quality_df$quality <- tolower(quality_df$quality)

# Keep the only good quality data
good_df <- quality_df[quality_df$quality %in% c('y', 'u'), ]

# Alter file names from original to processed versions
good_df$nighttime_volcano_files <- paste0('processed/',
                                          sub(good_df$nighttime_volcano_files,
                                              pattern = '.hdf',
                                              replacement = '_ImageData10_proj.tif'))

# Sort by date
dates <- substr(good_df$nighttime_volcano_files, start = 22, stop = 29)
good_df$dates <- strptime(dates, format = '%m%d%Y')
good_df <- good_df[order(good_df$dates), ]

# Read in 1 file, obtain a cropped extent and derive size of extent
t <- good_df$nighttime_volcano_files[inspection_index] 
t <- raster(t)
t <- projectRaster(t,
                   crs = CRS("+proj=longlat +datum=WGS84"))
reduced_extent <- extent(coord_x - 0.1,
                         coord_x + 0.1,
                         coord_y - 0.1,
                         coord_y + 0.1)
cropped_t <- crop(t, reduced_extent)
plot(cropped_t)
reduced_extent <- drawExtent()
backgrnd_extent <- drawExtent()
cropped_t <- crop(t, reduced_extent)
backgrnd_t <- crop(t, backgrnd_extent)
plot(cropped_t)

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

# Create a matrix to store values over time in 
main_time_series <- array(dim = c(14,
                                  round(1.3*length(cropped_t[,])),
                                  nrow(good_df)))
back_time_series <- array(dim = c(14,
                                  round(1.3*length(backgrnd_t[,])),
                                  nrow(good_df)))
count = 1
#par(mfrow = c(1, 2))
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
    
    # Getting the reduced area's pixel values
    # ... for the volcanic area
    main_r_T_cropped <- crop(r_T, reduced_extent)
    # ... for the background
    back_r_T_cropped <- crop(r_T, backgrnd_extent)
    
    # Storing pixel values into time series matrix
    # ... for the volcanic area
    for (index in 1:length(main_r_T_cropped[,])) {
      main_time_series[i, index, count] <- main_r_T_cropped[,][index]
    }
    # ... for the background
    for (index in 1:length(back_r_T_cropped[,])) {
      back_time_series[i, index, count] <- back_r_T_cropped[,][index]
    }
    
  }
  count <- count + 1
}

### Examining differences between TIR bands ###
# Sitting a color code based on image quality
good_df$color <- as.character(factor(good_df$quality,
                                     levels=c('y', 'u'),
                                     labels=c("black", "red")))
# Creating an easier matrix to work with that contains
# the maximum temperature from extracted from the raster
# when subsetted to just containing the volcano and immediate
# surroundings
max_t_series <- matrix(ncol = 5, nrow = dim(main_time_series)[3])
max_t_series[,1] <- apply(main_time_series[10,,], 2, max, na.rm = TRUE)
max_t_series[,2] <- apply(main_time_series[11,,], 2, max, na.rm = TRUE)
max_t_series[,3] <- apply(main_time_series[12,,], 2, max, na.rm = TRUE)
max_t_series[,4] <- apply(main_time_series[13,,], 2, max, na.rm = TRUE)
max_t_series[,5] <- apply(main_time_series[14,,], 2, max, na.rm = TRUE)

# Seeing correlation between bands
correlation_bands <- cor(max_t_series)
correlation_bands
min(correlation_bands)

# Determining which bnads are worstly correlated and 
# plotting their trends together along with a line
# for the difference between them
png(paste0('../../TIR_BandCompare_', name, '.png'),
    units = 'in', height = 8, width = 11.5, res = 300)
par(mfrow = c(2, 1), xpd = TRUE, mar=c(5.1,4.1,4.1,4.35))
min_row <- as.integer(which(correlation_bands == min(correlation_bands), arr.ind = TRUE)[1,][1])
min_col <- as.integer(which(correlation_bands == min(correlation_bands), arr.ind = TRUE)[1,][2])
plot(max_t_series[, min_row] ~ as.Date(good_df$dates),
     pch = 1, col = good_df$color,
     xlab = 'Date', ylab = 'Max Volcano Temperature (°C)',
     ylim = c(min(min(max_t_series[, min_row]),
                  min(max_t_series[, min_col])),
              max(max(max_t_series[, min_row]),
                  max(max_t_series[, min_col]))),
     main = paste0('Worst Correlated Bands\nBands ',
                   as.character(min_row + 9),
                   ' and ', as.character(min_col + 9),
                   ': ', as.character(round(min(correlation_bands),
                                            4))))
points(max_t_series[, min_col] ~ as.Date(good_df$dates),
       pch = 2, col = good_df$color)
for (i in 1:nrow(good_df)) {
  lines(c(as.Date(good_df$dates)[i],
          as.Date(good_df$dates)[i]),
        c(max_t_series[, min_row][i],
          max_t_series[, min_col][i]),
        col = rgb(0.5, 0.5, 0.5, 0.5))
}
legend('topright', 
       legend = c("Good", "Uncertain"), 
       col = c('black', 'red'), 
       pch = c(1,2),
       cex = 0.65, 
       text.col = "black",
       inset = c(-0.08, 0),
       bg = 'transparent')
# Same thing but for the best correlated bands
max_row <- as.integer(which(correlation_bands == sort(correlation_bands, decreasing = TRUE)[6],
                            arr.ind = TRUE)[1,][1])
max_col <- as.integer(which(correlation_bands == sort(correlation_bands, decreasing = TRUE)[6],
                            arr.ind = TRUE)[1,][2])
plot(max_t_series[, max_row] ~ as.Date(good_df$dates),
     pch = 1, col = good_df$color,
     main = paste0('Best Correlated Bands\nBands ',
                   as.character(max_row + 9),
                   ' and ', as.character(max_col + 9),
                   ': ', as.character(round(sort(correlation_bands,
                                                 decreasing = TRUE)[6],
                                            4))),
     xlab = 'Date', ylab = 'Max Volcano Temperature (°C)')
points(max_t_series[, max_col] ~ as.Date(good_df$dates),
       pch = 2, col = good_df$color)
for (i in 1:nrow(good_df)) {
  lines(c(as.Date(good_df$dates)[i],
          as.Date(good_df$dates)[i]),
        c(max_t_series[, max_row][i],
          max_t_series[, max_col][i]),
        col = rgb(0.5, 0.5, 0.5, 0.5))
}
legend('topright', 
       legend = c("Good", "Uncertain"), 
       col = c('black', 'red'), 
       pch = c(1,2),
       cex = 0.65, 
       text.col = "black",
       inset = c(-0.08, 0),
       bg = 'transparent')
dev.off()

# Getting the median temperature of a nearby area in the raster
back_t_series <- matrix(ncol = 5, nrow = dim(back_time_series)[3])
back_t_series[,1] <- apply(back_time_series[10,,], 2, median, na.rm = TRUE)
back_t_series[,2] <- apply(back_time_series[11,,], 2, median, na.rm = TRUE)
back_t_series[,3] <- apply(back_time_series[12,,], 2, median, na.rm = TRUE)
back_t_series[,4] <- apply(back_time_series[13,,], 2, median, na.rm = TRUE)
back_t_series[,5] <- apply(back_time_series[14,,], 2, median, na.rm = TRUE)

### Plotting final time series ###
png(paste0('../../TIR_TimeSeries_', name, '.png'),
    units = 'in', height = 8, width = 11.5, res = 300)
par(mfrow = c(2, 1), mar=c(5,6,4,1)+.1, xpd = TRUE)
# Raw temperature time series
plot(max_t_series[, max_row] ~ as.Date(good_df$dates),
     col = good_df$color, xlab = "",
     ylab = 'Max Volcano Temperature (°C)\n',
     main = paste('Thermal Time Series for', name,
                  '\nDetermined by ASTER Band ',
                  as.character(max_row + 9)))
lines(medianFilter(max_t_series[, max_row]) ~ as.Date(good_df$dates), xlab = 'Date',
      col = rgb(0.5, 0.5, 0.5, 0.5))
# Temperature time series of the volcano relative to the background
plot((max_t_series[, max_row] - back_t_series[, max_row]) ~ as.Date(good_df$dates),
     col = good_df$color, xlab = 'Date',
     ylab = 'Max Volcano Temperature\nabove Background (°C)')
lines(medianFilter(max_t_series[, max_row] - back_t_series[, max_row]) ~ as.Date(good_df$dates), xlab = 'Date',
      col = rgb(0.5, 0.5, 0.5, 0.5))
legend('topright', 
       legend = c("Good", "Uncertain", "Median"), 
       col = c('black', 'red', rgb(0.5, 0.5, 0.5, 0.5)), 
       pch = c(1,1,NA),
       lty = c(NA,NA,1),
       cex = 0.65, 
       text.col = "black",
       inset = c(0.5, -0.37),
       bg = 'transparent')
dev.off()

# Saving the Temp. above background data for future use
good_df$T_above_back <- (max_t_series[, max_row] - back_t_series[, max_row])
write.csv(good_df, "good_df.csv")




