




# File use
#   1. Reads all .hdf files in directory
#     i.  obtains their dates and times
#     ii. determines percent of files that are nighttime
#   2. Only considering the nighttime files...
#     i.   derives tiff files from the hdf data
#     ii.  interactively views tiff
#     iii. requires input of suitability for analysis:
#       a.   indicate whether it (y) is good, (n) is obscured, or (u) is uncertain
#       b.   this evaluation is made with consideration of the image and quadrant cloud cover

# Requirements / parameters
# 1. You need to be in the directory (i.e. data/volcano_name) with the .hdf files of interest
#     i.  These files should all pertain to 1 volcano
coord_x = 40.662474 # lon
coord_y = 13.604800 # lat
out_dir <- 'processed/'
par(mfrow = c(1, 2))

# Libraries
library(rgdal)
library(gdalUtils)
library(raster)





# Scan the file names
volcano_files <- list.files()
volcano_files <- volcano_files[grep('.hdf', volcano_files)]

# Extract dates and times from those file names
acquisition_dates <- c()
acquisition_times <- c()
for (i in 1:length(volcano_files)) {
  acquisition <- strsplit(volcano_files[i], '_')[[1]][3]
  
  acquisition_date <- substring(acquisition, 4, 11)
  acquisition_dates <- c(acquisition_dates,
                         acquisition_date)
  
  acquisition_time <- substring(acquisition, 12, 17)
  acquisition_times <- c(acquisition_times,
                         acquisition_time)
}

# Properly format as datetimes
acquisition_dates <- strptime(acquisition_dates, format = '%m%d%Y')
acquisition_times <- strptime(acquisition_times, tz = 'GMT', format = '%H%M%S')

# Determine the frequency of nighttime imagery
acquisition_times_calc <- as.numeric(acquisition_times)
midpoint <- min(acquisition_times_calc) + 1/2*(max(acquisition_times_calc) - min(acquisition_times_calc))
if (coord_y > 0) {
  nighttime_indicators <- as.numeric(acquisition_times) > midpoint
} else {
  nighttime_indicators <- as.numeric(acquisition_times) < midpoint
}
sum(nighttime_indicators) / length(nighttime_indicators)

# Limit data to nighttime
acquisition_dates <- acquisition_dates[nighttime_indicators]
acquisition_times <- acquisition_times[nighttime_indicators]
nighttime_volcano_files <- volcano_files[nighttime_indicators]

quality_df <- data.frame(nighttime_volcano_files)
quality_df$quality <- rep('unchecked', nrow(quality_df))
count <- 1
# Draws on the following tutorial:
# https://lpdaac.usgs.gov/resources/e-learning/working-aster-l1t-visible-and-near-infrared-vnir-data-r/
# reading metadata
for (file_name in nighttime_volcano_files) {
  # Read metadata
  md <- tryCatch(gdalinfo(file_name),
                 error = function(e) e)
  if (length(md) == 0) {
    count <- count + 1
    next
  } else {
    md2 <- data.frame(md)
    names(md2) <- 'File Metadata'
    
    # Search for lower right (LR) and upper left (UL) bounding box values from the metadata
    lr <- substr(md[grep('LOWERRIGHTM', md)], 15, 50)
    ul <- substr(md[grep('UPPERLEFTM', md)], 14, 50)
    
    # Define LR and UL corners, notice here we offset the pixel center by dividing the spatial resolution (15 m) by two.
    ### Correcting to thermal band resolution (90 m)
    ul_y <- as.numeric((substr(ul, 1, (regexpr(', ' , ul) - 1)))) + 45
    ul_x <- as.numeric((substr(ul, (regexpr(', ' , ul) + 2), 10000))) - 45
    lr_y <- as.numeric((substr(lr, 1, (regexpr(', ', lr) - 1)))) - 45
    lr_x <- as.numeric((substr(lr, (regexpr(', ', lr) + 2) , 10000))) + 45
    
    # Search for and define Universal Transverse Mercator (UTM) zone from metadata
    utm_z <- substr(md[grep('UTMZONECODE', md)[1]], 1, 50)
    utm_z <- substr(utm_z, regexpr('=', utm_z) + 1, 50)
    
    # Configure the bounding box (extent) properties
    y_min <- min(ul_y, lr_y); y_max <- max(ul_y, lr_y)
    x_min <- min(ul_x, lr_x); x_max <- max(ul_x, lr_x)
    
    # Here we define the extent using the bounding box values and the extent() function from the raster package.
    raster_dims_90m <- extent(x_min, x_max, y_min, y_max)
    
    # Compile the CRS
    crs_string <- paste('+proj=utm +zone=', utm_z, ' +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0', sep = '')
    
    # List of science dataset names
    sds <- get_subdatasets(file_name)
    
    # First, we set the band designations and generate output filenames for each SDS.
    band10 <- sds[grep('TIR_Swath:ImageData10', sds)]
    band11 <- sds[grep('TIR_Swath:ImageData11', sds)]
    band12 <- sds[grep('TIR_Swath:ImageData12', sds)]
    band13 <- sds[grep('TIR_Swath:ImageData13', sds)]
    band14 <- sds[grep('TIR_Swath:ImageData14', sds)]
    
    # Clean up the names of the specific SDS
    clip2 <- c(max(unlist((gregexpr(':', band10)))),
               max(unlist((gregexpr(':', band11)))),
               max(unlist((gregexpr(':', band12)))),
               max(unlist((gregexpr(':', band13)))),
               max(unlist((gregexpr(':', band14)))))
    
    # Generate output filename
    new_file_name <- strsplit(file_name, '.hdf')
    
    # Define band names by combining the filename with the band name
    b10_name <- paste(new_file_name, substr(band10, (clip2[1] + 1), 10000), sep = '_')
    b11_name <- paste(new_file_name, substr(band11, (clip2[2] + 1), 10000), sep = '_')
    b12_name <- paste(new_file_name, substr(band12, (clip2[3] + 1), 10000), sep = '_')
    b13_name <- paste(new_file_name, substr(band13, (clip2[2] + 1), 10000), sep = '_')
    b14_name <- paste(new_file_name, substr(band14, (clip2[3] + 1), 10000), sep = '_')
    
    # Add output directory to the filenames. We will later use this to export the output files
    band10_tif_name <- paste(out_dir, b10_name, '.tif', sep='')
    band11_tif_name <- paste(out_dir, b11_name, '.tif', sep='')
    band12_tif_name <- paste(out_dir, b12_name, '.tif', sep='')
    band13_tif_name <- paste(out_dir, b13_name, '.tif', sep='')
    band14_tif_name <- paste(out_dir, b14_name, '.tif', sep='')
    
    error_catch <- tryCatch(invisible(gdal_translate(file_name, band10_tif_name, overwrite = TRUE,
                                                     sd_index = as.numeric(strsplit(md[grep('TIR_Swath:ImageData10', md)], '_')[[1]][2]))),
                            error = function(e) e)
    if ("simpleError" %in% class(error_catch)) {
      count <- count + 1
      next
    } else {
      # Extract specified SDS and export as GeoTIFF (leaving in DN)
      invisible(gdal_translate(file_name, band10_tif_name, overwrite = TRUE,
                               sd_index = as.numeric(strsplit(md[grep('TIR_Swath:ImageData10', md)], '_')[[1]][2])))
      invisible(gdal_translate(file_name, band11_tif_name, overwrite = TRUE,
                               sd_index = as.numeric(strsplit(md[grep('TIR_Swath:ImageData11', md)], '_')[[1]][2])))
      invisible(gdal_translate(file_name, band12_tif_name, overwrite = TRUE,
                               sd_index = as.numeric(strsplit(md[grep('TIR_Swath:ImageData12', md)], '_')[[1]][2])))
      invisible(gdal_translate(file_name, band13_tif_name, overwrite = TRUE,
                               sd_index = as.numeric(strsplit(md[grep('TIR_Swath:ImageData13', md)], '_')[[1]][2])))
      invisible(gdal_translate(file_name, band14_tif_name, overwrite = TRUE,
                               sd_index = as.numeric(strsplit(md[grep('TIR_Swath:ImageData14', md)], '_')[[1]][2])))
      
      # Set CRS for each band
      aster_b10 <- raster(band10_tif_name, crs = crs_string)
      aster_b11 <- raster(band11_tif_name, crs = crs_string)
      aster_b12 <- raster(band12_tif_name, crs = crs_string)
      aster_b13 <- raster(band13_tif_name, crs = crs_string)
      aster_b14 <- raster(band14_tif_name, crs = crs_string)
      
      # Define Extent
      extent(aster_b10) <- raster_dims_90m
      extent(aster_b11) <- raster_dims_90m
      extent(aster_b12) <- raster_dims_90m
      extent(aster_b13) <- raster_dims_90m
      extent(aster_b14) <- raster_dims_90m
      
      # Update the written .tif files with intended projection
      writeRaster(aster_b10,
                  sub(band10_tif_name,
                      pattern = '.tif',
                      replacement = '_proj.tif'),
                  format = 'GTiff', full.names = TRUE)
      writeRaster(aster_b11,
                  sub(band11_tif_name,
                      pattern = '.tif',
                      replacement = '_proj.tif'),
                  format = 'GTiff', full.names = TRUE)
      writeRaster(aster_b12,
                  sub(band12_tif_name,
                      pattern = '.tif',
                      replacement = '_proj.tif'),
                  format = 'GTiff', full.names = TRUE)
      writeRaster(aster_b13,
                  sub(band13_tif_name,
                      pattern = '.tif',
                      replacement = '_proj.tif'),
                  format = 'GTiff', full.names = TRUE)
      writeRaster(aster_b14,
                  sub(band14_tif_name,
                      pattern = '.tif',
                      replacement = '_proj.tif'),
                  format = 'GTiff', full.names = TRUE)
      
      # Reproject to lat/lon for volcano identification
      aster_b10_latlon <- projectRaster(aster_b10, crs = CRS("+proj=longlat +datum=WGS84"))
      
      # Plotting
      library(RColorBrewer)
      aster_b10_latlon[aster_b10_latlon == 0] <- NA
      pal <- colorRampPalette(c("white","black"))
      plot(aster_b10_latlon, col = grey(1:1000/1000), colNA="black")
      # Adding metadata cloud coverage by quadrant
      title(paste0(md[grep('QUADRANTCLOUDCOVER', md)]), '\nUpperLeft, UpperRight, LowerLeft, LowerRight')
      # Volcano
      points(coord_x, coord_y, cex = 3, col = 'red')
      
      # Zoomed-in plot
      plot(crop(aster_b10_latlon,
                extent(coord_x - 0.1,
                       coord_x + 0.1,
                       coord_y - 0.1,
                       coord_y + 0.1)),
           col = grey(1:1000/1000), colNA="black")
      # Volcano
      points(coord_x, coord_y, cex = 3, col = 'red')
      
      # Record quality evaluation
      quality_df$nighttime_volcano_files[count] <- file_name # double insuring intended file is evaluated
      quality_df$quality[count] <- readline()
      count <- count + 1
    }
  }
}

# Save the quality analysis dataframe
write.csv(quality_df, 'quality_df.csv')

# View distribution of quality indicators
summary(factor(tolower(quality_df$quality)))

# Plot those quality indicators through time
par(mfrow = c(1, 1))
plot(as.Date(acquisition_dates),
     factor(tolower(quality_df$quality)),
     col = factor(tolower(quality_df$quality)), yaxt='n', cex.axis = 0.8,
     xlab = 'Date', ylab = '', ylabels = c())
title('ASTER image quality for Nyiragongo')
axis(side = 2, at = c(1, 2, 3),
     labels = c('Bad\nquality', 'Uncertain\nquality', 'Good\nquality'),
     las = 1, cex.axis = 0.8)




