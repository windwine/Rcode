# Load required packages
library(quantmod)
library(tidyquant)
library(tidyverse)
library(data.table)
library(fs)
library(foreach)
library(doParallel)


# Set Tiingo API Key (replace with your own key)
# You can get one from https://www.tiingo.com/
csv_folder="E:/Nutstore/CTAtest2/202509051500"
setwd(csv_folder)


files=list.files()
files=files[str_detect(files,"SH000001")]
nfiles=length(files)
# read data and convert to xts --------------------------------------------

normalize_ohlc <- function(data_dt) {
  # Copy to avoid modifying original
  dt <- copy(data_dt)
  
  # Normalize prices to start at 1
  norm_factor <- dt$close[1]
  dt[, `:=`(
    open  = open / norm_factor,
    high  = high / norm_factor,
    low   = low / norm_factor,
    close = close / norm_factor
  )]
  
  # Normalize amount (optional: for shape preservation only)
  max_amount <- max(dt$amount, na.rm = TRUE)
  dt[, amount := amount / max_amount]
  
  # Replace actual dates with synthetic ones (e.g., 1 to n)
  dt[, date := as.Date(seq_len(.N), origin = "2000-01-01")]
  
  return(dt)
}


for (nn in 1:nfiles)
{
  setwd(csv_folder)
  data1<-fread(files[nn])
  print(files[nn])
  # Convert to xts object for compatibility with quantmod
  # Ensure the date column is of Date class
  data1[, date := as.Date(date)]
  
  ID=unique(data1$StockID)[1]
  
  IDfolder=paste0("E:/Nutstore/GPT/",ID,"_pure2long")
  dir_create(IDfolder)
  setwd(IDfolder)
  
  days=sort(unique(data1$date))
  dds=days[days>"2003-01-01"]
  ndays=length(dds)
  # Setup parallel backend
  n_cores <- parallel::detectCores(logical = FALSE)
  cl <- makeCluster(8)
  registerDoParallel(cl)
  
  # Parallel chart generation
  foreach(i = 1:ndays, .packages = c("dplyr", "xts", "quantmod", "data.table")) %dopar% {
    today <- dds[i]
    
    data <- data1 %>%
      filter(date <= today)
    
    norm_data <- normalize_ohlc(data)
    
    data_xts <- xts::xts(
      x = norm_data[, .(open, high, low, close, amount)],
      order.by = norm_data$date
    )
    
    colnames(data_xts) <- c("data.Open", "data.High", "data.Low", "data.Close", "data.Volume")
    
    date_str <- format(today, "%Y%m%d")
    
    # --- Daily Chart ---
    daily_file <- paste0(date_str, "_daily.png")
    png(daily_file, width = 2400, height = 1600, res = 300)
    chartSeries(
      data_xts,
      subset    = "last 6 months",
      theme     = chartTheme("white"),
      log.scale = TRUE,
      name      = "Daily OHLC (last 6 months)",
      TA = paste(
        "addVo()",
        "addMACD()",
        "addTA(SMA(Cl(data_xts), n = 5), col = 'red', on = 1)",
        "addTA(SMA(Cl(data_xts), n = 20), col = 'blue', on = 1)",
        sep = "; "
      )
    )
    dev.off()
    
    # --- Weekly Chart ---
    weekly_file <- paste0(date_str, "_weekly.png")
    png(weekly_file, width = 2400, height = 1600, res = 300)
    
    data_weekly <- to.weekly(data_xts, name = "data")
    data_weekly <- tail(data_weekly, 100)
    
    chartSeries(
      data_weekly,
      name       = "Weekly OHLC (Last 100 Weeks)",
      theme      = chartTheme("white"),
      TA         = "addVo(); addMACD(); addTA(SMA(Cl(data_weekly), n = 50), on = 1, col = 'blue')",
      log.scale  = TRUE
    )
    dev.off()
  }
  
  # Cleanup
  stopCluster(cl)
  
  
}

