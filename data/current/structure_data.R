library(stringr)
current <- read.csv("./data/current/current_territories.csv", stringsAsFactors = FALSE)
n_bricks <- 22
bricks <- paste0("b", 1:n_bricks)

temp <- data.frame(bricks, temp = rep_len(NA, length.out = length(bricks)))
temp <- setNames(data.frame(t(temp[,-1])), temp[,1])
temp$center_brick <- NA
temp$SR <- NA
for(i in 1:nrow(current)){
  row <- current[i,]
  assignment <- paste0('b',str_split(row$bricks, ", ")[[1]])
  m <- match(colnames(temp), assignment, nomatch = 0)
  m[m > 0] <- 1
  m[length(m)-1] <- row$centerbrick
  m[length(m)] <- row$SR
  temp[i,] <- m
}
#find workload
wl <- read.csv("./data/current/workload.csv", stringsAsFactors = FALSE)
wl$brick <- paste0('b', wl$brick)
wl <- setNames(data.frame(t(wl[,-1])), wl[,1])
temp$sum_wl <- NA
for(i in 1:nrow(temp)){
  row <- temp[i,]
  assignment <- colnames(row)[which(row[,1:n_bricks] == 1)]
  loads <- wl[which(colnames(wl) %in% assignment)]
  temp$sum_wl[i] <- sum(loads)
}

write.csv(temp, "./data/current/formatted_territories.csv", row.names = FALSE)
