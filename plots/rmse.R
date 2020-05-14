theme_Publication <- function(base_size=14, base_family="helvetica") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(1.5), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = "black", size=1),
            axis.title = element_text(face = "bold",size = rel(1.3)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(), 
            # axis.line = element_line(colour="black"),
            axis.line = element_blank(),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour = "gray", linetype = "dashed"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "bottom",
            legend.direction = "horizontal",
            legend.key.size= unit(1.5, "cm"),
            legend.spacing = unit(0, "cm"),
            legend.title = element_text(face="italic"),
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}


library(dplyr)
library(ggplot2)


train <- read.csv('train_RMSE.csv', header=T)
val <- read.csv("val_RMSE.csv", header=T)

train <- train %>% select(-Wall.time)
val <- val %>% select(-Wall.time)

train$Step <- train$Step + 1
val$Step <- val$Step + 1

train$group = "train"
val$group = "val"

df <- rbind(train, val)


p1 <- ggplot(df, aes(x=Step, Value, group=group, linetype=group)) +
  geom_line(size=1) +
  theme_Publication() +
  theme(legend.title = element_blank()) +
  ggtitle("Validation Root Mean Square Error") + 
  labs(x = "Epochs", y = "RMSE") +
  scale_linetype_manual(values = c(1, 8), labels = c("Train", "Validation"))


p1


# 3, 8, 1, 4

# require(gridExtra)

# grid.arrange(p1, p2, ncol=2)