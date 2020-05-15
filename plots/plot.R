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


dist <- read.csv('D:/ITU/HumorHeadlines/plots/CNN.csv', header=T)
dist$group = "CNN W. Highway Network"

len <- read.csv("D:/ITU/HumorHeadlines/plots/KBLSTM.csv", header=T)
len$group = "KBLSTM"

pos <- read.csv("D:/ITU/HumorHeadlines/plots/MultiCNN.csv", header=T)
pos$group = "MultiCNN"

phon <- read.csv("D:/ITU/HumorHeadlines/plots/NNLM.csv", header=T)
phon$group = "NNLM"

simple <- read.csv("D:/ITU/HumorHeadlines/plots/simple.csv", header=T)
simple$group = "Base model"

context <- read.csv("D:/ITU/HumorHeadlines/plots/context.csv", header=T)
context$group = "Extended model"

dev <- read.csv("D:/ITU/HumorHeadlines/plots/dev.csv", header=T)
dev$group = "True Values"

dev <- dev[,c("id", "pred", "group")]

lin <- read.csv("D:/ITU/HumorHeadlines/plots/LINEAR.csv", header=T)
lin$group = "Linear Regression"


data = rbind(dist, len, pos, phon, dev, lin, simple, context)

ggplot(data, aes(x=pred)) +
  theme_Publication() +
  theme(legend.position = "none") +
  geom_histogram() + 
  facet_wrap(vars(group), ncol=2) +
  xlab("Predicted score") +
  ylab("Number of points")


# ggplot(df, aes(x=Step, Value, group=group, linetype=group)) +
#   geom_line(size=1) +
#   theme_Publication() +
#   theme(legend.title = element_blank()) +
#   ggtitle("Validation Root Mean Square Error") + 
#   labs(x = "Epochs", y = "RMSE") +
#   scale_linetype_manual(values = c(1, 8), labels = c("Train", "Validation"))
