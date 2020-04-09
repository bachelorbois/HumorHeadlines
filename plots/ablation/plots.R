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
            legend.position = "right",
            legend.direction = "vertical",
            legend.box.spacing = unit(0, "cm"),
            legend.spacing = unit(0, "cm"),
            legend.title = element_text(face="italic", margin=margin(l=23)),
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}


library(dplyr)
library(ggplot2)

entire <- read.csv('all_points.csv', header=T)
# sums <- read.csv("summary.csv", header=T)

entire$Group = as.factor(entire$Group)
# sums$Group = as.factor(sums$Group)

entire$Set = factor(entire$Set, levels=c("Train", "Dev"), labels=c("Training Set", "Development Set"), ordered=T)

  codes = c("F", "EW", "OW", "WE", "KB", "WD", "WP", "SL", "PD", "FE", "KF")
labels = c(
  "F - Full model",
  "EW - Edited Word",
  "OW - Original Word",
  "WE - Word Encoder",
  "KB - Knowledge Base",
  "WD - Word Distance",
  "WP - Word Position",
  "SL - Sentence Length",
  "PD - Phonetic Difference",
  "FE - Feature Encoder",
  "KF - Knowledge and Feature" 
)


# Boxplots

filter(entire, Kind=="RMSE") %>%
ggplot(aes(x=Group, y=Value, color=Group)) +
  geom_boxplot() +
  scale_x_discrete(
    labels=codes
  ) +
  scale_y_continuous() +
  scale_color_manual(values=rep("black",11), labels=labels) +
  theme_Publication() +
  facet_grid(cols=vars(Set), scales="free") +
  guides(color=guide_legend(override.aes = c(size=0))) +
  xlab("Ablated Part") +
  ylab("RMSE") +
  labs(color="Ablated Part")





