# model,         tn,    fp,    fn,    tp 
# sentRNN,      89210, 10961, 10032, 89789 
# sentRNN_JJ,   85368, 14612, 13290, 86722 
# sentRNN_SW,   95605, 4668,  32234, 67485 
# sentRNN_Topic,87158, 13013, 8558,  91263

library(reshape2)
library(ggplot2)

full <- data.frame(
  label=c("Positive", "Negative"),
  Positive=c(89789, 10932),
  Negative=c(10061, 89210))

JJ <- data.frame(
  label=c("Positive", "Negative"),
  Positive=c(86722, 13290),
  Negative=c(14612, 85368))

SW <- data.frame(
  label=c("Positive", "Negative"),
  Positive=c(67485, 32234),
  Negative=c(4668, 95605))

Dom <- data.frame(
  label=c("Positive", "Negative"),
  Positive=c(91263, 8558),
  Negative=c(13013, 87158))


conf.matrix <- function(data) {
  df <- melt(data, id="label")
  colnames(df) <- c("Predicted", "True", "value")
  df
  
  ggplot(df, aes(x=True, y=Predicted, fill=value)) +
    geom_tile() +
    geom_text(aes(label=value)) +
    theme_minimal() +
    theme(legend.position = "none",
          panel.border = element_rect(colour = "black", fill=NA, size=1)) +
    scale_fill_gradient(low="#FEF5EF", high="#75B2D6", limits=c(0, 96000)) +
    scale_x_discrete(expand = c(0,0)) +
    scale_y_discrete(expand = c(0,0))
}


conf.matrix(full)
conf.matrix(JJ)
conf.matrix(SW)
conf.matrix(Dom)
