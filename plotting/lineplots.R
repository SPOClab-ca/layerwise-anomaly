library(tidyverse)
library(ggplot2)

# Figure 2b in paper
{
  df <- read_csv("layer-accuracy.csv")
  
  df <- df %>%
    gather("model", "acc", 2:4)
  
  ggplot(df, aes(x=layer, y=acc, color=model)) +
    geom_point(aes(shape=model), size=2) +
    geom_line(size=0.8) +
    xlab("Layer") +
    ylab("Accuracy") +
    scale_x_continuous(breaks=0:12) +
    scale_y_continuous(breaks=seq(0,1,0.2), limits=c(0, 1)) +
    theme_bw() +
    theme(legend.title=element_blank(),
          legend.position=c(0.8, 0.25),
          legend.background=element_rect(linetype="solid", color="gray"),
          panel.border = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.line = element_line(colour = "black"))
  
  ggsave("layer-accuracy.pdf", width=5, height=3)
}

# Figure 3 in paper
{
  df <- read_csv("freq-correlation.csv")
  
  df <- df %>%
    gather("model", "corr", 2:4)
  
  ggplot(df, aes(x=layer, y=corr, color=model)) +
    geom_point(aes(shape=model), size=2) +
    geom_line(size=0.8) +
    xlab("Layer") +
    ylab("Pearson Correlation") +
    scale_x_continuous(breaks=0:12) +
    scale_y_continuous(breaks=seq(0,1,0.2), limits=c(-0.1, 1)) +
    theme_bw() +
    theme(legend.title=element_blank(),
          legend.position=c(0.8, 0.8),
          legend.background=element_rect(linetype="solid", color="gray"),
          panel.border = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.line = element_line(colour = "black"))
  
  ggsave("freq-correlation.pdf", width=5, height=3)
}

# Figure 2a in paper
{
  df <- read_csv("num-sent.csv")
  
  df <- df %>%
    filter(sentences <= 10000) %>%
    gather("model", "acc", 2:4)
  
  ggplot(df, aes(x=sentences, y=acc, color=model)) +
    geom_point(aes(shape=model), size=2) +
    geom_line(size=0.8) +
    xlab("Training Sentences") +
    ylab("Accuracy") +
    scale_x_continuous(trans='log10', breaks=c(10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)) +
    ylim(c(0.55, 0.85)) +
    theme_bw() +
    theme(legend.title=element_blank(),
          legend.position=c(0.8, 0.25),
          legend.background=element_rect(linetype="solid", color="gray"),
          panel.border = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.line = element_line(colour = "black"))
  
  ggsave("num-sent.pdf", width=5, height=3)
}


