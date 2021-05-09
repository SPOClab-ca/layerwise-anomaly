# Figure 4 in paper
library(tidyverse)
library(ggplot2)

df <- read_csv("roberta_manual.csv")

df$taskname <- paste(df$type, " - ", df$task)

task_factor <- function(x) {
  factor(x, unique(df$taskname))
}
df$taskname <- task_factor(df$taskname)

ggplot(df, aes(x=layer, y=score, fill=type)) +
  facet_wrap(~taskname, ncol=1) +
  geom_bar(stat="identity") +
  geom_hline(yintercept=0, color="black") +
  scale_fill_brewer(palette="Set2") +
  scale_x_continuous(breaks=0:12) +
  scale_y_continuous(breaks=seq(-3, 3, 1)) +
  xlab("Layer") +
  ylab("Surprisal Gap") +
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        legend.position="none",
        axis.line.y.left=element_line(color='black'),
        strip.background=element_blank())

ggsave("roberta_manual.pdf", width=4, height=12)
