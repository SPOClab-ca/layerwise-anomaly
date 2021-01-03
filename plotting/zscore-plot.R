library(tidyverse)
library(ggplot2)

df <- read_csv("roberta_manual.csv")

task_factor <- function(x) {
  factor(x, unique(df$task))
}
df$task <- task_factor(df$task)

ggplot(df, aes(x=layer, y=score)) +
  facet_wrap(~task, ncol=1) +
  geom_bar(stat="identity", fill="steelblue") +
  geom_hline(yintercept=0, color="black") +
  scale_x_continuous(breaks=0:12) +
  scale_y_continuous(breaks=seq(-3, 3, 1)) +
  xlab("Layer") +
  ylab("Z-Score") +
  theme_bw() +
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank(),
        axis.line.y.left=element_line(color='black'),
        strip.background=element_blank())

ggsave("roberta_manual.pdf", width=4, height=12)
