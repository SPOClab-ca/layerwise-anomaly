library(tidyverse)
library(ggplot2)

df <- read_csv('blimp_result.csv')

df_mean <- df %>% group_by(layer) %>% summarize(score=mean(score))

ggplot(df_mean, aes(x=layer, y=score)) +
  geom_point() +
  geom_line() +
  theme_bw()


for(cur_task_name in unique(df$task_name)) {
  df_task <- df %>% filter(task_name == cur_task_name)
  ggplot(df_task, aes(x=layer, y=score)) +
    geom_point() +
    geom_line() +
    ggtitle(cur_task_name) +
    scale_y_continuous(limits=c(0, 1)) +
    scale_x_discrete(limits=seq(0, 12)) +
    theme_bw()
  ggsave(paste('outfig/', cur_task_name, '.png'))
}
