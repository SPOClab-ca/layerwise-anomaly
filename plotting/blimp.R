library(tidyverse)
library(ggplot2)

{
  # Data loading
  df <- read_csv('blimp_result.csv')
  
  # Compare with published results, join with their csv
  warstadt_df <- read_csv("warstadt_blimp_results.csv") %>%
    filter(type == "sentence")
  
  df <- df %>% merge(warstadt_df, by.x="task_name", by.y="UID")
}

# Plot GMM accuracy by layer for each of 67 paradigms
for(cur_task_name in unique(df$task_name)) {
  df_task <- df %>% filter(task_name == cur_task_name)
  top_task_name <- df_task %>% head(1) %>% select(linguistics_term)
  ggplot(df_task, aes(x=layer, y=score)) +
    geom_point() +
    geom_line() +
    ggtitle(paste(top_task_name, '/', cur_task_name)) +
    scale_y_continuous(limits=c(0, 1)) +
    scale_x_discrete(limits=seq(0, 12)) +
    theme_bw()
  dir.create(paste0('outfig/', top_task_name), showWarnings=F)
  ggsave(paste0('outfig/', top_task_name, '/', cur_task_name, '.png'))
}

# Try to derive first column of table 3, but doesn't match...
warstadt_df %>%
  group_by(linguistics_term) %>%
  summarize(GPT2=mean(GPT2), n=sum(n)) %>%
  summarize(overall_score=weighted.mean(GPT2, n))

# Calculate per-phenomenon score of our model
df %>%
  filter(layer==11) %>%
  group_by(linguistics_term) %>%
  summarize(score=100*mean(score))

