library(tidyverse)
library(ggplot2)
library(ggrepel)

# Google Sheets: "Sentence pair patterns"
df <- read_csv('sentence_pairs.csv')

model_name = 'bert-base-uncased'
df_filtered <- df %>% filter(Model == model_name)
ggplot(df_filtered, aes(x=Max12, y=`Max12/max4`, label=Type)) +
  theme_bw() +
  geom_point() +
  geom_text_repel() +
  xlab('Max12 (Semantic distance)') +
  ylab('Max12/max4 (Distance increase in upper layer)') +
  ggtitle(paste('Using:', model_name))
