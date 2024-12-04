library(tidyverse)

files <- dir("results")
analysis <- str_split(files[[1]], "_validation\\.tsv", simplify = TRUE)[[1]]
results <- read_tsv(paste0("results/", analysis, "_validation.tsv")) |>
  mutate(
    gene = factor(str_match(label, "[^\\|]*")), group = factor(group),
    .keep = "unused"
  )

set.seed(1234)
results_small <- results |>
  nest_by(gene) |>
  ungroup() |>
  slice_sample(n = 20) |>
  unnest(cols = c(data))
results_small

metrics <- results_small |>
  group_by(gene, group) |>
  summarise(
    mean = mean(similarity), stderr = sd(similarity) / sqrt(n()),
    sd = sd(similarity)
  ) |>
  ungroup()

dodge <- position_dodge(width = 0.8)
ggplot(results_small, aes(x = gene, color = group)) +
  geom_errorbar(
    aes(
      y = mean, ymin = mean - (1.95 * sd), ymax = mean + (1.95 * sd),
      fill = group,
    ),
    data = metrics, color = "black", position = dodge, width = 0.5, size = 0.3
  ) +
  geom_errorbar(
    aes(
      y = mean, ymin = mean - (1.95 * stderr), ymax = mean + (1.95 * stderr),
      fill = group,
    ),
    data = metrics, color = "black", position = dodge, width = 0.8, size = 1
  ) +
  geom_point(aes(y = similarity), position = dodge, size = 0.7) +
  geom_point(
    aes(y = mean, fill = group),
    data = metrics, position = dodge, color = "black", size = 3, pch = 22
  ) +
  labs(y = "Similarity", x = "Gene") +
  ggtitle(paste(
    "Abstract embedding similarity using",
    str_replace_all(analysis, "_", " "), "model"
  )) +
  theme(axis.text.x = element_text(angle = 20))

ggsave("figures/template_similarity_full.png")
