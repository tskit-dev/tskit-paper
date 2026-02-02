library("ggplot2")
set.seed(123)

n <- 100
df <- data.frame(
        x = 5 + rnorm(n),
        y = 5 + rnorm(n)/2,
        z = rexp(n)
)
df$y <- df$y + df$x

pdf(file="scatterplot.pdf", width=6, height=6, pointsize=10)
df |> ggplot(aes(x=x, y=y, size=z)) +
    geom_point(color="#1eebb1") +
    labs(x='', y='') +
    theme_light() +
    theme(legend.position='none')
dev.off()

