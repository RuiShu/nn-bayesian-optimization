library(ggplot2)

mpi.df <- read.csv("mpi_time_data.csv", header=F)
mpi.df <- mpi.df[,1:250]

seq.df <- read.csv("sequential_time_data.csv", header=F)
seq.df <- seq.df[,1:250]

df <- data.frame(x=1:250, y1=rowMeans(t(mpi.df)), y2=rowMeans(t(seq.df)))
cairo_ps(file="plot.eps", width=7,height=7)
p <- (ggplot(df, aes(x, y = value, color = variable)) +
      geom_line(aes(y = y1, col = "Parallel")) +
      geom_line(aes(y = y2, col = "Sequential")) +
      xlab("Number of queries") + ylab("Seconds taken") +
      scale_fill_discrete(breaks=c("Sequential", "Parallel")))
plot(p)
dev.off()
