if (!require("devtools"))
  install.packages("devtools")

devtools::install_github("trnnick/TStools", force = TRUE)

require(TStools)

dados = read.csv("resultados.csv", sep=";")

dados

matriz = as.matrix(dados)
matriz

tsutils::nemenyi(matriz, conf.int=0.95, plottype = "vline")
