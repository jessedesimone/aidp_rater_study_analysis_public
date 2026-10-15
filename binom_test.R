library(binom)

TP <- 
TN <- 
FP <- 
FN <- 

metrics <- data.frame(
  metric = c("Sensitivity", "Specificity", "PPV", "NPV"),
  x = c(TP, TN, TP, TN),
  n = c(TP + FN, TN + FP, TP + FP, TN + FN)
)
ci <- binom.confint(metrics$x, metrics$n, methods = "wilson")
metrics$estimate <- round(ci$mean, 2)
metrics$lower <- round(ci$lower, 2)
metrics$upper <- round(ci$upper, 2)
print(metrics[, c("metric", "estimate", "lower", "upper")], row.names = FALSE)
