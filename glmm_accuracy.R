library(lme4)
library(performance)

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "agg_all.csv"
subset_to_analyze <- if (length(args) >= 2) args[2] else "all"

df <- read.csv(csv_path, stringsAsFactors = FALSE)
df_sub <- switch(subset_to_analyze,
  all = df,
  MSA_PSP = subset(df, case_type %in% c("MSAp", "PSP")),
  subset(df, case_type == subset_to_analyze)
)

df_sub$group <- factor(trimws(df_sub$group))
ref <- grep("^clinical$", levels(df_sub$group), ignore.case = TRUE, value = TRUE)
df_sub$group <- relevel(df_sub$group, ref = ref)

model_logit <- glmer(correct ~ group + (1 | rater) + (1 | case), data = df_sub, family = binomial)
model_rr <- glmer(correct ~ group + (1 | rater) + (1 | case), data = df_sub, family = poisson(link = "log"))
print(summary(model_logit))

fe_logit <- coef(summary(model_logit))
fe_rr <- coef(summary(model_rr))
coef_name <- grep("^group", rownames(fe_logit), value = TRUE)[1]

ratio_ci <- function(fe) {
  b <- fe[coef_name, "Estimate"]
  se <- fe[coef_name, "Std. Error"]
  exp(c(b, b - 1.96 * se, b + 1.96 * se))
}
or <- ratio_ci(fe_logit)
rr <- ratio_ci(fe_rr)
p_val <- fe_logit[coef_name, "Pr(>|z|)"]

r2_m <- r2_nakagawa(model_logit)$R2_marginal
f2 <- r2_m / (1 - r2_m)

prob_clin <- exp(fixef(model_rr)[1])
prob_aidp <- prob_clin * rr[1]
rrr <- ((1 - prob_clin) - (1 - prob_aidp)) / (1 - prob_clin)

cat(sprintf("OR: %.3f (95%% CI %.3f to %.3f)\n", or[1], or[2], or[3]))
cat(sprintf("RR: %.3f (95%% CI %.3f to %.3f)\n", rr[1], rr[2], rr[3]))
cat(sprintf("p: %.4f\n", p_val))
cat(sprintf("Cohen's f2: %.4f\n", f2))
cat(sprintf("Misdiagnosis risk, Clinical: %.1f%%, AIDP+: %.1f%%\n", (1 - prob_clin) * 100, (1 - prob_aidp) * 100))
cat(sprintf("Relative risk reduction: %.1f%%\n", rrr * 100))
