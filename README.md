# Global-Trade-Conflict-Classification
Machine learning analysis of conflict onset based on trade network structure and sanction exposure
# ============================================================
# STEP 1. 전처리 (Preprocessing)
# ============================================================
library(tidyverse)
library(caret)
library(corrplot)
library(car)        # VIF
library(smotefamily) # SMOTE
library(ggplot2)

# 데이터 불러오기
trade <- read.csv("C:/Users/dy041/Downloads/4-1/다변량/global_trade_conflict_risk_dataset.csv")
str(trade)
summary(trade)

# 범주형 변수 변환
trade$country                  <- as.factor(trade$country)
trade$quarter                  <- as.factor(trade$quarter)
trade$sanction_event           <- as.factor(trade$sanction_event)
trade$conflict_onset           <- as.factor(trade$conflict_onset)
trade$policy_intervention_flag <- as.factor(trade$policy_intervention_flag)


# 이진 변수 int -> num으로 타입 변환
trade$policy_intervention_flag_num <- as.numeric(as.character(trade$policy_intervention_flag))

# 분석에 사용할 변수 고르기
# feature_vars_full 한 번만 정의 (trade_volume_usd, trade_balance 제외)
feature_vars_full <- c(
  # 무역 구조
  "export_value",
  "import_value",
  "trade_network_degree",
  "network_centrality_score",
  "clustering_coefficient",

  # 공급망/글로벌 연결  
  "supply_chain_dependency",
  "digital_trade_index",
  
  # 경제 상태
  "economic_growth_rate",
  "inflation_rate",
  "unemployment_rate",
  "foreign_investment_inflow",
  "commodity_price_index",
  
  # 정치/리스크
  "sanction_exposure_index"
)

df <- trade[, c("conflict_risk_score", feature_vars_full)]
head(df)

# 결측치 확인
sum(is.na(df))
# -> 0개, 결측치 처리 필요 X

# 변수들 분포 확인
# 종속변수의 분포 확인
par(mfrow=c(1,1))
hist(df$conflict_risk_score, main='conflict risk score')
# 정규분포 형태
# 회귀 모델 사용에 적합하고, 변환은 필요 없음

# 설명변수의 분포 확인
# 논문에는 이런식으로 히스토그램+박스플랏 쌍으로 넣으면 좋을 것 같아서 추가해봣어용
par(mfrow=c(1,2))
for(i in 1:ncol(df)){
  hist(df[,i], main=colnames(df)[i])
  boxplot(df[,i], main=colnames(df)[i])
}

# 히스토그램
par(mfrow=c(4,4))
for(i in 1:ncol(df)){
  hist(df[,i], main=colnames(df)[i])
}

# 박스플랏
for(i in 1:ncol(df)){
  boxplot(df[,i], main=colnames(df)[i])
}
# 모든 변수가 균등분포이거나 정규분포이기 때문에 로그 변환은 필요 없음
# policy_intervention_flag_num은
# 대부분의 값이 0인 불균형 변수
# 거의 대부분의 값이 0이고 일부만 1인 불균형변수
# => 방법1. tree 모델에서는 상관 없기 때문에 그냥 두기
# => 방법2. 정보량이 거의 없고 노이즈의 가능성이 있기 때문에 제거하기
# 둘중에서 골라야될듯

# 이상치 확인(boxplot)
#이상치확인
# 이상치 확인 (박스플롯)
par(mfrow=c(1,4))
for(i in 1:ncol(df)){
  boxplot(df[,i], main=colnames(df)[i])
}


# 상관관계 분석
par(mfrow=c(1,1))

cor_mat <- cor(df)
corrplot(cor_mat, method="color")
cor_mat

# VIF 계산
model_temp <- lm(conflict_risk_score ~ ., data=df)
vif_result = vif(model_temp)

cat("=== VIF 결과 ===\n")
print(round(vif_result, 2))

cat("\n>> VIF > 5 변수:\n")
print(vif_result[vif_result > 5])

# 일단은 변수 제거 x

# 연관성 검정
# 반응변수 이진화
# 기준: 0.5
df$risk_group <- ifelse(df$conflict_risk_score > 0.5, "High", "Low")
df$risk_group <- as.factor(df$risk_group)

# 1) 연속형 vs 반응변수 t-test
cont_vars <- c(
  "export_value",
  "import_value",
  "supply_chain_dependency",
  "trade_network_degree",
  "network_centrality_score",
  "clustering_coefficient",
  "sanction_exposure_index",
  "digital_trade_index",
  "economic_growth_rate",
  "inflation_rate",
  "unemployment_rate",
  "foreign_investment_inflow",
  "commodity_price_index"
)

# 반복 실행
t_results <- lapply(cont_vars, function(v) {
  res <- t.test(df[[v]] ~ df$risk_group)
  data.frame(
    variable = v,
    p_value = res$p.value,
    mean_high = mean(df[[v]][df$risk_group=="High"]),
    mean_low  = mean(df[[v]][df$risk_group=="Low"])
  )
})

t_results <- do.call(rbind, t_results)
t_results[order(t_results$p_value), ]
# p-value < 0.05 => 두 그룹에 평균 차이가 있음


# 데이터 분할 (7:3)
#오류 나서 수정한 버전
# 데이터 분할 (7:3)
set.seed(123)

train_idx <- createDataPartition(df$conflict_risk_score,
                                 p = 0.7,
                                 list = FALSE)
train <- df[train_idx, ]
test  <- df[-train_idx, ]

# 표준화
# train 스케일링
train_x <- train[, -1]  # y 제외
test_x  <- test[, -1]

train_scaled <- scale(
  train_x[, sapply(train_x, is.numeric)]
) #범주형 제거

# 기준 저장
mean_train <- attr(train_scaled, "scaled:center")
sd_train   <- attr(train_scaled, "scaled:scale")

# test도 train과 같은 기준으로 스케일링
test_scaled <- scale(
  test_x[, sapply(test_x, is.numeric)],
  center = mean_train,
  scale  = sd_train
) #범주형 제거

# y 합치기
train_scaled <- as.data.frame(train_scaled)
test_scaled  <- as.data.frame(test_scaled)

train_scaled$conflict_risk_score <- train$conflict_risk_score
test_scaled$conflict_risk_score  <- test$conflict_risk_score
# STEP2 PCA
# pca 학습
pca_model <- prcomp(train_scaled[, -ncol(train_scaled)],
                    center = FALSE,
                    scale. = FALSE)

# 결과(분산설명력) 확인
pca_summary = summary(pca_model)
pca_summary
# PCA 분산 설명력 (상위 10개 PC)
pca_summary$importance[, 1:10]

# 누적 분산 기반 PC 개수 자동 선택 (90%)
var_explained <- pca_model$sdev^2 / sum(pca_model$sdev^2)
cum_var <- cumsum(var_explained)

n_pc_90 <- which(cum_var >= 0.90)[1]
cat("\n>> 누적 분산 90% 달성 PC 개수:", n_pc_90, "\n")

# scree plot
scree_df <- data.frame(
  PC  = 1:length(var_explained),
  Var = var_explained * 100
)

ggplot(scree_df[1:15, ], aes(x = PC, y = Var)) +
  geom_col(fill = "steelblue") +
  geom_line(aes(group = 1), color = "red", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(
    title = "PCA Scree Plot",
    x = "Principal Component",
    y = "Variance Explained (%)"
  ) +
  theme_minimal()

# 누적 분산 그래프
cum_df <- data.frame(
  PC = 1:length(cum_var),
  CumVar = cum_var * 100
)

ggplot(cum_df[1:15, ], aes(x = PC, y = CumVar)) +
  geom_line(size = 1, color = "darkgreen") +
  geom_point(size = 2, color = "darkgreen") +
  geom_hline(yintercept = 90, linetype = "dashed", color = "red") +
  labs(
    title = "Cumulative Variance Explained",
    x = "Number of PCs",
    y = "Cumulative Variance (%)"
  ) +
  theme_minimal()

# 로딩값
cat("\n=== PC1 ~ PC5 로딩값 ===\n")
print(round(pca_model$rotation[, 1:5], 3))

# PCA 적용 (train / test)
train_pca <- predict(pca_model,
                     newdata = train_scaled[, -ncol(train_scaled)])
test_pca <- predict(pca_model,
                    newdata = test_scaled[, -ncol(test_scaled)])

train_pca <- as.data.frame(train_pca)
test_pca  <- as.data.frame(test_pca)

train_pca$conflict_risk_score <- train$conflict_risk_score
test_pca$conflict_risk_score  <- test$conflict_risk_score

# PCA Scatter Plot (ggplot)
train_pca$group <- ifelse(train$conflict_risk_score > 0.5, "High", "Low")

ggplot(train_pca, aes(x = PC1, y = PC2, color = group)) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("Low" = "blue", "High" = "red")) +
  labs(
    title = "PCA Scatter Plot",
    subtitle = "Risk Level based on conflict_risk_score",
    x = "PC1", y = "PC2"
  ) +
  theme_minimal()

# PC1 vs Risk 비교 (PC1만 사용)
ggplot(train_pca, aes(x = group, y = PC1, fill = group)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Low" = "blue", "High" = "red")) +
  labs(
    title = "PC1 vs Risk Level",
    x = "Risk Group",
    y = "PC1"
  ) +
  theme_minimal()

# PC1만 사용한 모델 적합시켜봄
train_pc1 <- data.frame(
  PC1 = train_pca$PC1,
  y   = train$conflict_risk_score
)

test_pc1 <- data.frame(
  PC1 = test_pca$PC1,
  y   = test$conflict_risk_score
)

feature_vars_final <- feature_vars_full

# 선형회귀
model_pc1 <- lm(y ~ PC1, data = train_pc1)
summary(model_pc1)

# 예측
pred_pc1 <- predict(model_pc1, newdata = test_pc1)

# 성능
library(Metrics)
rmse(test_pc1$y, pred_pc1)
mae(test_pc1$y, pred_pc1)

# 기존 모델과 성능 비교
model_full <- lm(conflict_risk_score ~ ., data = train)

pred_full <- predict(model_full, newdata = test)

rmse_full <- rmse(test$conflict_risk_score, pred_full)
rmse_pc1  <- rmse(test_pc1$y, pred_pc1)

rmse_full
rmse_pc1


############################################################
# 패키지 로드
############################################################

library(caret)
library(randomForest)
library(xgboost)
library(pROC)



# ============================================================
# 분류 전처리: conflict_onset 연관성 검정
# ============================================================

# conflict_onset 수치형으로 임시 변환 (검정용)
trade$conflict_onset_num <- as.numeric(as.character(trade$conflict_onset))

cat("=== conflict_onset 분포 ===\n")
print(table(trade$conflict_onset))
# 0: 2295 (91.8%), 1: 205 (8.2%) → 불균형

# ── 연속형 변수 vs conflict_onset: t-test ──────────────
t_cls <- lapply(feature_vars_final, function(v) {
  res <- t.test(trade[[v]] ~ trade$conflict_onset)
  data.frame(
    variable  = v,
    p_value   = res$p.value,
    mean_0    = res$estimate[1],  # 분쟁 미발생 평균
    mean_1    = res$estimate[2]   # 분쟁 발생 평균
  )
})

t_cls_df <- do.call(rbind, t_cls)
cat("\n=== 연속형 변수 vs conflict_onset (t-test) ===\n")
print(t_cls_df[order(t_cls_df$p_value), ])

# ── 시각화: 유의한 변수만 박스플롯 ──────────────────────
sig_vars <- t_cls_df$variable[t_cls_df$p_value < 0.05]
cat("\n유의한 변수 (p < 0.05):", paste(sig_vars, collapse=", "), "\n")

par(mfrow = c(1, length(sig_vars)))
for (v in sig_vars) {
  boxplot(trade[[v]] ~ trade$conflict_onset,
          main = v,
          xlab = "conflict_onset",
          col  = c("steelblue", "coral"))
}
par(mfrow = c(1,1))

# ── 범주형 변수 vs conflict_onset: 카이제곱 검정 ─────────
# 우리 데이터에서 범주형 독립변수는 없지만
# 원본 trade 데이터의 범주형 변수들과 연관성 확인

cat_vars_cls <- c("sanction_event", "policy_intervention_flag")

chisq_cls <- lapply(cat_vars_cls, function(v) {
  tab <- table(trade[[v]], trade$conflict_onset)
  res <- chisq.test(tab)
  data.frame(
    variable = v,
    chi_sq   = round(res$statistic, 3),
    df       = res$parameter,
    p_value  = res$p.value
  )
})

chisq_cls_df <- do.call(rbind, chisq_cls)
cat("\n=== 범주형 변수 vs conflict_onset (카이제곱 검정) ===\n")
print(chisq_cls_df)

# 시각화: 범주별 conflict_onset 비율
for (v in cat_vars_cls) {
  tab_df <- as.data.frame(prop.table(
    table(trade[[v]], trade$conflict_onset), margin = 1
  ))
  colnames(tab_df) <- c("Category", "conflict_onset", "Proportion")
  
  print(
    ggplot(tab_df, aes(x = Category, y = Proportion, fill = conflict_onset)) +
      geom_col(position = "dodge") +
      scale_fill_manual(values = c("0" = "steelblue", "1" = "coral")) +
      labs(title = paste(v, "vs conflict_onset"),
           x = v, y = "비율", fill = "분쟁 발생") +
      theme_minimal()
  )
}

# ============================================================
# 분류 모델링
# ============================================================
library(caret)
library(smotefamily)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(e1071)

# ── 데이터 준비 ───────────────────────────────────────────
df_cls <- trade[, c("conflict_onset", feature_vars_final)]
df_cls$conflict_onset <- as.factor(df_cls$conflict_onset)

# ── Train/Test 분리 ───────────────────────────────────────
set.seed(123)
cls_idx   <- createDataPartition(df_cls$conflict_onset, p = 0.7, list = FALSE)
train_cls_raw <- df_cls[cls_idx, ]
test_cls_raw  <- df_cls[-cls_idx, ]

cat("\n원본 Train 클래스 분포:\n")
print(table(train_cls_raw$conflict_onset))

# ── 표준화 (train 기준) ───────────────────────────────────
train_cls_x <- train_cls_raw[, feature_vars_final]
test_cls_x  <- test_cls_raw[, feature_vars_final]

train_cls_scaled_x <- scale(train_cls_x)
mean_cls <- attr(train_cls_scaled_x, "scaled:center")
sd_cls   <- attr(train_cls_scaled_x, "scaled:scale")

test_cls_scaled_x <- scale(test_cls_x, center = mean_cls, scale = sd_cls)

# ── SMOTE (train에만) ─────────────────────────────────────
smote_input <- as.data.frame(train_cls_scaled_x)
smote_input$conflict_onset <- as.numeric(as.character(train_cls_raw$conflict_onset))

set.seed(42)
smote_result <- SMOTE(
  X        = smote_input[, feature_vars_final],
  target   = smote_input$conflict_onset,
  K        = 5,
  dup_size = 10
)

smote_df <- smote_result$data
colnames(smote_df)[ncol(smote_df)] <- "conflict_onset"
smote_df$conflict_onset <- as.factor(smote_df$conflict_onset)

cat("\nSMOTE 후 클래스 분포:\n")
print(table(smote_df$conflict_onset))

# test 표준화 데이터
test_cls_scaled <- as.data.frame(test_cls_scaled_x)
test_cls_scaled$conflict_onset <- test_cls_raw$conflict_onset

# ── 평가 함수 ─────────────────────────────────────────────
eval_cls <- function(pred, actual, prob = NULL, model_name = "") {
  cm  <- confusionMatrix(pred, actual, positive = "1")
  acc <- cm$overall["Accuracy"]
  pre <- cm$byClass["Precision"]
  rec <- cm$byClass["Recall"]
  f1  <- cm$byClass["F1"]
  spe <- cm$byClass["Specificity"]
  
  auc_val <- NA
  if (!is.null(prob)) {
    roc_obj <- roc(as.numeric(as.character(actual)), prob, quiet = TRUE)
    auc_val <- auc(roc_obj)
  }
  
  cat("\n==============================\n")
  cat("모델:", model_name, "\n")
  cat("Accuracy   :", round(acc, 4), "\n")
  cat("Precision  :", round(pre, 4), "\n")
  cat("Recall     :", round(rec, 4), "\n")
  cat("F1-Score   :", round(f1,  4), "\n")
  cat("Specificity:", round(spe, 4), "\n")
  if (!is.na(auc_val)) cat("AUC-ROC    :", round(auc_val, 4), "\n")
  cat("혼동행렬:\n")
  print(cm$table)
  cat("==============================\n")
  
  data.frame(Model       = model_name,
             Accuracy    = round(acc, 4),
             Precision   = round(pre, 4),
             Recall      = round(rec, 4),
             F1          = round(f1,  4),
             Specificity = round(spe, 4),
             AUC         = round(auc_val, 4))
}

cls_results <- list()
actual_test <- test_cls_raw$conflict_onset
actual_bin  <- as.numeric(as.character(actual_test))

# ── 1. Logistic Regression ────────────────────────────────
set.seed(42)
logit_model <- glm(conflict_onset ~ ., data = smote_df, family = binomial)
logit_prob  <- predict(logit_model, test_cls_scaled, type = "response")
logit_pred  <- factor(ifelse(logit_prob > 0.5, "1", "0"), levels = c("0","1"))
cls_results[["Logistic"]] <- eval_cls(logit_pred, actual_test, logit_prob, "Logistic Regression")

# ── 2. Decision Tree ──────────────────────────────────────
set.seed(42)
tree_cls  <- rpart(conflict_onset ~ ., data = smote_df, method = "class",
                   control = rpart.control(cp = 0.001, maxdepth = 8))
tree_pred <- predict(tree_cls, test_cls_scaled, type = "class")
tree_prob <- predict(tree_cls, test_cls_scaled, type = "prob")[, "1"]
cls_results[["DTree"]] <- eval_cls(tree_pred, actual_test, tree_prob, "Decision Tree")

# ── 3. Random Forest ──────────────────────────────────────
set.seed(42)
rf_cls  <- randomForest(conflict_onset ~ ., data = smote_df,
                        ntree = 300, mtry = 4, maxnodes = 50,
                        importance = TRUE)
rf_pred <- predict(rf_cls, test_cls_scaled)
rf_prob <- predict(rf_cls, test_cls_scaled, type = "prob")[, "1"]
cls_results[["RF"]] <- eval_cls(rf_pred, actual_test, rf_prob, "Random Forest")

# RF 변수 중요도
varImpPlot(rf_cls, main = "Random Forest Feature Importance (분류)")

# ── 4. XGBoost ────────────────────────────────────────────
xgb_tr_x <- as.matrix(smote_df[, feature_vars_final])
xgb_tr_y <- as.integer(as.character(smote_df$conflict_onset))
xgb_te_x <- as.matrix(test_cls_scaled[, feature_vars_final])

set.seed(42)
xgb_cls  <- xgboost(x = xgb_tr_x, y = factor(xgb_tr_y),
                    nrounds = 100, max_depth = 4,
                    learning_rate = 0.05, subsample = 0.8,
                    colsample_bytree = 0.8,
                    objective = "binary:logistic",
                    eval_metric = "logloss")
xgb_prob <- predict(xgb_cls, xgb_te_x)
xgb_pred <- factor(ifelse(xgb_prob > 0.5, "1", "0"), levels = c("0","1"))
cls_results[["XGB"]] <- eval_cls(xgb_pred, actual_test, xgb_prob, "XGBoost")

# ── 5. SVM ────────────────────────────────────────────────
set.seed(42)

svm_cls <- svm(
  conflict_onset ~ .,
  data = smote_df,
  kernel = "radial",
  probability = TRUE
)

svm_pred <- predict(
  svm_cls,
  test_cls_scaled,
  probability = TRUE
)

svm_prob <- attr(
  svm_pred,
  "probabilities"
)[, "1"]

cls_results[["SVM"]] <- eval_cls(
  svm_pred,
  actual_test,
  svm_prob,
  "SVM"
)

# ── 성능 비교표 ───────────────────────────────────────────
cls_df <- do.call(rbind, cls_results)
rownames(cls_df) <- NULL
cat("\n=== 분류 모델 성능 비교 ===\n")
print(cls_df)

# ── ROC Curve 비교 ────────────────────────────────────────
roc_to_df <- function(roc_obj, model_name) {
  data.frame(
    FPR   = 1 - roc_obj$specificities,
    TPR   = roc_obj$sensitivities,
    Model = paste0(model_name, " (AUC=", round(auc(roc_obj), 3), ")")
  )
}

roc_logit <- roc(actual_bin, logit_prob, quiet = TRUE)
roc_tree  <- roc(actual_bin, tree_prob,  quiet = TRUE)
roc_rf    <- roc(actual_bin, rf_prob,    quiet = TRUE)
roc_xgb   <- roc(actual_bin, xgb_prob,   quiet = TRUE)
roc_svm   <- roc(actual_bin, svm_prob,   quiet = TRUE)

roc_df <- bind_rows(
  roc_to_df(roc_logit, "Logistic"),
  roc_to_df(roc_tree,  "Decision Tree"),
  roc_to_df(roc_rf,    "Random Forest"),
  roc_to_df(roc_xgb,   "XGBoost"),
  roc_to_df(roc_svm,   "SVM")
)

ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray50") +
  scale_color_brewer(palette = "Set1") +
  labs(title = "ROC Curve Comparison (분류 모델)",
       x = "1 - Specificity (FPR)",
       y = "Sensitivity (TPR)") +
  theme_minimal()

# ── 성능 비교 시각화 ──────────────────────────────────────
cls_df %>%
  dplyr::select(Model, Accuracy, Precision, Recall, F1, AUC) %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1, AUC),
               names_to = "Metric", values_to = "Value") %>%
  filter(!is.na(Value)) %>%
  ggplot(aes(x = reorder(Model, Value), y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "분류 모델 성능 비교",
       x = "Model", y = "Score") +
  theme_minimal()

df_cls_cor <- trade[, c("conflict_onset", feature_vars_final)]

df_cls_cor$conflict_onset <- as.numeric(
  as.character(df_cls_cor$conflict_onset)
)

cor_mat_cls <- cor(df_cls_cor)

corrplot(
  cor_mat_cls,
  method = "color",
  tl.col = "black",
  tl.cex = 0.8
)

cor(
  trade$sanction_exposure_index,
  as.numeric(as.character(trade$conflict_onset))
)

# conflict_onset 숫자형 변환
conflict_num <- as.numeric(as.character(trade$conflict_onset))

# 확인할 변수들
vars <- c(
  "sanction_exposure_index",
  "network_centrality_score",
  "inflation_rate",
  "economic_growth_rate"
)

# 상관계수
sapply(vars, function(v) {
  cor(trade[[v]], conflict_num)
})
# ============================================================
# 회귀 모델링: conflict_risk_score 예측
# ============================================================

library(randomForest)
library(xgboost)
library(rpart)
library(caret)
library(ggplot2)

# ─────────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────────

df_reg <- trade[, c("conflict_risk_score", feature_vars_final)]

# train / test split
set.seed(123)

reg_idx <- createDataPartition(
  df_reg$conflict_risk_score,
  p = 0.7,
  list = FALSE
)

train_reg <- df_reg[reg_idx, ]
test_reg  <- df_reg[-reg_idx, ]

# ─────────────────────────────────────────────
# scaling
# ─────────────────────────────────────────────

train_x <- train_reg[, feature_vars_final]
test_x  <- test_reg[, feature_vars_final]

train_scaled_x <- scale(train_x)

reg_mean <- attr(train_scaled_x, "scaled:center")
reg_sd   <- attr(train_scaled_x, "scaled:scale")

test_scaled_x <- scale(
  test_x,
  center = reg_mean,
  scale = reg_sd
)

train_scaled <- as.data.frame(train_scaled_x)
test_scaled  <- as.data.frame(test_scaled_x)

train_scaled$conflict_risk_score <- train_reg$conflict_risk_score
test_scaled$conflict_risk_score  <- test_reg$conflict_risk_score

# ─────────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────────

eval_reg <- function(pred, actual, model_name = "") {
  
  rmse_val <- RMSE(pred, actual)
  mae_val  <- MAE(pred, actual)
  
  r2_val <- cor(pred, actual)^2
  
  cat("\n==============================\n")
  cat("모델:", model_name, "\n")
  cat("RMSE :", round(rmse_val, 4), "\n")
  cat("MAE  :", round(mae_val, 4), "\n")
  cat("R²   :", round(r2_val, 4), "\n")
  cat("==============================\n")
  
  data.frame(
    Model = model_name,
    RMSE  = round(rmse_val, 4),
    MAE   = round(mae_val, 4),
    R2    = round(r2_val, 4)
  )
}

reg_results <- list()

# ============================================================
# 1. Linear Regression
# ============================================================

lm_model <- lm(
  conflict_risk_score ~ .,
  data = train_scaled
)

lm_pred <- predict(
  lm_model,
  newdata = test_scaled
)

reg_results[["Linear"]] <- eval_reg(
  lm_pred,
  test_scaled$conflict_risk_score,
  "Linear Regression"
)

# ============================================================
# 2. Decision Tree Regression
# ============================================================

tree_reg <- rpart(
  conflict_risk_score ~ .,
  data = train_scaled,
  method = "anova",
  control = rpart.control(
    cp = 0.001,
    maxdepth = 8
  )
)

tree_pred <- predict(
  tree_reg,
  newdata = test_scaled
)

reg_results[["Tree"]] <- eval_reg(
  tree_pred,
  test_scaled$conflict_risk_score,
  "Decision Tree"
)

# ============================================================
# 3. Random Forest Regression
# ============================================================

set.seed(42)

rf_reg <- randomForest(
  conflict_risk_score ~ .,
  data = train_scaled,
  ntree = 300,
  importance = TRUE
)

rf_pred <- predict(
  rf_reg,
  newdata = test_scaled
)

reg_results[["RF"]] <- eval_reg(
  rf_pred,
  test_scaled$conflict_risk_score,
  "Random Forest"
)

# 변수 중요도
varImpPlot(
  rf_reg,
  main = "RF Variable Importance (Regression)"
)

# ============================================================
# 4. XGBoost Regression
# ============================================================

xgb_train_x <- as.matrix(
  train_scaled[, feature_vars_final]
)

xgb_train_y <- train_scaled$conflict_risk_score

xgb_test_x <- as.matrix(
  test_scaled[, feature_vars_final]
)

set.seed(42)

xgb_reg <- xgboost(
  x = xgb_train_x,
  y = xgb_train_y,
  objective = "reg:squarederror",
  nrounds = 100,
  max_depth = 4,
  learning_rate = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

xgb_pred <- predict(
  xgb_reg,
  newdata = xgb_test_x
)

reg_results[["XGB"]] <- eval_reg(
  xgb_pred,
  test_scaled$conflict_risk_score,
  "XGBoost"
)


# ============================================================
# 성능 비교
# ============================================================

reg_df <- do.call(rbind, reg_results)
rownames(reg_df) <- NULL

cat("\n=== 회귀 모델 성능 비교 ===\n")
print(reg_df)

# ============================================================
# 시각화
# ============================================================

library(tidyr)

reg_long <- pivot_longer(
  reg_df,
  cols = c(RMSE, MAE, R2),
  names_to = "Metric",
  values_to = "Value"
)

ggplot(reg_long,
       aes(x = Model,
           y = Value,
           fill = Metric)) +
  geom_bar(
    stat = "identity",
    position = "dodge"
  ) +
  labs(
    title = "회귀 모델 성능 비교",
    x = "Model",
    y = "Score"
  ) +
  theme_minimal()

#Actual vs Predicted plot
ggplot(data.frame(
  Actual = test_scaled$conflict_risk_score,
  Predicted = rf_pred
),
aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(
    slope = 1,
    intercept = 0,
    color = "red"
  ) +
  theme_minimal() +
  labs(
    title = "Actual vs Predicted (RF)",
    x = "Actual",
    y = "Predicted"
  )
#2. Residual plot
residuals_rf <- test_scaled$conflict_risk_score - rf_pred

ggplot(data.frame(
  Predicted = rf_pred,
  Residuals = residuals_rf
),
aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0,
             color = "red") +
  theme_minimal()


# =========================
# SHAP용 XGBoost 다시 학습
# =========================

library(xgboost)
library(ggplot2)

# -------------------------
# matrix 생성
# -------------------------

xgb_train_x <- as.matrix(
  train_scaled[, feature_vars_final]
)

xgb_test_x <- as.matrix(
  test_scaled[, feature_vars_final]
)

xgb_train_y <- train_scaled$conflict_risk_score

# -------------------------
# DMatrix 생성
# -------------------------

dtrain_reg <- xgb.DMatrix(
  data = xgb_train_x,
  label = xgb_train_y
)

dtest_reg <- xgb.DMatrix(
  data = xgb_test_x
)

# -------------------------
# parameter 설정
# -------------------------

params_reg <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  
  eta = 0.05,
  max_depth = 4,
  
  subsample = 0.8,
  colsample_bytree = 0.8
)

# -------------------------
# 모델 학습
# -------------------------

set.seed(42)

xgb_reg2 <- xgb.train(
  params = params_reg,
  data = dtrain_reg,
  nrounds = 100
)

# =========================
# SHAP VALUE 계산
# =========================

shap_matrix <- predict(
  xgb_reg2,
  dtrain_reg,
  predcontrib = TRUE
)

# 마지막 bias 제거
shap_matrix <- shap_matrix[, -ncol(shap_matrix)]

# data frame 변환
shap_df <- as.data.frame(shap_matrix)

# =========================
# 평균 SHAP 중요도
# =========================

mean_shap <- apply(
  abs(shap_df),
  2,
  mean
)

mean_shap <- sort(
  mean_shap,
  decreasing = TRUE
)

print(mean_shap)

# =========================
# SHAP 중요도 시각화
# =========================

shap_importance <- data.frame(
  Variable = names(mean_shap),
  Importance = mean_shap
)

ggplot(
  shap_importance,
  aes(
    x = reorder(Variable, Importance),
    y = Importance
  )
) +
  geom_bar(
    stat = "identity",
    fill = "steelblue"
  ) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "SHAP Feature Importance",
    x = "Variable",
    y = "Mean |SHAP value|"
  )

# =========================
# Dependence Plot
# =========================

dep_df <- data.frame(
  sanction_exposure_index =
    train_scaled$sanction_exposure_index,
  
  shap_value =
    shap_df$sanction_exposure_index
)

ggplot(
  dep_df,
  aes(
    x = sanction_exposure_index,
    y = shap_value
  )
) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(
    title = "SHAP Dependence Plot",
    x = "sanction_exposure_index",
    y = "SHAP value"
  )



# ============================================================
# 회귀 모델링 (Original Data 사용 버전)
# conflict_risk_score 예측
# ============================================================

library(randomForest)
library(xgboost)
library(rpart)
library(caret)
library(ggplot2)
library(Metrics)

# ------------------------------------------------
# 데이터 준비
# ------------------------------------------------

df_reg <- trade[, c("conflict_risk_score", feature_vars_final)]

# train / test split
set.seed(123)

reg_idx <- createDataPartition(
  df_reg$conflict_risk_score,
  p = 0.7,
  list = FALSE
)

train_reg <- df_reg[reg_idx, ]
test_reg  <- df_reg[-reg_idx, ]

# ------------------------------------------------
# 평가 함수
# ------------------------------------------------

eval_reg <- function(pred, actual, model_name = "") {
  
  rmse_val <- RMSE(pred, actual)
  mae_val  <- MAE(pred, actual)
  r2_val   <- cor(pred, actual)^2
  
  cat("\n==============================\n")
  cat("모델:", model_name, "\n")
  cat("RMSE :", round(rmse_val, 4), "\n")
  cat("MAE  :", round(mae_val, 4), "\n")
  cat("R²   :", round(r2_val, 4), "\n")
  cat("==============================\n")
  
  data.frame(
    Model = model_name,
    RMSE  = round(rmse_val, 4),
    MAE   = round(mae_val, 4),
    R2    = round(r2_val, 4)
  )
}

reg_results <- list()

# ============================================================
# 1. Linear Regression
# ============================================================

lm_model <- lm(
  conflict_risk_score ~ .,
  data = train_reg
)

lm_pred <- predict(
  lm_model,
  newdata = test_reg
)

reg_results[["Linear"]] <- eval_reg(
  lm_pred,
  test_reg$conflict_risk_score,
  "Linear Regression"
)

# ============================================================
# 2. Decision Tree Regression
# ============================================================

tree_reg <- rpart(
  conflict_risk_score ~ .,
  data = train_reg,
  method = "anova",
  control = rpart.control(
    cp = 0.001,
    maxdepth = 8
  )
)

tree_pred <- predict(
  tree_reg,
  newdata = test_reg
)

reg_results[["Tree"]] <- eval_reg(
  tree_pred,
  test_reg$conflict_risk_score,
  "Decision Tree"
)

# ============================================================
# 3. Random Forest Regression
# ============================================================

set.seed(42)

rf_reg <- randomForest(
  conflict_risk_score ~ .,
  data = train_reg,
  ntree = 300,
  importance = TRUE
)

rf_pred <- predict(
  rf_reg,
  newdata = test_reg
)

reg_results[["RF"]] <- eval_reg(
  rf_pred,
  test_reg$conflict_risk_score,
  "Random Forest"
)

# 변수 중요도
varImpPlot(
  rf_reg,
  main = "RF Variable Importance (Regression)"
)

# ============================================================
# 4. XGBoost Regression
# ============================================================

xgb_train_x <- as.matrix(
  train_reg[, feature_vars_final]
)

xgb_train_y <- train_reg$conflict_risk_score

xgb_test_x <- as.matrix(
  test_reg[, feature_vars_final]
)

set.seed(42)

xgb_reg <- xgboost(
  x = xgb_train_x,
  y = xgb_train_y,
  objective = "reg:squarederror",
  nrounds = 100,
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)

xgb_pred <- predict(
  xgb_reg,
  newdata = xgb_test_x
)

reg_results[["XGB"]] <- eval_reg(
  xgb_pred,
  test_reg$conflict_risk_score,
  "XGBoost"
)

# ============================================================
# 성능 비교표
# ============================================================

reg_df <- do.call(rbind, reg_results)
rownames(reg_df) <- NULL

cat("\n=== 회귀 모델 성능 비교 ===\n")
print(reg_df)

# ============================================================
# 성능 시각화
# ============================================================

library(tidyr)

reg_long <- pivot_longer(
  reg_df,
  cols = c(RMSE, MAE, R2),
  names_to = "Metric",
  values_to = "Value"
)

ggplot(
  reg_long,
  aes(x = Model,
      y = Value,
      fill = Metric)
) +
  geom_bar(
    stat = "identity",
    position = "dodge"
  ) +
  labs(
    title = "회귀 모델 성능 비교",
    x = "Model",
    y = "Score"
  ) +
  theme_minimal()

# ============================================================
# Actual vs Predicted (RF)
# ============================================================

ggplot(
  data.frame(
    Actual = test_reg$conflict_risk_score,
    Predicted = rf_pred
  ),
  aes(x = Actual, y = Predicted)
) +
  geom_point(alpha = 0.5) +
  geom_abline(
    slope = 1,
    intercept = 0,
    color = "red"
  ) +
  theme_minimal() +
  labs(
    title = "Actual vs Predicted (RF)",
    x = "Actual",
    y = "Predicted"
  )

# ============================================================
# Residual Plot (RF)
# ============================================================

residuals_rf <- test_reg$conflict_risk_score - rf_pred

ggplot(
  data.frame(
    Predicted = rf_pred,
    Residuals = residuals_rf
  ),
  aes(x = Predicted, y = Residuals)
) +
  geom_point(alpha = 0.5) +
  geom_hline(
    yintercept = 0,
    color = "red"
  ) +
  theme_minimal() +
  labs(
    title = "Residual Plot (RF)",
    x = "Predicted",
    y = "Residuals"
  )



# ============================================================
# conflict_onset 분포 확인
# ============================================================

# 개수 확인
table(trade$conflict_onset)

# 비율 확인
prop.table(table(trade$conflict_onset))

# 데이터프레임으로 변환
conflict_dist <- as.data.frame(
  table(trade$conflict_onset)
)

colnames(conflict_dist) <- c(
  "conflict_onset",
  "Count"
)

# 비율 추가
conflict_dist$Proportion <- round(
  conflict_dist$Count / sum(conflict_dist$Count),
  4
)

print(conflict_dist)

# ============================================================
# 시각화
# ============================================================

library(ggplot2)

ggplot(
  conflict_dist,
  aes(
    x = conflict_onset,
    y = Count,
    fill = conflict_onset
  )
) +
  geom_bar(
    stat = "identity",
    width = 0.6
  ) +
  geom_text(
    aes(
      label = paste0(
        Count,
        "\n(",
        round(Proportion * 100, 1),
        "%)"
      )
    ),
    vjust = -0.3,
    size = 5
  ) +
  scale_fill_manual(
    values = c(
      "0" = "steelblue",
      "1" = "coral"
    )
  ) +
  labs(
    title = "Distribution of conflict_onset",
    x = "conflict_onset",
    y = "Count"
  ) +
  theme_minimal()


library(ggplot2)
library(gridExtra)

# =====================================================
# 1. conflict_onset 분포
# =====================================================

conflict_dist <- as.data.frame(
  table(trade$conflict_onset)
)

colnames(conflict_dist) <- c(
  "conflict_onset",
  "Count"
)

conflict_dist$Proportion <- round(
  conflict_dist$Count / sum(conflict_dist$Count),
  4
)

p1 <- ggplot(
  conflict_dist,
  aes(
    x = conflict_onset,
    y = Count,
    fill = conflict_onset
  )
) +
  geom_bar(
    stat = "identity",
    width = 0.6
  ) +
  geom_text(
    aes(
      label = paste0(
        Count,
        "\n(",
        round(Proportion * 100, 1),
        "%)"
      )
    ),
    vjust = -0.3,
    size = 4
  ) +
  scale_fill_manual(
    values = c(
      "0" = "steelblue",
      "1" = "coral"
    )
  ) +
  labs(
    title = "Distribution of conflict_onset",
    x = "conflict_onset",
    y = "Count"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# =====================================================
# 2. sanction_exposure_index Boxplot
# =====================================================

p2 <- ggplot(
  trade,
  aes(
    x = conflict_onset,
    y = sanction_exposure_index,
    fill = conflict_onset
  )
) +
  geom_boxplot() +
  scale_fill_manual(
    values = c(
      "0" = "steelblue",
      "1" = "coral"
    )
  ) +
  labs(
    title = "Sanction Exposure Index",
    x = "conflict_onset",
    y = "sanction_exposure_index"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# =====================================================
# 3. network_centrality_score Boxplot
# =====================================================

p3 <- ggplot(
  trade,
  aes(
    x = conflict_onset,
    y = network_centrality_score,
    fill = conflict_onset
  )
) +
  geom_boxplot() +
  scale_fill_manual(
    values = c(
      "0" = "steelblue",
      "1" = "coral"
    )
  ) +
  labs(
    title = "Network Centrality Score",
    x = "conflict_onset",
    y = "network_centrality_score"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# =====================================================
# 한 화면에 출력
# =====================================================

grid.arrange(
  p1,
  p2,
  p3,
  ncol = 3
)
