# Script file: "thesisCode.R"

### Load packages
library(QuantPsyc) 
library(dplyr)
library(glmnet)
library(gbm)
library(gam)
# library(caret), is loaded in later to avoid conflicts
library(car)
library(bestglm)

### thesisFunctions.R file.
source("thesisFunctions.R")


## Inspect workers

### Load the data
qualtrics <- read.csv2("data/qualtricsNum.csv") 
dim(qualtrics)
amazon <- read.csv2("data/amazon.csv", stringsAsFactors = FALSE)
nrow(amazon)

### Join datasets
amazon <- rename(amazon, valCode = Answer.surveycode)
qualtrics$valCode <- as.character(qualtrics$valCode)
qualtrics <- semi_join(qualtrics, amazon, by = "valCode")
dim(qualtrics)

### Reject invalid workers
amazon[!(amazon$valCode %in% qualtrics$valCode), c("WorkerId", "valCode")]

### surveyTime
qualtrics <- rename(qualtrics, surveyTime = Duration..in.seconds.)
attach(qualtrics)
describeVariable(surveyTime, lab = "survey time (in seconds)", ylim = c(0,25),
  hist = TRUE)

## Pre-processing

### Inspect the data
names(qualtrics)

### Removing variables
qualtrics <- select(qualtrics, -(StartDate:consent), -contains("attention"), 
 -(gender:valCode))
dim(qualtrics)

### Recoding variable names
names(select(qualtrics, contains("willingness"), contains("description")))
names(qualtrics)[grepl("willingness*", names(qualtrics)) | 
  grepl("description*", names(qualtrics))] <- c("negRec1", "negRec2",
  "altruism1", "posRec1", "negRec3", "trust")

### Recoding risk
names(select(qualtrics, contains("risk"), -risk1))
qualtrics <- select(qualtrics, -c(contains("20"), contains("40"), contains("60"),
  contains("80"), contains("00")))
names(select(qualtrics, contains("risk"), -risk1))
qualtrics <- mutate(qualtrics, risk2 = rowSums(select(qualtrics, 
  contains("risk"), -risk1), na.rm = TRUE)) %>% 
    select(-c(risk70:risk310))
qualtrics$risk2

### Qualitative variables
summary(qualtrics[c("dilemma", "chicken")])
qualtrics <- mutate_at(qualtrics, vars(dilemma:chicken), funs(as.factor))
levels(qualtrics$dilemma) <- c("cooperate", "defect")
levels(qualtrics$chicken) <- c("chicken", "dare")
summary(qualtrics[c("dilemma", "chicken")])

### Re-ordering
names(qualtrics)
qualtrics <- qualtrics[c(grep("public", names(qualtrics)),
  grep("offer", names(qualtrics)), grep("respond", names(qualtrics)), 
  grep("dilemma", names(qualtrics)), grep("chicken", names(qualtrics)), 
  grep("negRec*", names(qualtrics)), grep("posRec*", names(qualtrics)),
  grep("altruism*", names(qualtrics)), grep("trust*", names(qualtrics)),
  grep("risk*", names(qualtrics)))]

### General description
names(qualtrics)
dim(qualtrics)
head(qualtrics)
str(qualtrics)
attach(qualtrics)


## Univariate analysis

## Game theory

### Public goods
describeVariable(public, lab = "Amount invested (in whole dollars)", ylim = c(0,60))
### Ultimatum game: offer
describeVariable(offer, lab = "Amount offered (in whole dollars)", ylim = c(0,120))
### Ultimatum game: respond
describeVariable(respond, lab = "Minimum acceptable amount (in whole dollars)", ylim = c(0,80))
### dilemma
describeVariable(dilemma)
### Chicken game
describeVariable(chicken)

## Preferences and risk

### Trust
describeVariable(trust, lab = "trust score (in integers)", ylim = c(0,50))

### Negative Reciprocaty
describeVariable(negRec1, lab = "negRec1 score (in integers)", ylim = c(0,35))
describeVariable(negRec2, lab = "negRec2 score (in integers)", ylim = c(0,35))
describeVariable(negRec3, lab = "negRec3 score (in integers)", ylim = c(0,60))

### Positive Reciprocaty
describeVariable(posRec1, lab = "posRec1 score (in integers)", ylim = c(0,100))
posRec2Recoded <- posRec2 * 5
describeVariable(posRec2Recoded, lab = "Amount gifted to a helpful to stranger (in dollars)", 
  ylim = c(0,60))

### Altruism
describeVariable(altruism1, lab = "altruism1 score (in integers)", ylim = c(0,50))
describeVariable(altruism2, lab = "Amount donated to a good cause (in whole dollars)",
  ylim = c(0,100), hist = TRUE)

### Risk
describeVariable(risk1, lab = "risk1 score (in integers)", ylim = c(0,40))
risk2Recoded <- risk2 * 10 - 5
describeVariable(risk2Recoded, lab = "Amount of sure payment preferred (in dollars)",
  ylim = c(0,40), hist = TRUE)


## Multi-indicator concepts
preferences <- select(qualtrics, c(negRec1:risk2))
plotCor(preferences)

### negRec
cbAlpha(negRec1, negRec2, negRec3)

#### negRecPooled
qualtrics$negRecPooled <- cbAlpha(negRec1, negRec2, negRec3)$alpha$scores
attach(qualtrics)
describeVariable(negRecPooled, lab = "negRecPooled Score", ylim = c(0,30), 
  hist = TRUE)

#### Re-ordering variables
qualtrics <- select(qualtrics, -c(negRec1:negRec3))
indexNegPooled <- grep("negRecPooled", names(qualtrics))
qualtrics <- qualtrics[c(1:5, indexNegPooled, 6:(indexNegPooled-1))]
names(qualtrics)

### posRec
cbAlpha(posRec1, posRec2)

### altruism
cbAlpha(altruism1, altruism2)
altruism2Log <- log(ifelse(altruism2 == 0, 1, altruism2))
cbAlpha(altruism1, altruism2Log)

## Risk
cbAlpha(risk1, risk2)

### Joined correlations
preferences <- model.matrix(~ ., data = qualtrics)[,-c(1:6)]
plotCor(preferences)


## Model building

### Test set
set.seed(256)
test <- sample(1:nrow(qualtrics), size = nrow(qualtrics) / 6)
testQualtrics <- qualtrics[test,]
dim(testQualtrics)

### Train set
trainQualtrics <- qualtrics[-test,]
dim(trainQualtrics)
trainPreferences <- preferences[-test,]
dim(trainPreferences)
trainPreferencesScaled <- data.frame(scale(trainPreferences))
trainPreferencesScaled <- model.matrix(~ ., data = trainPreferencesScaled)

### Cross-validation
set.seed(128)
foldid <- sample(rep(1:5, length = nrow(trainQualtrics)))
table(foldid)


## Public goods
trainPublic <- select(trainQualtrics, -c(offer, respond, dilemma, chicken))
attach(trainPublic)

### Bivariate analysis
plotCor(trainPublic)

### Base model
baseline <- baseModelcv(trainPublic, foldid)
baseline
mseBase <- baseline$meancv

### Best subset selection
bestPublic <- regsubsets(public ~ ., data = trainPublic)
plotR2Subs(bestPublic)

bestSubs <- bestSubsetcv(trainPublic, foldid)
bestSubs
mseBestSub <- bestSubs$bestnmean
plotModels(bestSubs$meancv, baseline)

bestSubMdl <- public ~ altruism1 + altruism2
bestSubFit <- lm(bestSubMdl, data = trainPublic)
summary(bestSubFit)
lm.beta(bestSubFit)

### Lasso
lassocv <- cv.glmnet(trainPreferences, public, alpha = 1, foldid = foldid)
resultsLasso(lassocv)
mseLasso <- min(lassocv$cvm)
plot(lassocv)

plotShrinkage(trainPreferences, public, lassocv)
legend('topleft', legend = c("altruism1", "altruism2", "trust", "posRec2"), 
  lwd = 2, col= c("blue2", "cyan", "mediumorchid1", "seagreen3"))
predict(lassocv, type = "coefficients", s = lassocv$lambda.min)

plotShrinkage(trainPreferencesScaled, public, lassocv)
legend('topleft', legend = c("altruism1", "altruism2", "trust", "posRec2"), 
  lwd = 2, col= c("blue2", "cyan", "mediumorchid1", "seagreen3"))

### Non-linearity
residualPlots((bestSubFit),pch=19)

plotBivariate(altruism1, public)

mdl2 <- public ~ poly(altruism1, 2) + altruism2
mdl3 <- public ~ poly(altruism1, 3) + altruism2
mdl4 <- public ~ poly(altruism1, 4) + altruism2
models <- c(bestSubMdl, mdl2, mdl3, mdl4)

polynomial <- nonLinearcv(trainPublic, foldid, models)
polynomial
msePoly <- polynomial$bestnmean

plotModels(polynomial$meancv, baseline, xlab = "Models plus polynomial to nth degree",
  axislabels = c("Base", "Bestsub", 2:4))

poly3Altruism1 <- lm(public ~ poly(altruism1, 3), data = trainPublic)
preds <- predict(poly3Altruism1, newdata = list(altruism1 = 0:max(altruism1)))
plotBivariate(altruism1, public)
lines(0:max(altruism1), preds, col = "blue", lwd = 2)

### Boosting
library(caret)

fitControl <- trainControl(method = "cv", number = 5)
gbmGrid <- expand.grid(interaction.depth = c(1, 2, 4, 8), n.trees = (1:10)*200, 
  n.minobsinnode = 10, shrinkage = c(0.001))

set.seed(64)
gbmfit <- train(public ~ ., data = trainPublic, method = "gbm", verbose = FALSE,
  trControl = fitControl, tuneGrid = gbmGrid, distribution = "gaussian")
gbmfit

plot(gbmfit)
gbmfit$results[which.min(gbmfit$results$RMSE),1:6]
mseBoost <- min(gbmfit$results$RMSE)^2
mseBoost

set.seed(32)
gbmBestFit <- gbm(public ~ ., data = trainPublic, distribution = "gaussian",
  n.trees= 1200, interaction.depth = 2)
summary(gbmBestFit)

### Final models
finalModels <- data.frame(MSE = c(mseBase, mseBestSub, mseLasso, msePoly, mseSqrt, 
  mseCombined, mseBoost), row.names = c("Base", "BestSub", "Lasso", "Poly", "Sqrt",
  "Combo", "Boost"))
finalModels

plotModels(finalModels$MSE, xlab = "Final models", axislabels = rownames(finalModels))

### Final predictions
ytrue <- qualtrics[test, "public"]
baseMse <- mean((ytrue - mean(public))^2)
baseMse
sqrt(baseMse)

finalModel <- public ~ poly(altruism1, 3) + sqrt(altruism2) + trust
finalFit <- lm(finalModel, data = trainPublic)
yhat <- predict(finalFit, qualtrics[test,])
finalMse <- mean((ytrue - yhat)^2)
finalMse
sqrt(finalMse)

### Significance testing
qualtricsPublic <- select(qualtrics, -c(offer, respond, dilemma, chicken)) 
attach(qualtricsPublic)
plotCor(qualtricsPublic, pvalues = TRUE)

bestSubFit <- lm(bestSubMdl, data = qualtricsPublic)
summary(bestSubFit)

mdl2 <- public ~ poly(altruism1, 3) + altruism2
fit2 <- lm(mdl2, data = qualtricsPublic)
summary(fit2)

mdl3 <- public ~ poly(altruism1, 3) + sqrt(altruism2)
fit3 <- lm(mdl3, data = qualtricsPublic)
summary(fit3)

finalFit <- lm(finalModel, data = qualtricsPublic)
summary(finalFit)

anova(bestSubFit, fit2, fit3, finalFit)

### Diagnostics
par(mfrow=c(2,2))
plot(finalFit, pch = 19) 

#### Non-linearity
par(mfrow=c(1,1))
residualPlots((lm(finalFit, data = qualtricsPublic)), pch=19)

#### Correlation of error terms
durbinWatsonTest(finalFit)

#### Heteroskedasticity
plot(finalFit, which = 1, pch = 19)

#### Outliers
rstud <- rstudent(finalFit)
plot(rstud, pch = 19, main = "Rstudent plot")

#### High leverage points 
hlev <- hatvalues(finalFit)
plot(hlev, pch = 19, main = "High leverage plot")
abline(h = 0.12, col = "red", lty = 2)

summary(lm(finalFit, data = qualtricsPublic))
summary(lm(finalFit, data = qualtricsPublic[!(hlev > 0.10),]))

#### Influential points
plot(finalFit, which = 5, pch = 19)
plot(finalFit, which = 4, pch = 19)

#### Collinearity
vif(finalFit) 
plotCor(data.frame(altruism1, altruism2, trust))


## Offer
trainOffer <- select(trainQualtrics, -c(public, respond, dilemma, chicken))
attach(trainOffer)

### Bivariate analysis
plotCor(trainOffer)

### Baseline model
baseline <- baseModelcv(trainOffer, foldid)
baseline
mseBase <- baseline$meancv

### Best subset selection
bestOffer <- regsubsets(offer ~ ., data = trainOffer)
plotR2Subs(bestOffer)

bestSubs <- bestSubsetcv(trainOffer, foldid)
bestSubs
mseBestSub <- bestSubs$bestnmean
plotModels(bestSubs$meancv, baseline)

bestSubFit <- lm(offer ~ posRec2, data = trainOffer)
summary(bestSubFit)
plotBivariate(posRec2, offer)

### Lasso
lassocv <- cv.glmnet(trainPreferences, offer, alpha = 1, foldid = foldid)
resultsLasso(lassocv)
mseLasso <- min(lassocv$cvm)
plot(lassocv)

plotShrinkage(trainPreferences, offer, lassocv)
legend('topleft', legend = c("posRec2"), lwd = 2, col= c("seagreen3"))
predict(lassocv, type = "coefficients", s = lassocv$lambda.min)

### Non-linearity
residualPlots((bestSubFit), pch=19)

### Boosting
fitControl <- trainControl(method = "cv", number = 5)
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 4, 8), n.trees = (0.1:10)*100,
  n.minobsinnode = 20, shrinkage = c(0.001))

set.seed(64)
gbmFit <- train(offer ~ ., data = trainOffer, method = "gbm", verbose = FALSE,
  trControl = fitControl, tuneGrid = gbmGrid, distribution = "gaussian")
gbmFit
plot(gbmFit)
gbmFit$results[which.min(gbmFit$results$RMSE),1:6]
mseBoost <- min(gbmFit$results$RMSE)^2
mseBoost

set.seed(32)
gbmBestFit <- gbm(offer ~ ., data = trainOffer, distribution = "gaussian",
  n.trees= 710, n.minobsinnode = 20)
summary(gbmBestFit)

### Final models
finalModels <- data.frame(MSE = c(mseBase, mseBestSub, mseLasso, mseBoost), 
  row.names = c("Base", "BestSub", "Lasso", "Boost"))
finalModels

plotModels(finalModels$MSE, xlab = "Final models", axislabels = rownames(finalModels))

### Final predictions
ytrue <- qualtrics[test, "offer"]
baseMse <-  mean((ytrue - mean(trainOffer$offer))^2)
baseMse
sqrt(baseMse)

yhat <- predict(gbmBestFit, newdata = qualtrics[test,], n.trees = 710)
finalFit <- mean((ytrue - yhat)^2)
finalFit
sqrt(finalFit)

### Final model inspection
qualtricsOffer <- select(qualtrics, -c(public, respond, dilemma, chicken)) 

set.seed(32)
gbmFinal <- gbm(offer ~ ., data = qualtricsOffer, distribution = "gaussian", n.trees= 710, n.minobsinnode = 20)
summary(gbmFinal)

plot(gbmFinal, i = "posRec2", main = "Bivariate plot gbm")
plot(gbmFinal, i = "altruism2", main = "Bivariate plot gbm")


## Respond
trainRespond <- select(trainQualtrics, -c(public, offer, dilemma, chicken))
attach(trainRespond)

### Bivariate analysis
plotCor(trainRespond)

### Base model
baseline <- baseModelcv(trainRespond, foldid)
baseline
mseBase <- baseline$meancv

### Best subset selection
bestRespond <- regsubsets(respond ~ ., data = trainRespond)
plotR2Subs(bestRespond)

bestSubs <- bestSubsetcv(trainRespond, foldid)
bestSubs
mseBestSub <- bestSubs$bestnmean
plotModels(bestSubs$meancv, baseline)

bestSubMdl <- respond ~ negRecPooled
bestSubFit <- lm(bestSubMdl, data = trainPublic)
summary(bestSubFit)
plotBivariate(negRecPooled, respond)

### Lasso
lassocv <- cv.glmnet(trainPreferences, respond, alpha = 1, foldid = foldid)
resultsLasso(lassocv)
mseLasso <- min(lassocv$cvm)
plot(lassocv)

plotShrinkage(trainPreferences, respond, lassocv)
legend('topleft', legend = c("negRecPooled"), lwd = 2, col= c("black"))
predict(lassocv, type = "coefficients", s = lassocv$lambda.min)

### Non-linearity
residualPlots((bestSubFit), spch=19)
mdl2 <- respond ~ poly(negRecPooled, 2)
mdl3 <- respond ~ poly(negRecPooled, 3)
mdl4 <- respond ~ poly(negRecPooled, 4)
models <- c(bestSubMdl, mdl2, mdl3, mdl4)

polynomial <- nonLinearcv(trainRespond, foldid, models)
polynomial
msePoly <- polynomial$bestnmean

plotModels(polynomial$meancv)

### Smoothing spline
smoothSpline <- smoothSplinecv(trainRespond, foldid, "negRecPooled", maxdegree = 10)
smoothSpline
mseSmooth <- smoothSpline$bestnmean
plotModels(smoothSpline$meancv, baseline, xlab = "degrees of freedom",
  axislabels = c("Base", "Bestsub", 3:10))

gam3 <- gam(respond ~ s(negRecPooled, 3))
summary(gam3)
preds <- predict(gam3, newdata = list(negRecPooled = 0:max(negRecPooled)))

plotBivariate(negRecPooled, respond)
lines(0:max(negRecPooled), preds, col = "blue", lwd = 2)

### Boosting
fitControl <- trainControl(method = "cv", number = 5)
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 4, 8), n.trees = (1:10)*200, 
  n.minobsinnode = 10, shrinkage = 0.001)

set.seed(64)
gbmFit <- train(respond ~ ., data = trainRespond, method = "gbm", verbose = FALSE, trControl = fitControl, tuneGrid = gbmGrid, distribution = "gaussian")
gbmFit
plot(gbmFit)
gbmFit$results[which.min(gbmFit$results$RMSE),1:6]
mseBoost <- min(gbmFit$results$RMSE)^2
mseBoost

set.seed(32)
gbmBestFit <- gbm(respond ~ ., data = trainRespond, distribution = "gaussian", n.trees = 1400)
summary(gbmBestFit)

### Final models
finalModels <- data.frame(MSE = c(mseBase, mseBestSub, mseLasso, msePoly, 
  mseSmooth, mseBoost), row.names = c("Base", "BestSub", "Lasso", "Poly", 
  "Smooth", "Boost"))
finalModels

plotModels(finalModels$MSE, xlab = "Final models", axislabels = rownames(finalModels))

### Final predictions
ytrue <- qualtrics[test, "respond"]
baseMse <- mean((ytrue - mean(respond))^2)
baseMse
sqrt(baseMse)

yhat <- predict(gbmBestFit, newdata = qualtrics[test,], n.trees = 1400)
finalFit <- mean((ytrue - yhat)^2)
finalFit
sqrt(finalFit)

### Final model inspection
qualtricsRespond <- select(qualtrics, -c(public, offer, dilemma, chicken)) 

set.seed(32)
gbmBest <- gbm(respond ~ ., data = qualtricsRespond, distribution = "gaussian",
  n.trees = 1400, interaction.depth = 1, shrinkage = 0.001)
summary(gbmBest)

plot(gbmBest, i = "negRecPooled", main = "Bivariate plot gbm")
plot(gbmBest, i = "risk2", main = "Bivariate plot gbm")


## Prisoner's dilemma
trainDilemma <- select(trainQualtrics, -c(chicken, offer, respond, public))
attach(trainDilemma)
dilemmaBinary <- ifelse(dilemma == "cooperate", 0, 1)

### Bivariate analysis
plotBoxes(trainQualtrics, dilemma)

### Base model
baseline <- baseModelcv(trainDilemma, foldid)
baseline
accBase <- baseline$meancv

### Best subset selection
set.seed(32)
bestDilemma <- bestglm(data.frame(trainPreferences, dilemmaBinary), IC = "CV",
  CVArgs = list(Method="HTF", K=5, REP=1), family=binomial)
bestDilemma$Subsets

mdl1 <- dilemma ~ altruism1
mdl2 <- dilemma ~ altruism1 + trust
mdl3 <- dilemma ~ altruism1 + trust + negRecPooled
mdl4 <- dilemma ~ altruism1 + trust + negRecPooled + posRec2
mdl5 <- dilemma ~ altruism1 + trust + negRecPooled + posRec2 + altruism2
mdl6 <- dilemma ~ altruism1 + trust + negRecPooled + posRec2 + altruism2 + risk2
mdl7 <- dilemma ~ altruism1 + trust + negRecPooled + posRec2 + altruism2 + risk2 + risk1
mdl8 <- dilemma ~ .
models <- c(mdl1, mdl2, mdl3, mdl4, mdl5, mdl6, mdl7, mdl8)

bestSubs <- bestSubsetcv(trainDilemma, foldid, models)
bestSubs
accBestSub <- bestSubs$bestnmean

plotModels(bestSubs$meancv, baseline, classification = TRUE)

fit3 <- glm(dilemma ~ altruism1 + trust + negRecPooled, family = binomial)
summary(fit3)

plotBivariate(altruism1, dilemmaBinary)
plotBivariate(trust, dilemmaBinary)
plotBivariate(negRecPooled, dilemmaBinary)

plotThreshold(trainDilemma, foldid, mdl3)
plotThreshold(trainDilemma, foldid, mdl3, "ROC")

### Lasso
lassocv <- cv.glmnet(trainPreferences, dilemma, alpha = 1, foldid = foldid,
  family = "binomial", type.measure = "class")
resultsLasso(lassocv, measure = "acc")
accLasso <- 1 - min(lassocv$cvm)
plot(lassocv)

plotShrinkage(trainPreferences, dilemma, lassocv)
legend('topleft', legend = c("posRec2", "negRecPooled", "trust", "altruism1"), 
  lwd = 2, col= c("seagreen3", "black", "mediumorchid1", "blue2"))
predict(lassocv, type = "coefficients", s = lassocv$lambda.min)

plotShrinkage(trainPreferencesScaled, dilemma, lassocv)
legend('topleft', legend = c("posRec2", "negRecPooled", "trust", "altruism1"), 
  lwd = 2, col= c("seagreen3", "black", "mediumorchid1", "blue2"))

### Boosting
fitControl <- trainControl(method = "cv", number = 5)
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 4, 8), n.trees = (1:10)*600,
  n.minobsinnode = 20, shrinkage = 0.001)

set.seed(64)
gbmFit <- train(dilemma ~ ., data = trainDilemma, method = "gbm", verbose = FALSE,
  trControl = fitControl, tuneGrid = gbmGrid, distribution = "bernoulli")
gbmFit
plot(gbmFit)
gbmFit$results[which.max(gbmFit$results$Accuracy),1:6]
accBoost <- max(gbmFit$results$Accuracy)

set.seed(32)
gbmBestFit <- gbm(dilemmaBinary ~ . -dilemma, data = trainDilemma, n.trees = 1800,
  interaction.depth = 4, distribution = "bernoulli")
summary(gbmBestFit)

### Final models
finalModels <- data.frame(acc = c(accBase, accBestSub, accLasso, accBoost), 
  row.names = c(c("Base", "BestSub", "Lasso", "Boost")))
finalModels

plotModels(finalModels$acc, xlab = "Final models", axislabels = rownames(finalModels), 
  classification = TRUE)

### Final predictions
ytrue <- qualtrics[test, "dilemma"]

majorityIndex <- which.max(table(trainDilemma$dilemma))
majorityClass <- levels(trainDilemma$dilemma)[majorityIndex]
sum(ytrue == majorityClass)/length(ytrue)

finalFit <- glmnet(trainPreferences, dilemma, alpha = 1, family = "binomial", lambda =  lassocv$lambda.min)
yhatProb <- predict(finalFit, s = lassocv$lambda.min,  newx = preferences[test,], type = "response")
yhatClass <- ifelse(yhatProb < 0.5, "cooperate", "defect")
sum(ytrue == yhatClass)/length(ytrue)

### Final model inspection
qualtricsDilemma <- select(qualtrics, -c(public, offer, respond, chicken)) 
attach(qualtricsDilemma)

finalFit <- glmnet(preferences, dilemma, alpha = 1, family = "binomial", lambda = lassocv$lambda.min)
finalFit
coef(finalFit)

## Chicken game
trainChicken <- select(trainQualtrics, -c(dilemma, offer, respond, public))
attach(trainChicken)
chickenBinary <- ifelse(chicken == "chicken", 0, 1)

### Bivariate analysis
plotBoxes(trainQualtrics, chicken)

### Base model
baseline <- baseModelcv(trainChicken, foldid)
baseline
accBase <- baseline$meancv

### Best subset selection
set.seed(32)
bestChicken <- bestglm(data.frame(trainPreferences, chickenBinary), IC = "CV",
  CVArgs = list(Method="HTF", K=5, REP=1), family=binomial)
bestChicken$Subsets

mdl1 <- chicken ~ altruism2
mdl2 <- chicken ~ altruism2 + trust
mdl3 <- chicken ~ altruism2 + trust + risk1
mdl4 <- chicken ~ altruism2 + risk1 + altruism1 + posRec1
mdl5 <- chicken ~ altruism2 + risk1 + altruism1 + posRec1 + trust
mdl6 <- chicken ~ altruism2 + risk1 + altruism1 + posRec1 + trust + posRec2
mdl7 <- chicken ~ altruism2 + risk1 + altruism1 + posRec1 + trust + posRec2 + risk2
mdl8 <- chicken ~ .
models <- c(mdl1, mdl2, mdl3, mdl4, mdl5, mdl6, mdl7, mdl8)

bestSubs <- bestSubsetcv(trainChicken, foldid, models)
bestSubs
accBestSub <- bestSubs$bestnmean

plotModels(bestSubs$meancv, baseline, classification = TRUE)

fit3 <- glm(mdl3, trainDilemma, family = "binomial")
summary(fit3)

trainChickenScaled <- data.frame(chicken, trainPreferencesScaled)
scaledFit3 <- glm(mdl3, trainChickenScaled, family = "binomial")
coef(scaledFit3)

plotBivariate(altruism2, chickenBinary)
plotBivariate(trust, chickenBinary)
plotBivariate(risk1, chickenBinary)

plotThreshold(trainChicken, foldid, mdl3)
plotThreshold(trainChicken, foldid, mdl3, "ROC")

### Lasso
lassocv <- cv.glmnet(trainPreferences, chicken, alpha = 1, foldid = foldid,
  family = "binomial", type.measure = "class")
resultsLasso(lassocv, measure = "acc")
accLasso <- 1 - min(lassocv$cvm)
plot(lassocv)

plotShrinkage(trainPreferences, chicken, lassocv)
predict(lassocv, type = "coefficients", s = lassocv$lambda.min)

### Boosting
fitControl <- trainControl(method = "cv", number = 5)
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 4, 8), n.trees = (1:10)*500, 
  n.minobsinnode = 10, shrinkage = c(0.001))

set.seed(64)
gbmFit <- train(chicken ~ ., data = trainChicken, method = "gbm", verbose = FALSE, trControl = fitControl, tuneGrid = gbmGrid)
gbmFit
plot(gbmFit)
gbmFit$results[which.max(gbmFit$results$Accuracy),1:6]
accBoost <- max(gbmFit$results$Accuracy)

## Final models
finalModels <- data.frame(acc = c(accBase, accBestSub, accLasso, accBoost), 
  row.names = c(c("Base", "BestSub", "Lasso", "Boost")))
finalModels

plotModels(finalModels$acc, xlab = "Final models", axislabels = rownames(finalModels),
  classification = TRUE)

### Final predictions
ytrue <- qualtrics[test, "chicken"]

majorityIndex <- which.max(table(trainChicken$chicken))
majorityClass <- levels(trainChicken$chicken)[majorityIndex]
sum(ytrue == majorityClass)/length(ytrue)

finalFit <- glm(mdl3, trainChicken, family = "binomial")
yhatProb <- predict(finalFit, qualtrics[test,], type = "response")
yhatClass <- ifelse(yhatProb < 0.5, "chicken", "dare")
sum(ytrue == yhatClass)/length(ytrue)

### Significance testing
qualtricsChicken <- select(qualtrics, -c(dilemma, offer, respond, public)) 
attach(qualtricsChicken)
chickenBinary <- ifelse(chicken == "chicken", 0, 1)
plotBoxes(qualtrics, chicken)
apply(preferences, 2, function(x) wilcox.test(x ~ chicken))

mdl0 <- chicken ~ 1
fit0 <- glm(mdl0, data = qualtricsChicken, family = binomial)

mdl1 <- chicken ~ altruism2
fit1 <- glm(mdl1, data = qualtricsChicken, family = binomial)
summary(fit1)

mdl2 <- chicken ~ altruism2 + trust
fit2 <- glm(mdl2, data = qualtricsChicken, family = binomial)
summary(fit2)

finalModel <- chicken ~ altruism1 + trust + risk2
finalFit <- glm(finalModel, data = qualtricsChicken, family = binomial)
summary(finalFit)

anova(fit0, fit1, fit2, finalFit, test = "Chisq")

### Diagnostics of the model

#### Non-linearity
residualPlot(finalFit, pch = 19)

#### Correlation of error terms
durbinWatsonTest(finalFit)

#### High leverage points
hlev <- hatvalues(finalFit)
plot(hlev, pch = 19)

#### Collinearity
vif(finalFit)
plotCor(data.frame(altruism2, trust, risk1))