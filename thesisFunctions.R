# Script file: "thesisFunctions.R"

## Load packages
library(psych) # (describe, corr.test, alpha)
library(corrplot) #(corrplot)
library(leaps) #(regsubsets)
library(splines) #(smooth.spline)
library(ROCR) #(prediction, performance)
library(glmnet) #(glmnet)

## plotBoxHist

# Plots a standardized boxplot and a histogram in one figure. The histogram includes
# a mean and median line. Used in the function describeVariable.

# Input: a variable called x, main, lab and ylim, breaks these are all standard graphical parameters.
plotBoxHist <- function(x, main, lab, ylim) {
  par(mfrow = c(1,2))
  boxplot(x, main = paste("Boxplot of", main), ylab = lab)
  hist(x, main = paste("Histogram of", main), xlab = lab, ylim = ylim, breaks = 30)
  abline(v = mean(x), col = "tomato1", lwd = 3) 
  abline(v = median(x), col = "skyblue", lwd = 3)
  legend('topright', legend = c("mean", "median"), lwd = 2, col= c("tomato1", "skyblue"))
}

## plotBoxBar

# Plots a standardized boxplot and a bar plot in one figure. Used in describeVariable.

# Input: a variable called x, main, lab and ylim these are all standard graphical parameters.
plotBoxBar <- function(x, main, lab, ylim) {
  par(mfrow = c(1,2))
  boxplot(x, main = paste("Boxplot of", main), ylab = lab)
  plot(as.factor(x), main = paste("Bar plot of", main), ylab = "Frequency", 
    xlab = lab, ylim = ylim)
}

## plotBar

# Plots a standardized bar plot. Used in describeVariable.

# Input: a variable called x and main.
plotBar <- function(x, main) {
  par(mfrow = c(1,1))
  plot(x, main = paste("Bar plot of", main), ylab = "Frequency",
    col = c("skyblue", "tomato1"), xlim = c(0,160), horiz = T)
}

## describeVariable

# Describes a variable and returns the appropiate descriptive statistics. 
# The function changes the description and plots based on whether the type of variable
# is a factor, discrete- or a continuous variable. In case of the latter, 
# the parameter histogram = TRUE needs to be supplied.

# Input: a variable called x, lab and ylim are the standard graphical parameters.
# hist = <boolean>, whether or not a histogram and corresponding statistics 
# should be produced.
describeVariable <- function(x, lab, ylim, hist = FALSE, main = NULL) {
  description <- list()
  if (is.null(main)) main <- deparse(substitute(x))
  else main <- main
  if (class(x) == "factor") {
    description$table <- table(x)
    description$prop.table <- round(prop.table(table(x)), 3)
    plotBar(x, main)
  }
  else if(hist) {
    description$describe <- describe(x)
    description$summary <- summary(x)
    plotBoxHist(x, main, lab, ylim)
  }
  else {
    description$describe <- describe(x) 
    description$summary <- summary(x)
    description$table <- table(as.factor(x))
    description$prop.table <- round(prop.table(table(as.factor(x))), 3)
    plotBoxBar(x, main, lab, ylim)
  }
  return(description)
}

## plotCor

# Plots a detailed correlation plot, If pvalues = FALSE, all correlations are colored
# gradiently from -1 to +1, negative relations are colored reddish and positive relationships
# are colored bleuish. If pvalues = TRUE, all significant relations are colored gradiently, 
# all other insignificant relations will have a white background. 
# Returns the correlation table and if pvalues = TRUE also the p-values table.

# Input: a dataset of quantitive variables and pvalues = <boolean>.
plotCor <- function(data, pvalues = FALSE) {
  par(mfrow=c(1,1))
  corPreferences <- cor(data, method = "spearman")
  plotColors <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
  if (pvalues) {
    pmat <- corr.test(corPreferences, method = "spearman", adjust = "none")$p
    corrplot(corPreferences, method = "color", col = plotColors(200), type = "upper",
      addCoef.col = "black", tl.col = "black", tl.srt = 45, p.mat = pmat, 
      sig.level = 0.10, insig = "blank", diag = FALSE)
    return(list(cor = round(corPreferences, 2), pvalues = round(pmat, 2)))
  }
  else {
    corrplot(corPreferences, method = "color", col = plotColors(200), type = "upper",
      addCoef.col = "black", tl.col = "black", tl.srt = 45, diag = FALSE)
    return(list(cor = round(corPreferences, 2)))
  }
}

## cbAlpha

# calculates the Cronbach's alpha score, once for the normal input and once for the 
# z-scaled variables of this input. Returns detailed scores and the the correlations 
# between the variables.

# Input: the variables which should be joined together.
cbAlpha <- function(...) {
  cb <- list()
  cb$cor <- cor(data.frame(...), method = "spearman")
  cb$alpha <- alpha(data.frame(...))
  cb$scaledAlpha <- alpha(scale(data.frame(...)))
  return(cb)
}

## plotBoxes

# Plots boxplots for all preferences in one figure, splitted by a binary factor variable.

# Input: a dataset and the y factor variable
plotBoxes <- function(data, y){
  par(mfrow = c(2,4))  
  preferences <- select(data, negRecPooled:risk2)
  for (colName in colnames(preferences)) {
    plot(y, data[[colName]], xlab = colName, 
      col = c("skyblue", "tomato1"), names = c(levels(y)[1], levels(y)[2]),
      varwidth = T)
  }
  mtext("Boxplots of preferences", outer=TRUE, cex=1, line=-2)
}

## baseModelcv

# Creates a baseline model and applies cross-validation. If a classification model is appropiate,
# then the majority class is taken as baseline and accuracy is measured.
# When a regression model is appropiate, RMSE is measured
# and the mean is used as a baseline. Returns the cross-validated score
# for every fold and the mean of these folds.

# Input: a dataset, where the first variable is a dependent variable.
# When this variable is a factor variable, majority class is taken, in all other cases
# the mean is taken. The second parameter is a foldid for every observation of the 
# dataset
baseModelcv <- function(data, foldid) {
  y <- names(data)[1]
  kfolds <- max(foldid)
  cv <- rep(NA, kfolds)
  if(class(data[[y]]) == "factor") {
    for (k in 1:kfolds) {
      ytrue <- data[foldid == k, y]
      majorityIndex <- which.max(table(data[foldid != k, y]))
      majorityClass <- levels(data[[y]])[majorityIndex]
      cv[k] <- sum(ytrue == majorityClass)/length(ytrue)
    }
    return(list(cv = cv, meancv = mean(cv), sdcv = sd(cv)))
  }
  else {
    for (k in 1:kfolds) {
      ytrue <- data[foldid == k, y]
      yhat <- mean(data[foldid != k, y])
      cv[k] <- mean((ytrue - yhat)^2)
    }
    return(list(cv = cv, meancv = mean(cv), sdcv = sd(cv), rmse = sqrt(mean(cv))))
  }
}

## bestSubsetcv

# Applies best subset selection and cross-validates the scores for every best subset
# corresponding with that number of variables. In case of a classification, 
# logistic regression is conducted and all best models are supplied as a parameter.
# These are externally found via the bestglm function. In case of a regression model,
# linear regression is used and all best models are internally found using regsubsets 
# and the R2 score. Returns a matrix with cross-validated scores for every fold with
# that number variables, for all number of variables. Also returns the standard deviation,
# the mean for every number of variables, the 'true' best number of variables and the 
# corresponding score and standard deviation.

# Input: a dataset, where the first variable is a dependent variable. 
# When this variable is a factor variable, logistic regression is applied and 
# all models should be suplied. In other cases, linear regression is used. 
# The second parameter is a foldid for every observation of the 
# dataset. The models should be supplied in case of factor variable.
bestSubsetcv <- function(data, foldid, models = NULL) {
  y <- names(data)[1]
  kfolds <- max(foldid)
  nvars <- length(data[-1])
  cv <- matrix(NA, kfolds, nvars, dimnames = list(NULL, paste(1:nvars)))
  if(class(data[[y]]) == "factor") { 
    levels(data[[y]])[1] <- "0" 
    levels(data[[y]])[2] <- "1"
    for (k in 1:kfolds) {
      i = 0
      ytrue <- data[foldid == k, y]
      for (mdl in models) {
        i = i + 1 
        fit <- glm(mdl, family = binomial, data = data[foldid != k,] )
        yhatProb <- predict(fit, data[foldid == k,], type = "response")
        yhat <- ifelse(yhatProb < 0.5, "0", "1")
        cv[k, i] <- sum(ytrue == yhat) / length(ytrue)
      }
    }
    meancv <- colMeans(cv)
    sdcv <- apply(cv, 2, sd)
    bestnmdl = which.max(unname(meancv))
    bestnmean <- max(unname(meancv))
    return(list(cv = cv, meancv = meancv, sdcv = sdcv, bestnmdl = bestnmdl,
      bestnmean = bestnmean, bestnsd = sdcv[[bestnmdl]]))
  }
  else { 
    #internal prediction function for regsubsets as object
    predict.regsubsets <- function(object, newdata, id, ...) {
      mat <- model.matrix(mdl, newdata)
      coefi <- coef(object, id = id)
      xvars <- names(coefi)
      return (mat[,xvars] %*% coefi) 
    }
    for (k in 1:kfolds) {
      ytrue <- data[foldid == k, y]
      mdl <- as.formula(paste(y, "~ ."))
      fit <- regsubsets(mdl, data = data[foldid != k,], nvmax = nvars)
      for (nvar in 1:nvars) {
        yhat <- predict(fit, data[foldid == k,], id = nvar)
        cv[k, nvar] <- mean((ytrue - yhat)^2)
      }
    }
    meancv <- colMeans(cv)
    sdcv <- apply(cv, 2, sd)
    bestnmdl = which.min(unname(meancv))
    bestnmean <- min(unname(meancv))
    return(list(cv = cv, meancv = meancv, sdcv = sdcv, bestnmdl = bestnmdl,
      bestnmean = bestnmean, bestnsd = sdcv[[bestnmdl]], bestnrmse = sqrt(bestnmean)))
  }
}

## plotR2Subs

# plots the best R2 score for every number of variables and the
# variables corresponding with these scores in one figure. Returns
# the summary of regsubset.

# Input: a regsubset model
plotR2Subs <- function(regsubset) {
  par(mfrow=c(1,2))
  plot(c(0, summary(regsubset)$rsq), xaxt = "n", xlab = "Number of variables", 
    ylab = "r2", type = "b", pch = 19, main = "R2 scores and number of variables")
  axis(1, at=1:9, labels = c("base", 1:8))
  plot(regsubset, scale = "r2", main = "R2 scores and best models")
  return(summary(regsubset))
}

## plotModels

# plots the output scores of cross-validated models and places a red dot at the
# best performing model in the plot and returns these scores.

# Input: output of the created cv functions, when needed a basemodel can be supplied.
# xlab, are the labels of the x-axis, and axislabels the labels corresponding with the ticks in the plot,
# classification = <boolean> should be supplied to place the red dot at the correct
# best model.
plotModels <- function(cvscores, basemodel = NULL, xlab = 'number of variables', axislabels = c("base", 1:(length(scores)-1)), classification = FALSE) {
  par(mfrow=c(1,1))
  if (!(is.null(basemodel))) {
    scores <- c(basemodel$meancv, cvscores)
    names(scores)[1] <- "base"
    plot(scores, type = "b", xaxt = "n", xlab= xlab, ylab = "cv MSE", pch = 19,
         main = "Performance of models")
    axis(1, at=1:length(scores), labels = axislabels)
    if(classification) {points(which.max(scores), max(scores), col = "red", pch = 19)}
    else{points(which.min(scores), min(scores), col = "red", pch = 19)}
  }
  else {
    scores <- cvscores
    plot(scores, type = "b", xaxt = "n", xlab= xlab, ylab = "cv MSE", pch = 19,
         main = "Performance of models")
    axis(1, at=1:length(scores), labels = axislabels)
    if(classification) {points(which.max(scores), max(scores), col = "red", pch = 19)}
    else{points(which.min(scores), min(scores), col = "red", pch = 19)}
  }
  return(scores)
}

## plotThreshold

# Plots an Accuracy- and F-score- vs. threshold plot in one figure. 
# If plotcontent = "ROC", a ROC curve is plotted and the AUC is returned.

# Input: a dataset and corresponding with a foldid for every observation of the 
# dataset. Furthermore the formula of the bestmodel should be supplied and the plotContent.
plotThreshold <- function(data, foldid, bestmodel, plotcontent = "measure") {
  y <- names(data)[1]
  kfolds <- max(foldid)
  preds <- as.numeric()
  ytrue <- as.numeric()
  for (k in 1:kfolds) {
    fit <- glm(bestmodel, family = binomial, data = data[foldid != k,])
    preds <- c(preds, predict(fit, data[foldid == k,], type = "response"))
    ytrue <- c(ytrue, data[foldid == k, y])
  }
  pred <- prediction(preds, ytrue)
  if (plotcontent == "measure") {
    par(mfrow=c(1,2))
    for (measure in c("acc", "f")) {
      plot(performance(pred, measure = measure))
      abline(v = 0.5, col = "red", lwd = 2, lty = 2)
    }
    mtext("Threshold plots", outer=TRUE,  cex=1.2, line=-2)
  }
  else if(plotcontent == "ROC") {
    par(mfrow=c(1,1))
    plot(performance(pred, measure = "tpr", x.measure = "fpr"), main = "ROC plot")
    abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
    auc <- performance(pred, "auc")
    return(unlist(slot(auc, 'y.values')))
  }
}

## plotShrinkage

# Generates a glmnet model and then plots the shrinkage the coefficients.
# The red dotted line corresponds with the best lambda and a blue dotted line 
# corresponds with the best lambda plus one standard error.

# Input: a model.matrix of preferences, the y variable and the output of cv.glmnet
plotShrinkage <- function(preferences, y, lassocv) {
  par(mfrow=c(1,1))
  if(class(y) == "factor") lasso <- glmnet(preferences, y, alpha = 1, family = "binomial")
  else lasso <- glmnet(preferences, y, alpha = 1)
  #plot coefficients and lambda's.
  plot(lasso, xvar = "lambda", label = T)
  abline(v = log(lassocv$lambda.min), col= "tomato1", lty = 2, lwd = 2)
  abline(v = log(lassocv$lambda.1se), col = "skyblue", lty = 2, lwd = 2) 
}

## resultsLasso

# Returns the results of the lassocv in a standardized way as 
# in the other created cv functions. Returns the 'true' best lambda
# and the corresponding score and standard deviation and the best
# lambda plus one standard error. Takes a meausure input, and adjusts
# the output in case of a general linear model.

# Input: outout of glmnet.cv function and what kind of measure is needed = "acc" is standard.
# and corresponds with a general linear model
resultsLasso <- function(lassocv, measure = NULL) {
  if (is.null(measure)) {
     index <- which.min(lassocv$cvm)
    bestnmean <- min(lassocv$cvm)
    list(glmnet.fit = lassocv$glmnet.fit, bestlambda = lassocv$lambda.min, 
      bestnmean = bestnmean, bestnsd = lassocv$cvsd[index], bestnrmse = sqrt(bestnmean),
      lambda1se = lassocv$lambda.1se)
  }
  else if (measure == "acc") {
    index <- which.min(lassocv$cvm)
    bestnmean <- 1 - min(lassocv$cvm)
    list(glmnet.fit = lassocv$glmnet.fit, bestlambda = lassocv$lambda.min, 
      bestnmean = bestnmean, bestnsd = lassocv$cvsd[index], lambda1se = lassocv$lambda.1se)
  }
}

## plotBivariate

# plots a biviarate plot with a x variable and y variable and draws 
# the general linear model or linear model best fit line. 
# The plot and fit is adjusted if y == binary variable.

# Input: binary variable if y is a qualititative variable else a regular dependent variable
# the x variable should also be supplied.
plotBivariate <- function(x, y) {
  if (max(y) == 1) {
    plot(jitter(x), jitter(y), pch = 19, 
     col = ifelse(y == 0, "skyblue", "tomato1"), main = "Bivariate plot")
    fit1 <- glm(y ~ x, family = binomial)
    preds <- predict(fit1, newdata = list(x = 0:max(x)), type = "response")
    lines(0:max(x), preds, lwd = 2, col = "black")
    abline(h = 0.5, col = "black", lty = 2)
  }
  else {
    plot(jitter(x), jitter(y), pch = 19, main = "Bivariate plot",
      xlab = deparse(substitute(x)), ylab = deparse(substitute(y)))
    abline(h = mean(y), lwd = 2)
    fit <- lm(y ~ x)
    abline(fit, lwd = 2, col = "red")
  }
}

## nonLinearcv

# Cross-validation is applied on smoothing splines for a range of supplied degrees
# of freedom from 2 to until a supplied parameter of maxdegree. Returns the 
# standardized output as in bestsubsetcv, but with the degrees of freedom included.

# Input: a dataset, where the first variable is a dependent variable. 
# The second parameter is a foldid for every observation of the 
# dataset. The models should be supplied.
nonLinearcv <- function(data, foldid, models) {
  y <- names(data)[1]
  kfolds <- max(foldid)
  cv <- matrix(NA, kfolds, length(models), dimnames = list(NULL, paste(1:length(models))))
  for (k in 1:kfolds) {
    i = 0
    for (mdl in models) {
      i = i + 1
      fit <- lm(mdl, data = data[foldid != k,])
      yhat <- predict(fit, data[foldid == k,])
      ytrue <- data[foldid == k, y]
      cv[k, i] <- mean((ytrue - yhat)^2)
    }
  }
  meancv <- colMeans(cv)
  sdcv <- apply(cv, 2, sd)
  bestnmdl = which.min(unname(meancv))
  bestnmean <- min(unname(meancv))
  return(list(cv = cv, meancv = meancv, sdcv = sdcv, bestnmdl = bestnmdl,
    bestnmean = bestnmean, bestnsd = sdcv[[bestnmdl]], bestnrmse = sqrt(bestnmean)))
}

## smoothSplinecv

# cross-validation is applied on smoothing splines for a range of supplied degrees 
# of freedom from 2 until a supplied parameter of maxdegree. Returns the standardized 
# output as in bestsubsetcv, but with the degrees of freedom included.

# Input: a dataset, where the first variable is a dependent variable.
# The second parameter is a foldid for every observation of the 
# dataset. X is the independent variable and maxdegree to what degree the spline should be conducted.
smoothSplinecv <- function(data, foldid, x, maxdegree) {
  y <- names(data)[1]
  kfolds <- max(foldid)
  cv <- matrix(NA, kfolds, maxdegree - 1, dimnames = list(NULL, paste(2:(maxdegree))))
  for (k in 1:kfolds) {
    for (freedom in seq(from = 2, to = maxdegree, by = 1)){
      fit <- smooth.spline(data[foldid != k, x], data[foldid != k, y], df = freedom)
      yhat <- predict(fit, data[foldid == k, x])$y
      ytrue <- data[foldid == k, y]
      cv[k, freedom-1] <- mean((ytrue - yhat)^2)
    } 
  }
  meancv <- colMeans(cv)
  sdcv <- apply(cv, 2, sd)
  bestnmdl = which.min(unname(meancv))
  bestnmean <- min(unname(meancv))
  return(list(cv = cv, meancv = meancv, sdcv = sdcv, bestnmdl = bestnmdl,
    bestdf = bestnmdl + 1, bestnmean = bestnmean, bestnsd = sdcv[[bestnmdl]], 
    bestnrmse = sqrt(bestnmean)) )
}