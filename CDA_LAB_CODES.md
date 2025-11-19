# Computational Data Analytics -- Code-Only Lab Manual

### (All Experiments • Cleaned • Corrected • Markdown-Ready)

## Experiment 1 -- Introduction to R (Up to Vectors)

``` r
# 1. Simple R program
first_str <- "hello world"
first_str

# 2. Comment
str <- "hello world!"
str

# 3. Variables
var.1 = c(1, 2, 3)
var.2 <- c("lotus", "rose")
c(FALSE, 1) -> var.3
var.1
cat("var1 is", var.1, "\n")
cat("var2 is", var.2, "\n")
cat("var3 is", var.3, "\n")

# 4. Arithmetic operators
a <- c(10,20,30,40)
b <- c(2,2,3,4)
cat("sum=", (a+b), "\n")
cat("difference=", (a-b), "\n")
cat("product=", (a*b), "\n")
cat("quotient=", (a/b), "\n")
cat("remainder=", (a%%b), "\n")
cat("int division=", (a%/%b), "\n")
cat("exponent=", (a^b), "\n")
```

## Experiment 2 -- Matrices, Factors, Data Frames

``` r
M <- matrix(c(1:12), nrow=4, byrow=TRUE)
M
N <- matrix(c(1:12), nrow=4, byrow=FALSE)
N
```

## Experiment 3 -- Sampling in R

``` r
sample(1:20,10)
sample(1:6,4,replace=TRUE)
sample(1:6,4,replace=FALSE)
```

## Experiment 4 -- Central Tendency

``` r
marks <- c(97,67,89,34)
mean(marks)
median(marks)
```

## Experiment 5 -- Variability

``` r
x <- c(5,6,7,3,12,44)
range(x)
var(x)
sd(x)
```

## Experiment 6 -- Data Visualization

``` r
barplot(airquality$Ozone)
hist(airquality$Temp)
boxplot(airquality$Wind)
```

## Experiment 7 -- Power Analysis

``` python
from numpy import array
from matplotlib import pyplot as plt
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()
```

## Experiment 8 -- Date & Time

``` r
Sys.Date()
Sys.time()
```

## Experiment 9 -- Linear Regression

``` r
model <- lm(LungCap ~ Age, data=LungCapData)
summary(model)
```

## Experiment 10 -- Logistic Regression

``` r
logit_model <- glm(HighLungCap ~ Age + Height, data=LungCapData, family=binomial)
summary(logit_model)
```

## Experiment 11 -- Gradient Descent

``` r
m <- 0; b <- 0
for(i in 1:1000){
  y_pred <- m*x + b
}
```
