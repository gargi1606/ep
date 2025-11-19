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

5. Relational operator
a<-c(10,20,30,40)
b<-c(25,2,30,3)

cat(a, "less than", b, (a<b), "\n")
cat(a, "greater than", b, (a>b), "\n")
cat(a, "less than or equal to", b, (a<=b),"\n")
cat(a, "greater than or equal to", b, (a>=b), "\n")
cat(a, "equal", b, (a==b), "\n")
cat(a, "not equal to", b, (a!=b), "\n")

6. Assignment operator
#leftward assigment
var.a=c(0,20, TRUE)
var.b<-c(0,20, TRUE)
var.c<<-c(0,20, TRUE)
var.a
var.b
var.c
#rightward assignment
c(1,2, TRUE)->v1
c(1,2, TRUE)->>v2
v1
v2
```

## Experiment 2 -- Matrices, Factors, Data Frames

``` r
# Matrices
M <- matrix(c(1:12), nrow=4, byrow=TRUE)
M
N <- matrix(c(1:12), nrow=4, byrow=FALSE)
N

rnames=c("r1","r2","r3","r4")
cnames=c("c1","c2","c3")
P <- matrix(c(1:12), nrow=4, byrow=TRUE, dimnames=list(rnames,cnames))
P

# Using cbind/rbind
M = cbind(c(1,2,3), c(4,5,6))
M
N = rbind(c(1,2,3), c(4,5,6))
N

# Factors
x <- factor(c("single","married","married","single","divorced"))
x; class(x); levels(x); str(x)

# Data frame
x <- data.frame("roll"=1:2, "name"=c("Jack","Jill"), "age"=c(20,22))
x
names(x); nrow(x); ncol(x)
str(x); summary(x)

```

## Experiment 3 -- Sampling in R

``` r
sample(1:20,10)
sample(1:6,4,replace=TRUE)
sample(1:6,4,replace=FALSE)
sample(LETTERS)
data <- c(1,3,5,6,7,8,9,10,11,12,14)
sample(data, size=5)
sample(data, size=5, replace=TRUE)

df <- data.frame(
  x=c(3,4,5,6,8,12,14),
  y=c(12,6,4,23,25,8,9),
  z=c(2,7,8,8,15,17,29)
)
df

rand_df <- df[sample(nrow(df), size=3),]
rand_df

library(dplyr)
set.seed(1)
df <- data.frame(
  grade=rep(c("Freashmen","Sophomore","Junior","Senior"), each=15),
  gpa = rnorm(60, mean=85, sd=3)
)
start_sample <- df %>% group_by(grade) %>% sample_n(10)
table(start_sample$grade)

start_sample <- df %>% group_by(grade) %>% slice_sample(n=15)
table(start_sample$grade)

```

## Experiment 4 -- Central Tendency

``` r
marks <- c(97,67,89,34)
mean(marks)

median(marks)
median(c(97,67,68,89,34))

marks <- c(97,67,89,34,97)
mode <- function() { names(sort(-table(marks)))[1] }
mode()

# CSV operations
data <- data.frame(
 Product=c("TM195","TM195","TM195","TM195","TM195","TM195"),
 Age=c(18,19,19,19,20,20),
 Gender=c("Male","Male","Female","Male","Male","Female"),
 Education=c(14,15,14,12,13,14),
 MaritalStatus=c("Single","Single","Partnered","Single","Partnered","Partnered"),
 Usage=c(3,2,4,3,4,3),
 Fitness=c(4,3,4,3,2,3),
 Income=c(29562,31836,30699,28465,75643,61243),
 Miles=c(12,75,66,85,47,66)
)
write.csv(data,"data.csv",row.names=FALSE)

mydata <- read.csv("data.csv", stringsAsFactors=FALSE)
mean(mydata$Age)
median(mydata$Age)

mode <- function(){ names(sort(-table(mydata$Age)))[1] }
mode()

```

## Experiment 5 -- Variability

``` r
x <- c(5,6,7,3,12,44)
range(x)
max(x) - min(x)

sqrt(var(x))
var(x)

quantile(x, probs=0.5)
IQR(x)

print(range(mydata$Miles))
max(mydata$Miles)-min(mydata$Miles)
sqrt(var(mydata$Miles))
sd(mydata$Miles)
var(mydata$Miles)
quantile(mydata$Miles)
IQR(mydata$Miles)

```

## Experiment 6 -- Data Visualization

``` r
# Bar plots
barplot(airquality$Ozone, main='Ozone Concentration', xlab='Ozone', horiz=TRUE)
barplot(airquality$Ozone, main='Ozone Concentration', xlab='Ozone', col='blue')

# Histogram
hist(airquality$Temp,
     main="Max Temperature",
     xlab="Temp(F)",
     xlim=c(50,125),
     col="yellow",
     freq=TRUE)

# Box plot
boxplot(airquality$Wind,
        main="Wind Speed",
        xlab="MPH", ylab="Wind",
        col="orange", border="brown",
        horizontal=TRUE, notch=TRUE)

boxplot(airquality[,1:4], main="Air Quality Parameters")

plot(airquality$Ozone, airquality$Month,
     main="Scatterplot Example",
     xlab="Ozone",
     ylab="Month",
     pch=19)

# Heatmap
data <- matrix(rnorm(25,0,5), nrow=5, ncol=5)
colnames(data)=paste0("col",1:5)
rownames(data)=paste0("row",1:5)
heatmap(data)

library(maps)
map("world")

df <- data.frame(
 city=c("New York","Los Angeles","Chicago","Houston","Phoenix"),
 lat=c(40.7128,34.0522,41.8781,29.7604,33.4484),
 lng=c(-74.0060,-118.2437,-87.6298,-95.3698,-112.0740)
)
points(df$lng, df$lat, col="red")

```

## Experiment 7 -- Power Analysis

``` python
from numpy import array
from matplotlib import pyplot as plt
from statsmodels.stats.power import TTestIndPower

effect_sizes = array([0.2,0.5,0.8])
sample_sizes = array(range(5,100))

analysis = TTestIndPower()

fig, ax = plt.subplots()
analysis.plot_power(
    dep_var='nobs',
    nobs=sample_sizes,
    effect_size=effect_sizes,
    alpha=0.05,
    ax=ax
)

plt.title('Power of Two-Sample t-Test')
plt.xlabel('Sample Size per Group')
plt.ylabel('Statistical Power')
plt.legend(['Small (0.2)','Medium (0.5)','Large (0.8)'])
plt.grid(True)
plt.show()

```

## Experiment 8 -- Date & Time

``` r
x <- as.Date("2004-10-31"); x
x <- Sys.time(); x
date()
Sys.Date()
Sys.time()

library(lubridate)
now()

dates <- c("2025-08-22","2012-04-19","2017-03-05")
year(dates); month(dates); mday(dates)

my_date <- as.Date("2022-05-27")
class(my_date)
format(my_date,"%y-%h-%d")
format(my_date,"%d-%m-%y")
format(my_date,"%d-%m-%Y")
format(my_date,"%Y-%h-%d")
format(my_date,"%Y-%m-%h-%d-%H-%M-%S")

date <- ymd("2025-08-22")
update(date, year=2004, month=10, mday=12)
update(date, year=2004, month=9, mday=1)
update(date, year=2004, minut=10, seconds=20)

```

## Experiment 9 -- Linear Regression

``` r
LungCapData <- read.csv("LungCapData.csv", header=TRUE)
print(LungCapData)
names(LungCapData)

model <- lm(LungCap ~ Age, data=LungCapData)
summary(model)

plot(LungCapData$Age, LungCapData$LungCap,
     main="Age vs Lung Capacity",
     xlab="Age", ylab="LungCap",
     pch=19, col="blue")
abline(model, col="red", lwd=2)

# Multiple Linear Regression
LungCapData$Smoke <- as.factor(LungCapData$Smoke)
LungCapData$Gender <- as.factor(LungCapData$Gender)
LungCapData$Caesarean <- as.factor(LungCapData$Caesarean)

multi_model <- lm(LungCap ~ Age + Height + Smoke + Gender + Caesarean,
                   data=LungCapData)
summary(multi_model)

pred <- predict(multi_model)
plot(LungCapData$LungCap, pred,
     main="Actual vs Predicted",
     xlab="Actual", ylab="Predicted",
     pch=19, col="darkgreen")
abline(0,1,col="red",lwd=2)

```

## Experiment 10 -- Logistic Regression

``` r
LungCapData <- read.csv("LungCapData.csv", header=TRUE, stringsAsFactors=TRUE)
LungCapData$HighLungCap <- as.factor(ifelse(LungCapData$LungCap > 7, 1, 0))

LungCapData$Smoke <- as.factor(LungCapData$Smoke)
LungCapData$Gender <- as.factor(LungCapData$Gender)
LungCapData$Caesarean <- as.factor(LungCapData$Caesarean)

logit_model <- glm(
  HighLungCap ~ Age + Height + Smoke + Gender + Caesarean,
  data=LungCapData,
  family=binomial
)
summary(logit_model)

pred_prob <- predict(logit_model, type="response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

table(Predicted=pred_class, Actual=LungCapData$HighLungCap)

accuracy <- mean(pred_class == LungCapData$HighLungCap)
accuracy

```

## Experiment 11 -- Gradient Descent

``` r
set.seed(42)
n <- 100
x <- runif(n, 0, 100)
y <- 50*x + 100 + rnorm(n, 0, 10)

plot(x, y, main="Scatter Plot", xlab="x", ylab="y",
     pch=19, col="blue")

m <- 0
b <- 0
alpha <- 0.00001
iterations <- 1000
n <- length(y)

for(i in 1:iterations){
  y_pred <- m*x + b
  D_m <- (-2/n) * sum(x * (y - y_pred))
  D_b <- (-2/n) * sum(y - y_pred)
  m <- m - alpha*D_m
  b <- b - alpha*D_b
}

final_m <- m
final_b <- b

cat("Final slope:", final_m, "\nFinal intercept:", final_b, "\n")

plot(x, y,
     main="Gradient Descent Regression",
     xlab="x", ylab="y", pch=19, col="blue")
abline(a=final_b, b=final_m, col="red", lwd=2)

```
