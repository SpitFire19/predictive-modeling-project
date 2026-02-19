rm(list=objects())
library(tidyverse)
library(lubridate)
library(readr)
library(magrittr)

source('~/Documents/predictive-modeling-project/R/score.R')

setwd("~/Documents/predictive-modeling-project")
Data0 <- read_delim("~/Documents/predictive-modeling-project/Data/net-load-forecasting-during-soberty-period/train.csv", delim=",")
Data1<- read_delim("~/Documents/predictive-modeling-project/Data/net-load-forecasting-during-soberty-period/test.csv", delim=",")


range(Data0$Date)
range(Data1$Date)

Data0$Time <- as.numeric(Data0$Date)
Data1$Time <- as.numeric(Data1$Date)



sel_a <- which(Data0$Year<=2021)
sel_b <- which(Data0$Year>2021)

# plot(Data0$Date, Data0$Net_demand, type='l')
lines(Data0$Date[sel_b], Data0$Net_demand[sel_b], col='red')

###############################################################################################################################################################
#####################################################feature engineering
###############################################################################################################################################################

#############cycle hebdo
Data0$WeekDays <- as.factor(Data0$WeekDays)
Data1$WeekDays <- as.factor(Data1$WeekDays)

Data0$WeekDays2 <- weekdays(Data0$Date)
Data0$WeekDays3 <- forcats::fct_recode(Data0$WeekDays2, 'WorkDay'='Thursday' ,'WorkDay'='Tuesday', 'WorkDay' = 'Wednesday')

Data1$WeekDays2 <- weekdays(Data1$Date)
Data1$WeekDays3 <- forcats::fct_recode(Data1$WeekDays2, 'WorkDay'='Thursday' ,'WorkDay'='Tuesday', 'WorkDay' = 'Wednesday')


###################################cycle annuel: fourier
w<-2*pi/(365)
Nfourier<-50
for(i in c(1:Nfourier))
{
  assign(paste("cos", i, sep=""),cos(w*Data0$Time*i))
  assign(paste("sin", i, sep=""),sin(w*Data0$Time*i))
}
objects()
# plot(Data0$Date, cos10,type='l')
# plot(Data0$Date, cos1,type='l')

cos<-paste('cos',c(1:Nfourier),sep="",collapse=",")                         
sin<-paste('sin',c(1:Nfourier),sep="",collapse=",")
Data0<-eval(parse(text=paste("data.frame(Data0,",cos,",",sin,")",sep="")))


w<-2*pi/(365)
Nfourier<-30
for(i in c(1:Nfourier))
{
  assign(paste("cos", i, sep=""),cos(w*Data1$Time*i))
  assign(paste("sin", i, sep=""),sin(w*Data1$Time*i))
}
objects()
# plot(Data1$Date, cos1,type='l')

cos<-paste('cos',c(1:Nfourier),sep="",collapse=",")                         
sin<-paste('sin',c(1:Nfourier),sep="",collapse=",")

Data1<-eval(parse(text=paste("data.frame(Data1,",cos,",",sin,")",sep="")))


#############Temp
Data0$Temp_trunc1 <- pmax(Data0$Temp-286,0)
Data0$Temp_trunc2 <- pmax(Data0$Temp-290,0)
Data1$Temp_trunc1 <- pmax(Data1$Temp-286,0)
Data1$Temp_trunc2 <- pmax(Data1$Temp-290,0)

Data0$T1_cos1 <- Data0$Temp_trunc1 * Data0$cos1
Data0$T2_cos1 <- Data0$Temp_trunc2 * Data0$cos1

Data1$T1_cos1 <- Data1$Temp_trunc1 * Data1$cos1
Data1$T2_cos1 <- Data1$Temp_trunc2 * Data1$cos1



Nfourier<-10
lm.fourier<-list()
eq<-list()
for(i in c(1:Nfourier))
{
  cos<-paste(c('cos'),c(1:i),sep="")
  sin<-paste(c('sin'),c(1:i),sep="")
  fourier<-paste(c(cos,sin),collapse="+")
  eq[[i]]<-as.formula(paste("Net_demand~ Net_demand.1 + Net_demand.7 + WeekDays3 + Temp + Temp_trunc1 + Temp_trunc2+",fourier,sep=""))
}


###############################################################################################################################################################
#####################################################LARS
###############################################################################################################################################################
library(lars)

x <- model.matrix(eq[[10]], data = Data0)[, -1]
y <- Data0$Net_demand

l = lars(x[sel_a,], y[sel_a], type = c("lasso"))
plot(l, plottype = c("coefficients")) 

fits <- predict.lars(l, x[sel_b, ], type="fit")

rmse_lambda = apply(fits$fit, 2, rmse, y=y[sel_b])
plot(fits$s, rmse_lambda, type='l')
plot(fits$s[-c(1:5)], rmse_lambda[-c(1:5)],, type='l')

pb_lambda = apply(fits$fit, 2, pinball_loss, y=y[sel_b], quant=0.8)

plot(fits$s, pb_lambda, type='l')
plot(fits$s[-c(1:5)])

x <- model.matrix(eq[[10]], data = Data0)[, -1]
y <- Data0$Net_demand

l = lars(x[sel_a,], y[sel_a], type = c("lasso"))
plot(l, plottype = c("coefficients")) 

fits <- predict.lars(l, x[sel_b, ], type="fit")

rmse_lambda = apply(fits$fit, 2, rmse, y=y[sel_b])
plot(fits$s, rmse_lambda, type='l')
plot(fits$s[-c(1:5)], rmse_lambda[-c(1:5)],, type='l')

pb_lambda = apply(fits$fit, 2, pinball_loss, y=y[sel_b], quant=0.8)

plot(fits$s, pb_lambda, type='l')
plot(fits$s[-c(1:5)], pb_lambda[-c(1:5)],, type='l')
abline(v= which.min(pb_lambda), col='red')


fit_final <- predict.lars(l, x[sel_b, ], type="fit", s= fits$s[which.min(pb_lambda)])
coef_final <- predict.lars(l, x[sel_b, ], type="coefficients", s= fits$s[which.min(pb_lambda)])


nom = names(coef_final$coefficients)
plot(coef_final$coefficients, type='h', ylim=range(coef_final$coefficients, coef_final$coefficients*1.5))
text(c(1:length(coef_final$coefficients)), -5000, labels= nom, pos=3, srt=90, adj=1)

length(nom[which(coef_final$coefficients==0)])
nom[-which(coef_final$coefficients==0)]

cov = nom[-which(coef_final$coefficients==0)][-grep('WeekDays3', nom[-which(coef_final$coefficients==0)])]
cov = paste(cov, collapse = '+')

form <- Net_demand ~ WeekDays3+Net_demand.1+Net_demand.7+
  +Temp+Temp_trunc1+Temp_trunc2+
  +T1_cos1+T2_cos1 +
  +cos2+cos3+cos4+cos5+cos6+cos7+cos8+cos9+cos10+
  +sin2+sin3+sin4+sin5+sin6+sin7+sin8+sin9+sin10
lasso.rq <- rq(form, data = Data0[sel_a, ], tau=0.8)
summary(lasso.rq)
lasso.rq.rq.forecast <- predict(lasso.rq, newdata=Data0[sel_b,])
pb_rq <- pinball_loss(y=Data0$Net_demand[sel_b], lasso.rq.rq.forecast, quant=0.8, output.vect=FALSE)
pb_rq


lasso.rq <- rq(form, data = Data0, tau=0.8)
summary(lasso.rq)
lasso.rq.rq.forecast <- predict(lasso.rq, newdata=Data1)

submit <- read_delim( file="Data/sample_submission.csv", delim=",")
submit$Net_demand <- lasso.rq.rq.forecast
write.table(submit, file="Data/submission_lm_lasso.csv", quote=F, sep=",", dec='.',row.names = F)
plot(pb_lambda[-c(1:5)], type='l')
abline(a= which.min(pb_lambda), col='red')
abline(1, 0, col = 'red')

fit_final <- predict.lars(l, x[sel_b, ], type="fit", s= fits$s[which.min(pb_lambda)])
coef_final <- predict.lars(l, x[sel_b, ], type="coefficients", s= fits$s[which.min(pb_lambda)])
