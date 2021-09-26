##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

Sys.setlocale("LC_ALL", "English")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Data exploration
edx%>%sample_n(5)

#Unique users and Movies
edx%>%summarize(Users=length(unique(userId)),
                Movies=length(unique(movieId)))

#Most rated movies are
edx%>%group_by(title)%>%
  summarize(N_ratings=n(),Avg_rating=mean(rating))%>%arrange(desc(n_ratings))%>%
  top_n(10,n_ratings)


# Models creation
# Create a function to calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#Split edx by test and training set
set.seed(755)
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE) #split by training and test sets
train_edx <- edx[-test_index,]
test_edx <- edx[test_index,]

test_edx <- test_edx %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId") 



# Simple Average model
mu<-mean(train_edx$rating)
simple_average<-RMSE(test_edx$rating,mu)
print(simple_average) #RMSE >1 meaning that our prediction may differ by 1 start rating.


#Movie effect - How movies differ around average
movie_avg<-train_edx%>%group_by(movieId,title)%>%summarize(b_i=mean(rating-mu),num=n())
movie_avg%>%ggplot(aes(b_i))+geom_histogram(bins = 20, color = "black",fill="grey")+ggtitle("Histogram of Movie effect")+
  ylab("Frequency")
#we can see that there only a few movies with 5 start rating (mu+1.5) and many movies below the average

model_with_movie_effect_prediction<-mu+test_edx%>%
  left_join(movie_avg, by='movieId') %>%
  pull(b_i)

model_with_movie_effect_RMSE<-RMSE(test_edx$rating,model_with_movie_effect_prediction)
print(model_with_movie_effect_RMSE) #RMSE is getting better but what about users ? do they differ as well?


#User Effect - to find how people differ from each other if we exclude average and movie effect
user_avg<-train_edx%>%left_join(movie_avg, by='movieId')%>%
  group_by(userId)%>%summarize(b_u=mean(rating-mu-b_i))

user_avg%>%ggplot(aes(b_u))+geom_histogram(bins = 30, color = "black")+ggtitle("Histogram of User effect")+
  ylab("Frequency")
#Users average preference is normal distributed around average and we can see a tail of pessimistic viewers
#and a tail of very optimistic viewers

model_with_movie_user_effect_prediction<-mu+test_edx%>%
  left_join(movie_avg, by='movieId') %>%pull(b_i)+test_edx%>%
  left_join(user_avg, by='userId') %>%pull(b_u)

model_with_movie_user_effect_RMSE<-RMSE(test_edx$rating,model_with_movie_user_effect_prediction)
print(model_with_movie_user_effect_RMSE)
#RMSE is getting even lower 


#Let's have a look into highest discrepancies
#we can see that highest discrepancies happen due to low # of ratings (<2^5)
test_edx%>%left_join(movie_avg, by='movieId')%>%select(-title.y)%>%
  left_join(user_avg, by='userId')%>%
  mutate(movie_user_effect_prediction=mu+b_i+b_u)%>%
  mutate(residual=(rating-movie_user_effect_prediction))%>%
  group_by(title.x)%>%summarize(n_ratings=n(),avg_residual=mean(residual))%>%
  ggplot(aes(log2(n_ratings),avg_residual))+geom_point()+ggtitle("Average residuals of movies grouped by # of Ratings")+
  ylab("Average residuals")+
  xlab("Log2 # of Ratings")


#To eliminate this noise and improve RMSE we need to penalize for large estimates which comes from small sample sizes
lambdas <- seq(0, 10, 0.1)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_edx$rating)
  
  b_i <- train_edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l)) #+ Movie effect  and LAMBDA
  
  b_u <- train_edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l)) #+ User effect and LAMBDA
  
  predicted_ratings <- 
    test_edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% 
    pull(pred)
  
  return(RMSE(predicted_ratings, test_edx$rating))
})

qplot(lambdas, rmses)  
#we'll choose the best lambda for our Regularized model
best_lambda<-lambdas[which.min(rmses)]
#best lambda is just 4.9




# The best lambda=4.9 and we will use it as part of our regularized Movie and User effects model.  
b_i_regularized <- train_edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+best_lambda)) #+ Movie effect  and best LAMBDA

b_u_regularized <- train_edx %>% 
  left_join(b_i_regularized, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+best_lambda)) #+ User effect and best LAMBDA

predicted_ratings <- 
  test_edx %>% 
  left_join(b_i_regularized, by = "movieId") %>%
  left_join(b_u_regularized, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)

regularized_model_prediction<-RMSE(predicted_ratings, test_edx$rating)
print(regularized_model_prediction) 


#We can see that RMSE hasn't changed much because the % ratings for not popular movies is very low 
#in comparison the total population of ratings (< 1%).
total_ratings<-train_edx%>%summarize(n=n())%>%.$n

train_edx %>% 
  group_by(movieId) %>% summarize(n_ratings=n())%>%arrange(n_ratings)%>%
  filter(n_ratings<=2^5)%>%summarize(Percent_ratings_of_not_popular_movies=sum(n_ratings)/total_ratings)

#Even though these movies create a big clusters out of total number of movies (30%).

train_edx %>% 
  group_by(movieId) %>% summarize(n_ratings=n())%>%arrange(n_ratings)%>%
  mutate(log_n=log(n_ratings))%>%
  ggplot(aes(log_n))+geom_histogram(aes(fill=(log_n<5)),bins=30, color = "black")+
  scale_fill_manual(values = c("grey", "orange"))+
  ylab("# Movies")+
  xlab("Log2 # of Ratings")+
  ggtitle("Distribution of Movies by # of Ratings")

train_edx %>% 
  group_by(movieId) %>% summarize(n_ratings=n())%>%arrange(n_ratings)%>%
  summarize(Percent_movies_with_ratings_less_than_32=sum(n_ratings<=2^5)/n())



#The graph below will show how movie effect has been minimized for movies with low number of ratings
tibble(original=movie_avg$b_i,regularized=b_i_regularized$b_i)%>%
  ggplot(aes(original, regularized)) + 
  geom_point(alpha=0.5)+
  xlab("Movie effect")+
  ylab("Regularized Movie effect")



#To compare distribution of Residuals we need to build density plot
residuals_movie_model<-test_edx%>%
  left_join(movie_avg, by='movieId')%>%
  mutate(pred=mu+b_i)%>%
  mutate(residual_movie=rating-pred)%>%.$residual_movie

residuals_movie_user_model<-test_edx%>%
  left_join(movie_avg, by='movieId')%>%
  left_join(user_avg, by='userId')%>%
  mutate(pred=mu+b_i+b_u)%>%
  mutate(residual_movie_user=rating-pred)%>%.$residual_movie_user

residuals_Finalmodel<-test_edx%>%
  left_join(b_i_regularized, by='movieId')%>%
  left_join(b_u_regularized, by='userId')%>%
  mutate(pred=mu+b_i+b_u)%>%mutate(pred=ifelse(pred>5,5,pred))%>% #estimation can't be more than 5
  mutate(pred=ifelse(pred<0,0,pred))%>% #estimation can't be less than 0
  mutate(residual_regular=rating-pred)%>%.$residual_regular

res <- data.frame(movie=residuals_movie_model,movie_user=residuals_movie_user_model,regularised=residuals_Finalmodel)
res <-gather(res,model)
ggplot(res,aes(x=value, fill=model)) + geom_density(alpha=0.35)

#qqplot to compare residuals distribution to normal distribution

qqnorm(residuals_Finalmodel)
qqline(residuals_Finalmodel)

#Plot shows as a big tail where residuals <-1, overestimation, far from normal


#Are there any time effect left? week number?
library(scales)
train_edx%>%
  left_join(b_i_regularized, by='movieId')%>%
  left_join(b_u_regularized, by='userId')%>%
  mutate(pred=mu+b_i+b_u)%>%
  mutate(residual_regular=(rating-pred))%>%
  mutate(date_rate=as_datetime(timestamp))%>%mutate(week=round_date(date_rate,"week"))%>%
  group_by(week)%>%summarize(avg_res=mean(residual_regular))%>%
  ggplot(aes(week,avg_res))+geom_point()+geom_smooth()+
  scale_x_datetime(breaks = date_breaks("3 years"),labels = date_format("%Y"))

#There are much higher residuals earlier 1997 (it means ratings are better than we estimated)
#it looks more like an outliers and we need to filter these dates out from our model
#After this dates week doesn't really effect residuals


#Do we have effect of rating per week day (Monday-Sunday?) Do people like movies more when you're more relaxed on weekends?

train_edx%>%
  left_join(b_i_regularized, by='movieId')%>%
  left_join(b_u_regularized, by='userId')%>%
  mutate(date_rate=as_datetime(timestamp))%>%mutate(week=round_date(date_rate,"week"))%>%
  filter(week>"1997-01-01")%>%
  mutate(day_of_week=weekdays(date_rate))%>%
  mutate(pred = mu + b_i + b_u)%>%
  mutate(residual_regular=rating-pred)%>%
  ggplot(aes(x=day_of_week,y=residual_regular))+geom_boxplot()
  
#No such effect of day of week


#Let's create training table excluding ratings earlier 1997 and call it train_edx_wide
train_edx_wide<-train_edx%>%
  left_join(b_i_regularized, by='movieId')%>%
  left_join(b_u_regularized, by='userId')%>%
  mutate(date_rate=as_datetime(timestamp))%>%mutate(week=round_date(date_rate,"week"))%>%
  filter(week>"1997-01-01")%>%
  mutate(pred = mu + b_i + b_u)%>%
  mutate(residual_regular=rating-pred)

#########################################################################################

#PCA - Principal Component Analysis
# Let's have a look into movie with highest number of residuals < (-1)
total_negative<-train_edx_wide%>%
  filter(residual_regular<(-1))%>%summarize(n_negative=n())

train_edx_wide%>%
  filter(residual_regular<(-1))%>%
  group_by(movieId)%>%summarize(n_negative=n())%>%arrange(desc(n_negative))%>%
  mutate(cumul_sum=cumsum(n_negative)/total_negative)%>%summarize(top_movies_50_percent=sum(cumul_sum<=0.5))

#We can see that 533 movies will have 50% of all negative residuals
#We will choose them to improve model and create PCA as there must be difference in movies preferences

high_res_by_movie<-train_edx_wide%>%
  filter(residual_regular<(-1))%>%
  group_by(movieId)%>%summarize(n_negative=n())%>%
  top_n(533,n_negative)%>%
  .$movieId

#Let's prepare matrix for PCA model (User x Movie)
x <- train_edx_wide %>% 
  filter(movieId %in% high_res_by_movie) %>%
  group_by(userId) %>%
  filter(n() >= 20) %>% #Assume that 20 ratings per user is enough to understand preferences
  ungroup() %>% 
  select(title, userId, rating) %>%
  pivot_wider(names_from = "title", values_from = "rating") %>%
  as.matrix()

rownames(x)<- x[,1]
x <- x[,-1]

x <- sweep(x, 2, colMeans(x, na.rm = TRUE))
x <- sweep(x, 1, rowMeans(x, na.rm = TRUE))

x[is.na(x)] <- 0  #replace NA with O for residuals
pca <- prcomp(x)

summary(pca)$importance[,1:25] # first 25 components
plot(summary(pca)$importance[3,])
dim(pca$rotation)

pcs_matrix<-data.frame(pca$rotation, title=colnames(x))

library(ggrepel)
ggplot(pcs_matrix,aes(PC1,PC2))+
geom_point(cex=3, pch=21) +
coord_fixed(ratio = 1)+
geom_label_repel(data=pcs_matrix%>%filter(PC2<(-0.12) | PC2>0.12 | PC1 >0.12 | PC1<(-0.12)),
                 aes(label=substr(title,0,21)),
                 box.padding = 0.5, max.overlaps = 10)

#PC2: Movie for the whole family vs movie for Adults
#PC1: Art movies vs Blockbusters

ggplot(pcs_matrix,aes(PC3,PC4))+
geom_point(cex=3, pch=21) +
coord_fixed(ratio = 1)+
geom_label_repel(data=pcs_matrix%>%filter(PC3 <(-0.15) | PC4>(0.1)| PC4<(-0.05)),aes(label=substr(title,0,21)),
                 box.padding = 0.5, max.overlaps = 10)
#PC3: Nerd movies/Sci-Fi vs others
#PC4: Serious Oscar movies vs Comedy movies


library(caret)
library(tidyverse)
library(tidyr)
library(textshape)

#Now let's evaluate how many PCs we need to take to minimize RMSE. We will tune PC number below.
PC_n<-seq(2, 150, 10)

rmses_pca <- sapply(PC_n, function(p){
  
  p_q_matrix<-pca$x[,1:p]%*%t(pca$rotation[,1:p])
  colnames(p_q_matrix)<-rownames(pca$rotation)
  
  p_q_matrix<-tidy_matrix(p_q_matrix, row.name ="userId", col.name = "title",value.name="residual_estimate")
  p_q_matrix$userId<-as.integer(p_q_matrix$userId)
  
  prediction_PCA<-test_edx%>%
    left_join(b_i_regularized, by='movieId')%>%
    left_join(b_u_regularized, by='userId')%>%
    left_join(p_q_matrix, by=c('userId','title'))%>%
    mutate(residual_estimate=replace(residual_estimate,is.na(residual_estimate),0))%>%
    mutate(pred=mu+b_i+b_u+residual_estimate)%>%.$pred
  
  return(RMSE(prediction_PCA, test_edx$rating))
  
})

qplot(PC_n, rmses_pca,label=round(rmses_pca,3),geom=c("point", "text"),vjust=2)+
  labs(x="PC number",y="RMSE of PCA model")

best_PC_number<-PC_n[which.min(rmses_pca)]

#From the graph above we can find that RMSE is minimal when we use 32 principal components.

#We are going to use these 32 principal components for our model which are expected to improve the model by extra 0.01.  
#With code below we calculate $p*q$ matrix of estimated residuals.

p<-32
#Final p*q matrix of estimated residuals
p_q_matrix<-pca$x[,1:p]%*%t(pca$rotation[,1:p])
colnames(p_q_matrix)<-rownames(pca$rotation)

p_q_matrix<-tidy_matrix(p_q_matrix, row.name ="userId", col.name = "title",value.name="residual_estimate")
p_q_matrix$userId<-as.integer(p_q_matrix$userId)


residuals_PCA<-test_edx%>%
  left_join(b_i_regularized, by='movieId')%>%
  left_join(b_u_regularized, by='userId')%>%
  left_join(p_q_matrix, by=c('userId','title'))%>%
  mutate(residual_estimate=replace(residual_estimate,is.na(residual_estimate),0))%>%
  mutate(pred=mu+b_i+b_u+residual_estimate)%>%mutate(if_else(pred>5,5,if_else(pred<0,0,pred)))%>%mutate(residual_PCA=rating-pred)%>%.$residual_PCA

res <- data.frame(movie=residuals_movie_model,movie_user=residuals_movie_user_model,PCA=residuals_PCA)
res <-gather(res,model)
ggplot(res,aes(x=value, fill=model)) + geom_density(alpha=0.35)

qqnorm(residuals_PCA)
qqline(residuals_PCA)

#There is still a tail with overestimated ratings which connected to movie and users which didn't become part of PCA models
#(movies or users with low # ratings)

prediction_PCA<-test_edx%>%
  left_join(b_i_regularized, by='movieId')%>%
  left_join(b_u_regularized, by='userId')%>%
  left_join(p_q_matrix, by=c('userId','title'))%>%
  mutate(residual_estimate=replace(residual_estimate,is.na(residual_estimate),0))%>%
  mutate(pred=mu+b_i+b_u+residual_estimate)%>%mutate(if_else(pred>5,5,if_else(pred<0,0,pred)))%>%.$pred

prediction_PCA_RMSE<-RMSE(prediction_PCA, test_edx$rating)
print(prediction_PCA_RMSE) 
#Prediction improved but slightly, however, still better than regularized model


#Results-the best is PCA model
t(data.frame(simple_average,model_with_movie_effect_RMSE,model_with_movie_user_effect_RMSE,regularized_model_prediction,prediction_PCA_RMSE))

##########################################################
#Final model application to the Validation dataset
prediction_Final<-validation%>%
  left_join(b_i_regularized, by='movieId')%>%
  left_join(b_u_regularized, by='userId')%>%
  left_join(p_q_matrix, by=c('userId','title'))%>%
  mutate(residual_estimate=replace(residual_estimate,is.na(residual_estimate),0),b_i=replace(b_i,is.na(b_i),0),b_u=replace(b_u,is.na(b_u),0))%>%
  mutate(pred=mu+b_i+b_u+residual_estimate)%>%.$pred

prediction_Final_RMSE<-RMSE(prediction_Final,validation$rating)
print(prediction_Final_RMSE)