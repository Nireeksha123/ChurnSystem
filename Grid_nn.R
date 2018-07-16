library(h2o)
library(data.table)
h2o.init(nthreads=-1,enable_assertions=FALSE,min_mem_size='2g',max_mem_size='4g')
input <- h2o.importFile("C:\\Users\\m.nirreeksha\\Desktop\\data\\iris.csv",sep=",")
input$Species <- as.factor(input$Species)
split_data <- h2o.splitFrame(input, ratios = 0.65, seed = 1234, 
                             destination_frames = c("train.hex" , "test.hex"))
train <- split_data[[1]]
test <- split_data[[2]]
y <- "Species"
x <- colnames(train)[c(2:5)]
hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)

search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 28800, max_models = 50, stopping_rounds=5, stopping_tolerance=1e-2)
dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=train,
  x=x, 
  y=y,
  nfolds = 3,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE,
  epochs=1,
  stopping_metric="logloss",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="logloss",decreasing=FALSE)
#grid <- h2o.getGrid"dl_grid_random",sort_by = "auc",decreasing=TRUE)
#grid <- h2o.getGrid("dl_grid_random",sort_by="err",decreasing=FALSE)
grid
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model
model_para <- best_model@allparameters
model_para
model<-h2o.saveModel(object=best_model,path=getwd(),force=FALSE)
print(model)
