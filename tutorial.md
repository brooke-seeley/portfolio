---
title: "Tutorial Blog - Cleaning and Predicting from Data in R"
---

## My Predictive Experience

Last semester, I completed a Predictive Analytics course, which had a focus on machine learning practices in R. In this course, we practiced our skills in multiple Kaggle competitions. My final project of the course was with the [Kobe Bryant Shot Selection](https://www.kaggle.com/competitions/kobe-bryant-shot-selection) data, where we are given many factors and are asked to predict whether or not Kobe made an attempted shot. Before I was able to predict on this data, I had to do quite a bit of feature engineering. I also had to try various models before reaching a low log-loss. I decided to frame my work as a data cleaning and predition tutorial in this blog. In it, you will find the following:

1. Initial Preparation
    - Necessary Packages
    - Read in Data
    - Split into Train and Test Data
2. First Recipe
    - New Features
    - Factor Conversion
    - Removed Features
3. Preliminary Models
    - Penalized Logistic Regression
    - Random Forest
    - K-Nearest Neighbors
4. New Recipe
    - Create New Features
    - Remove Other Features
5. Final Model

These steps and models can be applied to many different datasets. Feel free to try this with the Kobe dataset, or your own! How low can you get the log-loss?

## Step 1: Initial Preparation

### Necessary Packages

```r
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
```

These 4 packages are key for machine learning and predictive analytics in R.

-**tidyverse**: the ultimate R package, especially for tidying and manipulating
-**tidymodels**: makes it easier to create multiple models by using recipes
-**vroom**: quickly reads in and writes datasets, especially when in .csv format
-**embed**: encodes categorical variables in data recipes

### Read in Data

```r
data <- vroom('data.csv')
```

`vroom()` reads in this large dataset super easily, as opposed to a function like `read.csv()` which can be more computationally expensive.

### Split into Train and Test Data

As opposed to many other Kaggle competitions, the train and test data came in one .csv file. So, we will need to split it ourselves. When looking at the data, there is a `shot_made_flag` variable, which is what we are predicting. It has a `0` if the shot wasn't made, and a `1` if it was. Some of the rows have `NA` values, telling us that these are the ones we are supposed to predict.

We can use the R functions `drop_na()` to give us the train data, and `is.na()` for the test data:

```r
trainData <- data %>%
  drop_na(shot_made_flag) %>%
  mutate(shot_made_flag = factor(shot_made_flag))

testData <- data %>%
  filter(is.na(shot_made_flag)) %>%
  select(-shot_made_flag)
```

My goal for the assignment was to get a log-loss (the score for this Kaggle competition) close to or lower than 0.6. Let's see if we can do that!

## Step 2: First Recipe

This `tidymodels` recipe was my first draft. Let's discuss each line of it. Then, think of what you would add or remove to improve it. Click on the link above to the Kaggle competition to get a preview of the data, specifically its variables.

```r
target_recipe <- recipe(shot_made_flag ~ ., data = trainData) %>%
  step_date(game_date, features="month") %>%
  step_date(game_date, features="year") %>%
  step_mutate(game_event_id = factor(game_event_id)) %>%
  step_mutate(period = factor(period)) %>%
  step_mutate(playoffs = factor(playoffs)) %>%
  step_mutate(game_date_month = factor(game_date_month)) %>%
  step_mutate(game_date_year = factor(game_date_year)) %>%
  step_rm(shot_id, team_name, team_id, matchup, game_id, game_date) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_mixed(all_factor_predictors(),
                     outcome = vars(shot_made_flag)) %>%
  step_normalize(all_factor_predictors())
```

1. As discussed above, `shot_made_flag` is our response variable and what we are predicting based on various factors.
2. There was a `game_date` variable, which a lot could be done with. I added a `month` feature to see if the time of year during the season affected Kobe's accuracy.
3. I also added a `year` feature to observe his improvement over his career.
4. `game_event_id` gave a number for what number shot it was in the game. I made this a factor, would you have done the same?
5. The NBA has 4 quarters, and shot accuracy may change based on the quarter. The dataset had the `period` listed as simply 1-4, so I converted that to a factor.
6. If it was the playoffs, the value for `playoffs` was `1`, and `0` if not. I made that a factor as well.
7. I made the above `game_date_month` variable I created into a factor.
8. I did the same for `game_date_year`.
9. I removed the following variables for these reasons:
    - `shot_id` - just identified which shot it was, not helpful for predicting
    - `team_name` and `team_id` - this was for his own team, and Kobe was a Lakers lifer
    - `matchup` and `game_id` - these became redundant thanks to other variables, such as `opponent`
    - `game_date` - we already created `game_date_month` and `game_date_year` which explain more
10. With target encoding, I wanted to combine the variables of very small amounts so it didn't become an overfit model.
11. This line is code for target encoding of variables.
12. I normalized the factor predictors to help with the fit.

## Step 3: Preliminary Models

### Model 1: Penalized Logistic Regression

I first started with a simpler model, Penalized Logistic Regression. I used a `workflow` function to make the modeling easier, and ran a cross-validation to tune the best values for `mixture` and `penalty`. I used `brier_class` as the metric, since that is the definer for log-loss. Note that to use this model, you will also need the `glmnet` library.

Here is the code:

```r
library(glmnet)

preg_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
  set_engine("glmnet")

preg_workflow <- workflow() %>%
  add_recipe(target_recipe) %>%
  add_model(preg_mod)

### Grid of values to tune over

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

### Split data for CV

folds <- vfold_cv(trainData, v = 5, repeats = 1)

### Run the CV

CV_results <- preg_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics(metric_set(brier_class)))

### Find Best Tuning Parameters

bestTune <- CV_results %>%
  select_best(metric="brier_class")

### Finalize the Workflow & fit it

final_wf <-
  preg_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

### Predict

pen_reg_predictions <- final_wf %>%
  predict(new_data = testData, type="prob")

### Kaggle

pen_reg_kaggle_submission <- pen_reg_predictions %>%
  bind_cols(., testData) %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag=.pred_1)

vroom_write(x=pen_reg_kaggle_submission, file="./PenRegPreds.csv", delim=',')
```

Note the "Kaggle" section of the code. This was to get the predictions into the proper format for Kaggle scoring. You need to do something similar for every Kaggle competition. That is what the sample submission document is for, to give you an example of what your submission should look like! I recommend naming the files very specifically to make it easier to see what model gave you what Kaggle score.

This gave me a log-loss of 0.61244, which is quite good! Perhaps with your seed, it could be lower or higher. Let's see if we can take that down a notch.

### Model 2: Random Forest

Next is a popular model for binary or categorical predictions, which is a Random Forest model. This creates a bunch of regression trees for each row and averages their results. You'll see that this workflow looks very similar to the one for the last model, and is a standard ML/Predictive Analytics workflow. 

Note that I set the `trees` myself. You can tune for them, but it becomes very computationally expensive. This time, we are tuning for `mtry`, the number of features considered at each node split, and `min_n`, the number of datapoints in a node for it to split. With the `rand_forest` function, I set the mode as `"classification"` so it will give us a `1` or `0` for the shot rather than a probability.

```r
library(rpart)

tree_mod <- rand_forest(mtry=tune(),
                        min_n=tune(),
                        trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

tree_workflow <- workflow() %>%
  add_recipe(target_recipe) %>%
  add_model(tree_mod)

## Grid of values to tune over

tuning_grid <- grid_regular(mtry(range=c(1,20)),
                            min_n(),
                            levels=5)

### CV

folds <- vfold_cv(trainData, v = 5, repeats = 1)

CV_results <- tree_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=(metric_set(brier_class)))

### Find best tuning parameters

bestTune <- CV_results %>%
  select_best(metric="brier_class")

### Finalize workflow

final_wf <-
  tree_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

### Predict

tree_predictions <- final_wf %>%
  predict(new_data = testData, type="prob")

### Kaggle

tree_kaggle_submission <- tree_predictions %>%
  bind_cols(., testData) %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag=.pred_1)

vroom_write(x=tree_kaggle_submission, file="./RegTreePreds.csv", delim=',')
```

Here, I got a log-loss of 0.61162, which is a slight improvement on Penalized Regression. How will our last model do?

### Model 3: K-Nearest Neighbors

Finally, I tried K-Nearest Neighbors, which is specifically a ML algorithm. It makes predictions based on a majority vote of the training data points around our test points. The number of points it looks at is decided by the `neighbors` argument, which is what we will tune.

```r
library(kknn)

knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_workflow <- workflow() %>%
  add_recipe(target_recipe) %>%
  add_model(knn_model)

### Tuning Parameters

tuning_grid <- grid_regular(neighbors())

### CV

folds <- vfold_cv(trainData, v = 5, repeats = 1)

CV_results <- knn_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics(metric_set(brier_class)))

### Find best K

bestTune <- CV_results %>%
  select_best(metric="brier_class")

### Finalize Workflow

final_wf <-
  knn_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainData)

### Predict

knn_predictions <- final_wf %>%
  predict(knn_workflow, new_data=testData, type="prob")

### Kaggle

knn_kaggle_submission <- knn_predictions %>%
  bind_cols(., testData) %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag=.pred_1)

vroom_write(x=knn_kaggle_submission, file="./KNNTreePreds.csv", delim=',')
```

Our log loss went up to 0.97432, which is a major downgrade. So, clearly we should stick to one of the above. Maybe we could adjust our recipe a bit too?

## Step 4: New Recipe

```r
new_recipe <- recipe(shot_made_flag ~ ., data = trainData) %>%
  step_date(game_date, features="month") %>%
  step_date(game_date, features="year") %>%
  step_mutate(home_away = if_else(str_detect(matchup, "@"), "away", "home")) %>%
  step_mutate(game_event_id = factor(game_event_id)) %>%
  step_mutate(period = factor(period)) %>%
  step_mutate(playoffs = factor(playoffs)) %>%
  step_mutate(home_away = factor(home_away)) %>%
  step_mutate(game_date_month = factor(game_date_month)) %>%
  step_mutate(game_date_year = factor(game_date_year)) %>%
  step_rm(shot_id, team_name, team_id, matchup, game_id, game_date,
          combined_shot_type) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_mixed(all_factor_predictors(),
                     outcome = vars(shot_made_flag)) %>%
  step_normalize(all_factor_predictors())
```

Here, I made some adjustments to the recipe.

- I made a new feature called `home_away`, where I looked in the `matchup` variable to see whether the game was home or away. I did this by detecting `@` signs, which would tell us that it was an away game. I made that a factor as well.
- I removed another variable, `combined_shot_type`, since it was very similar to `action_type` but had less information.

Anything else you would change?

## Step 5: Final Model

Seeing as the Random Forest did the best before, perhaps we could tune it a bit more. I did a very similar workflow, but changed my tuning grid. This one gave me the best results, in which I shortened the `mtry` range and added one for `min_n`.

```r
tuning_grid <- grid_regular(mtry(range = c(2, 8)),
                            min_n(range = c(60, 120)),
                            levels=5)
```

The best `mtry` ended up being 4, and the best `min_n` was 120. What did it look like for you?

## Conclusion

The above model gave me a final Kaggle score of 0.60771, which is super close to our goal!

I did try other models to get lower, such as tuning trees and and XGBoost model, but they were either much too computationally expensive or did not lessen my log-loss.

What models would you try? Would you change the recipe at all? I encourage you to try this competition out for yourself, it was a lot of fun! You should also try this workflow on a different Kaggle competition, and adjust it to what works for you.

Good luck predicting!

## More From Me

- A Kaggle Notebook describing this project: [Kobe Bryant Notebook](https://www.kaggle.com/code/brookeseeley/kobe-bryant-shot-selection)
- My GitHub for this project: [Kobe Bryant Repository](https://github.com/brooke-seeley/KobeBryant)
