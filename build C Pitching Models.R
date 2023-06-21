
library(tidyverse)
options(scipen=999)
library(xgboost)
library(modelr)
setwd("")

# read in data
s2017 <- read_csv('statcast_2017.csv')
s2018 <- read_csv('statcast_2018.csv')
s2019 <- read_csv('statcast_2019.csv') %>% filter(game_date > "2019-03-27")
s2021 <- read_csv('statcast_2021.csv')
s2022 <- read_csv('statcast_2022.csv') %>% filter(game_date > "2022-04-06")

## Bring data together. Create filter for model that will be created (ex RHP vs. LHB)
mlbraw <- bind_rows(s2017, s2018, s2019, s2021, s2022) %>% distinct() %>% filter(p_throws == 'R' & stand == 'L')
rm(s2022, s2021, s2019, s2018, s2017)

### Filter out pitchers hitting, besides Ohtani. This filters out any hitter that's thrown 75 or more pitches between 2017-2022
pitchers <-  mlbraw %>% group_by(pitcher) %>% summarize(pitches=n()) %>% ungroup() %>% filter(pitches > 74, pitcher != "660271")

## remove: NA / incorrect values, rare pitch types, outcomes that have nothing to do with pitch quality (like pickoffs).
# normalize pfx_x, spin and release points for lefties, treat all field outs as the same, treat sacrifice plays as the same,
# group two seamers and sinkers, group curve and knuckle curve, group slider slurve and sweeper, create variable for count, 
mlbraw1 <- mlbraw %>% anti_join(pitchers, by = c("batter"="pitcher")) %>%
  filter(description != "pitchout", balls < 4, strikes < 3, outs_when_up < 3, !is.na(release_speed), !is.na(release_spin_rate), !is.na(release_extension),
         !is.na(release_pos_x), !is.na(release_pos_z), !is.na(p_throws), !is.na(stand), !is.na(zone), !is.na(plate_x), !is.na(plate_z),
         !pitch_type %in% c("EP", "PO", "KN", "FO", "CS", "SC", "FA"), !is.na(spin_axis), !is.na(pfx_x), !is.na(pfx_z), !is.na(delta_run_exp),
         !str_detect(des, "pickoff"),!str_detect(des, "caught_stealing"), !str_detect(des, "stolen_"), !des %in% c("game_advisory", "catcher_interf")) %>%
  mutate(release_pos_x = ifelse(p_throws == "R", release_pos_x, -release_pos_x),
         pfx_x = ifelse(p_throws == "R", pfx_x, -pfx_x),
         spin_axis = ifelse(p_throws == "R", spin_axis, -spin_axis), year = year(as.Date(game_date)),
         events = case_when(
           events %in% c("double_play", "triple_play", "field_error", "field_out", "fielders_choice", "fielders_choice_out",
                         "force_out", "grounded_into_double_play") ~ "out",
           events %in% c("sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play") ~ "sacrifice",
           TRUE ~ events),
         pitch_type = case_when(
           pitch_type %in% c("FT", "SI") ~ "SI",
           pitch_type %in% c("CU", "KC") ~ "CU",
           pitch_type %in% c("ST", "SL", "SV") ~ "SL",
           TRUE ~ pitch_type), 
         count = paste(balls, strikes, sep="-")) %>%
  mutate(above_zone = plate_z-sz_top, below_zone = sz_bot-plate_z, zone_x_max = plate_x-.83, zone_x_min = plate_x+.83)

## Find average delta run values for each ball or strike (pitches not put in play)
bs_vals <- mlbraw1 %>% filter(type != "X") %>%  group_by(balls, strikes, type) %>% summarize(dre_bs = mean(delta_run_exp, na.rm=T))

## same thing but for balls in play
ip_filt <- mlbraw1 %>% filter(type == 'X')

event_lm <- lm(delta_run_exp ~ balls + strikes + events, data=ip_filt)
summary(event_lm)

## derive dre values for balls in play
ip_vals <- ip_filt %>% add_predictions(event_lm, var = "pred_dre") %>%  group_by(balls, strikes, events) %>%
  summarize(dre_ip = mean(pred_dre, na.rm=T))

## merge all dre values into main dataframe
mlbraw1 <- mlbraw1 %>% left_join(bs_vals, by = c("balls", "strikes", "type")) %>%
  left_join(ip_vals, by = c("balls", "strikes", "events")) %>%
  mutate(dre_final = case_when(
    type != "X" ~ dre_bs,
    TRUE ~ dre_ip))

## establish each pitcher's fastball metrics by season
pitcher_fastballs <- mlbraw1 %>% filter(pitch_type %in% c("FF", "FC", "SI")) %>% group_by(pitcher, year) %>% 
  summarize(fb_velo = mean(release_speed), fb_max_ivb = quantile(pfx_z, .8), fb_max_x = quantile(pfx_x, .8), fb_min_x = quantile(pfx_x, .2), 
            fb_max_velo = quantile(release_speed, .8),fb_axis = mean(spin_axis, na.rm=T))

## join to main dataframe, create difference variables
mlbraw2 <- mlbraw1 %>% left_join(pitcher_fastballs, by = c("year", "pitcher")) %>% mutate(spin_dif = spin_axis - fb_axis, velo_dif = release_speed-fb_velo,
                                                                                          ivb_dif = fb_max_ivb-pfx_z, break_dif = (fb_max_x*.5+fb_min_x*.5)-pfx_x)

## filter for only variables to be used in model
final_vars <- mlbraw2 %>% select(dre_final, starts_with("fb_"), release_speed, release_spin_rate, count, above_zone, below_zone,
                                 release_extension, release_pos_x, release_pos_z, above_zone, below_zone, pfx_x, pfx_z, zone_x_max, zone_x_min,
                                 plate_x, plate_z, pitch_type, spin_axis, spin_dif, velo_dif, ivb_dif, break_dif) 


## create dummy variables where needed
library(caret)
dmy <- dummyVars(" ~ count+pitch_type", data = final_vars)
trsf <- data.frame(predict(dmy, newdata = final_vars))

### join data. remove original variables which dummies have been created for. filter out any missing run values
vars <- cbind(final_vars, trsf) %>% select(-c(count, pitch_type)) %>% filter(!is.na(dre_final))

# Remove all objects in R except "vars" dataframe
rm(list = setdiff(ls(), "vars"))
gc()

set.seed(2552)
library(tidymodels)
library(mlr)

#split into training (70%) and testing set (30%)
c_pitching_split <- initial_split(vars, strata = dre_final, prop = .7) 
train <- training(c_pitching_split)
test <- testing(c_pitching_split)

## hyperparam testing
traintask <- makeRegrTask (data = train,target = "dre_final")
testtask <- makeRegrTask (data = test,target = "dre_final")

rm(train, test)

#create learner
lrn <- makeLearner("regr.xgboost")
lrn$par.vals <- list( objective="reg:squarederror", eval_metric="rmse")

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = "gbtree"),
                        makeIntegerParam("max_depth",lower = 4L,upper = 12L), 
                        makeIntegerParam("min_child_weight",lower = 2L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.3,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.3,upper = 1),
                        makeIntegerParam("gamma",lower = 0L,upper = 5L),
                        makeIntegerParam("alpha",lower = 0L,upper = 5L),
                        makeDiscreteParam("nrounds", 
                                          values = c(50, 100, 200, 300, 400, 500, 600)),
                        makeDiscreteParam("eta",
                                          values = c(.01, .03, .05, .075, .1, .15, .2)))

#set resampling strategy
rdesc <- makeResampleDesc("CV", iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)
gc()
#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

# parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                     par.set = params, control = ctrl, show.info = T)

##Observe rmse results. .1992 LHP vs. LHB; .2064 lhp vs. RHB; .2077 RHP vs RHB; .2017 rhp vs lhb
sqrt(mytune$y)

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)

## view feature importance
imp <- mlr::getFeatureImportance(xgmodel)
view(imp$res)

#predict on test data .2014 lhp vs lhb; .2082 lhp vs rhb; .2067 RHP vs. RHB; .2027 rhp vs lhb
xgpred <- predict(xgmodel,testtask) %>% as.data.frame()

caret::RMSE(xgpred$truth, xgpred$response)


## retrain model on full dataset and save for future use
fulltask <- makeRegrTask (data = vars,target = "dre_final")

xgmodel <- train(learner = lrn_tune,task = fulltask)

saveRDS(xgmodel, file = "c_pitch_RHPvsLHB.rds")