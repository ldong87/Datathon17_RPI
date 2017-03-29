#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:37:52 2017

@authors: Li Dong, Han Wang, Mike Agiorgousis
"""

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
import xgboost as xgb 
from matplotlib import pyplot
from sklearn import linear_model as sklm
from sklearn import svm as sksvm
from sklearn import metrics
from sklearn import ensemble as ensemble

layout_m = np.zeros([1])
def read_layout():
	global layout_m
	robjects.r['load']('./ahrf2016r/data/ahrf_county_layout.rda')
	layout_m = np.array(robjects.r['ahrf_county_layout']).T
	(lx, ly ) = layout_m.shape 
	for i in range(lx):
		layout_m[i,0]=layout_m[i,0].strip()
	
def find_index(flabel) :
    global layout_m
    flabel=flabel.strip()
    count = 0
#    print 'kkk',layout_m.shape
    for i in layout_m[:,0]:
        count = count + 1
        if (i == flabel):
            return count - 1
        
def clean_data(data_raw):
    (d_x, d_y) = data_raw.shape
    data = np.zeros([d_x, d_y])
    for i in range(d_x ):
    	for j in range(d_y):
    		try:
    			data[i,j] = float(data_raw[i,j])
    		except:
    			data[i,j] = 0.0
    return data

def targets_transform(train):
    targets = np.zeros([np.shape(train)[0], 3])
    num_bed = np.average(train[:,3298:3301],1) + 1.
    targets[:,0] = np.average(train[:,3904:3907],1)/(365.*num_bed)
    num_doc = np.sum([np.average(train[:,69:74],1),np.average(train[:,74:79],1)],0) + 1.
    targets[:,1] = np.sum([np.average(train[:,3974:3977],1),np.average(train[:,3977:3980],1),\
                           np.average(train[:,3980:3983],1),                train[:,3983]],0)/(365.*num_doc)
    num_srg = np.sum([np.average(train[:,4025:4028],1),np.average(train[:,4028:4031],1)],0) + 1.
    targets[:,2] = np.sum([np.average(train[:,4018:4021],1),np.average(train[:,4021:4024],1),\
                           train[:,4024]],0)/(365.*num_srg)
    targets_ensemble = 0.594*targets[:,0]+0.0459*targets[:,1]+0.36*targets[:,2]
    return targets

def samples_transform(train):
    (size_x, size_y) = train.shape
    num_sample = 500
    sample_f=np.zeros( [size_x, num_sample])
    
    #targets = targets_transform(train)
    
    read_layout()
#    print find_index('F04449')
#    print train[find_index('F04449')]
    
    #0
    
    ##Get population data
    tmp_popu = train[:, find_index('F1198415')]
#    print tmp_popu
    
    start_i = find_index('F0852210')
    end_i = find_index('F1165510')
    label_col = [' ']* num_sample
    count = 0
    for i in range(start_i, end_i+1   ,2) :
        sample_f[:,count] =  train[:, i]
        label_col[count] = layout_m[i,5]
    
        sample_f[:,count]  = sample_f[:,count] / (1.+tmp_popu[:])
        count = count +1
#    print label_col
#    print sample_f
    
    ##End population data
     
    ##Urban and Rural Population
    index_tmp = find_index('F1491710') 
    urb_cens=train[:,index_tmp ]/(1.+tmp_popu[:])
    sample_f[:,count]  = urb_cens
    label_col[count] =  layout_m[index_tmp,5] 
    count = count +1
    
    index_tmp = find_index('F1492010') 
    rur_cens=train[:,index_tmp ]/(1.+tmp_popu[:])
    sample_f[:,count]  = rur_cens
    label_col[count] =  layout_m[index_tmp,5]
    
    count = count +1
    
    # house size
    index_tmp = find_index('F1351310') 
    sample_f[:,count]  = train[:,index_tmp ] 
    label_col[count] =  layout_m[index_tmp,5]
    
    count = count +1
    
    #  total number of household
    index_tmp = find_index('F0874510') 
    
#    sample_f[:,count]  = train[:,index_tmp ] 
    tot_hd = train[:,index_tmp ] #sample_f[:,count] 
#    label_col[count] =  layout_m[index_tmp,5]
    
#    count = count +1
    
#    print label_col
    
    ## household size dist
    
    start_i = find_index('F0873810')
    end_i = find_index('F0874310')
     
     
    for i in range(start_i, end_i+1   ,2) :
        sample_f[:,count] =  train[:, i]
        label_col[count] = layout_m[i,5]
    
        sample_f[:,count]  = sample_f[:,count]  / (1.+tot_hd)
        count = count +1
        
    #print label_col
    #print sample_f
     
    ##U birth and death difference
    index_tmp = find_index('F1254612') 
    birth=train[:,index_tmp ] 
    label_col[count] =  layout_m[index_tmp,5] 
    index_tmp = find_index('F1194112') 
    death=train[:,index_tmp ] 
    
    sample_f[:,count]  = birth - death
    label_col[count] =  layout_m[index_tmp,5] +'-'+ label_col[count] 
    count = count +1
    
    ## death reason rate
    
    start_i = find_index('F1266910')
    end_i = find_index('F1317008')
     
     
    for i in range(start_i, end_i+1   ,2) :
        sample_f[:,count] =  train[:, i]
        label_col[count] = layout_m[i,5]
    
        sample_f[:,count]  = sample_f[:,count]  / (1.+death)
        count = count +1
    
    ##  median household income
    #  total number of household
    index_tmp = find_index('F1434510') 
    
    sample_f[:,count]  = train[:,index_tmp ] 
    label_col[count] =  layout_m[index_tmp,5]
    
    count = count +1
    
    ##  poverty  data dist
    
    start_i = find_index('F1440110')
    end_i = find_index('F1440710')
     
     
    for i in range(start_i, end_i+1   ,2) :
        sample_f[:,count] =  train[:, i]
        label_col[count] = layout_m[i,5]
    
        sample_f[:,count]  = sample_f[:,count]  / (1.+tot_hd)
        count = count +1
    
##  insurace vs age
    index_tmp = find_index('F1547114') 
    young_popu=train[:,index_tmp ]
    
    index_tmp = find_index('F1547314') 
    
    sample_f[:,count]  = train[:,index_tmp ]  /  ( young_popu +1.0)
    label_col[count] =  layout_m[index_tmp,5]
    
    count = count +1
     
     #Households on government assistance
    index_tmp = find_index('F1444010') 
    adu_popu=train[:,index_tmp ]
    
    start_i = find_index('F1443710')
    end_i=find_index('F1443910')
    
    for i in range(start_i,end_i+1,2) :
        sample_f[:,count] =  train[:, i]
        label_col[count] = layout_m[i,5]
    
        sample_f[:,count]  = sample_f[:,count]  / (1.+adu_popu)
        count = count +1
    #% education data
    start_i = find_index('F1445010')
    end_i = find_index('F1447910')
     
     
    for i in range(start_i, end_i+1   ,2) :
        sample_f[:,count] =  train[:, i]
        label_col[count] = layout_m[i,5]
    
        sample_f[:,count]  = sample_f[:,count]  / (1.+tot_hd)
        count = count +1 
    
    # work force data
    index_tmp = find_index('F1451010') 
    workforce_popu=train[:,index_tmp ]
    
    start_i = find_index('F1451110')
    end_i =   find_index('F1456906')
     
     
    for i in range(start_i, end_i+1   ,2) :
        sample_f[:,count] =  train[:, i]
        label_col[count] = layout_m[i,5]
    
        sample_f[:,count]  = sample_f[:,count]  / (1.+workforce_popu)
        count = count +1 
        
        
    # environment, air quality days measured
    index_tmp = find_index('F1526415') 
    days_meansure=train[:,index_tmp ]
    
    index_tmp = find_index('F1526515') 
    days_good=train[:,index_tmp ]
    
    sample_f[:,count]  = days_good /(1.0 + days_meansure)
    label_col[count] =  layout_m[index_tmp,5]
    
    count = count +1   
    
    label_col = np.array(label_col[0:count-1])
    sample_f = np.array(sample_f[:,0:count-1])
    
#    imp_ind = [261, 255, 250, 223, 252, 312, 217, 272, 27, 333]
    label_col = label_col[imp_ind]
    sample_f = sample_f[:,imp_ind]

    return [label_col,sample_f]

def feature_importance(s,t,s_,t_,nFeat):
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=50, silent=False, \
                             objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1, \
                             max_delta_step=0, subsample=1, colsample_bytree=1, \
                             colsample_bylevel=1, reg_alpha=2, reg_lambda=0, scale_pos_weight=1, \
                             base_score=0.5, seed=0, missing=None)
    model.fit(s, t)
    feature_imp = model.feature_importances_
    feature_imp_norm = feature_imp/sum(feature_imp)
    feature_imp_ind = feature_imp.argsort()[-nFeat:][::-1]
    feature_unimp_ind = feature_imp.argsort()[0:nFeat]
    print 'feature_imp_ind=',feature_imp_ind,' feature_imp_norm=',feature_imp_norm[feature_imp_ind],\
    'feature_unimp_ind=',feature_unimp_ind,' feature_unimp_norm=',feature_imp_norm[feature_unimp_ind]
    
def model_lasso(s,t,s_,t_,flagCV,nFeat):
    if flagCV:
        #bad r2
        #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV
        clf = sklm.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, \
                           precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, \
                           verbose=False, n_jobs=1, positive=False, random_state=None, \
                           selection='cyclic')
    else:
        #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
        clf = sklm.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, \
                         copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False,\
                         random_state=None, selection='cyclic')
    clf.fit(s, t)
    print 'coeffs = ',clf.coef_, '  intercept = ',clf.intercept_
    
    feature_imp = clf.coef_
    feature_imp_ind_neg = feature_imp.argsort()[0:nFeat]
    feature_imp_ind_pos = feature_imp.argsort()[-nFeat:][::-1]
    print 'feature_imp_ind_neg=',feature_imp_ind_neg,'feature_imp_ind_pos=',feature_imp_ind_pos
    
    r2_train = clf.score(s,t)
    r2_test  = clf.score(s_,t_)
    print 'r2_train=',r2_train,' r2_test=',r2_test

def model_xgb(s,t,s_,t_,eta,depth,alpha,num_iter):
    #http://xgboost.readthedocs.io/en/latest/parameter.html
    xg_train = xgb.DMatrix(s, label=t)
    xg_test = xgb.DMatrix(s_, label=t_)
    watchlist = [(xg_train,'train'), (xg_test,'test')]
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = eta
    param['max_depth'] = depth
    param['silent'] = 1 
    param['nthread'] = 4 
    param['alpha'] = alpha
    num_round = num_iter
    bst = xgb.train(param, xg_train, num_round, watchlist);
    r2_train = metrics.r2_score(t, bst.predict(xg_train))
    r2_test  = metrics.r2_score(t_,bst.predict(xg_test))
    print 'r2_train=',r2_train,' r2_test=',r2_test

def model_svm(s,t,s_,t_,flagLinear):
    # bad r2
    if flagLinear==0:
        #http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR
        clf = sksvm.NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=5, gamma='auto', coef0=0.0, shrinking=True, \
                    tol=0.001, cache_size=200, verbose=False, max_iter=1000)
        clf.fit(s, t)
    else:
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
        # this loss function is L1, thus insensitive to outliers
        clf = sksvm.LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', \
                              fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, \
                              random_state=None, max_iter=1000)
        clf.fit(s, t)
        print 'coeffs = ',clf.coef_, '  intercept = ',clf.intercept_
    r2_train = clf.score(s,t)
    r2_test  = clf.score(s_,t_)
    print 'r2_train=',r2_train,' r2_test=',r2_test

def model_GBreg(s,t,s_,t_):
    #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    clf = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300, subsample=1.0, \
                                       criterion='friedman_mse', min_samples_split=2, \
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5, \
                                       min_impurity_split=1e-07, init=None, random_state=None, \
                                       max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, \
                                       warm_start=False, presort='auto')
    clf.fit(s,t)
    r2_train = clf.score(s,t)
    r2_test  = clf.score(s_,t_)
    print 'r2_train=',r2_train,' r2_test=',r2_test
    
    
robjects.r['load']('./ahrf2016r/data/ahrf_county_train.rda')
trainAll = np.array(robjects.r['ahrf_county_train']).T
train = trainAll#[0:200,:];
                
robjects.r['load']('./ahrf2016r/data/ahrf_county_test.rda')
testAll = np.array(robjects.r['ahrf_county_test']).T
test = testAll#[0:100,:];

train = clean_data(train)
test = clean_data(test)
targets = targets_transform(train)
targets_ = targets_transform(test)
[labels,samples] = samples_transform(train)
[_,samples_] = samples_transform(test)


output = 0

model_xgb(samples,targets[:,output],samples_,targets_[:,output],0.1,5,2,50)
#model_lasso(samples,targets[:,1],samples_,targets_[:,1],0,10)

feature_importance(samples,targets[:,output],samples_,targets_[:,output],10)