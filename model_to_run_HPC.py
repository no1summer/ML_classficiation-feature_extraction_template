#!/usr/bin/env python
# coding: utf-8

# In[16]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

sample_folder=''


# In[3]:
for experiment_group in ['a','b','c']:
    feature_list=[]

    import pandas as pd 
    import numpy as np     
    df = pd.read_feather('/sample.feather')
    pd.set_option('display.max_columns', None)
    df_twogroup=df.loc[df['Treatment'].isin(['NT',experiment_group])]



    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.inspection import permutation_importance
    from mrmr import mrmr_classif
    from xgboost import XGBClassifier

    def data_process (df,large_sample):
        '''
        this data process function is for selection of useful numerical feature
        df: data frame with Treatment column as the y
        large_sample: True or False as determine if all samples will be output or just a small fraction will be output
        '''
        X = df.drop(['Treatment'],axis = 1)
        y = df.Treatment
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        #X.fillna(0,inplace=True)
        X_numerical = X.select_dtypes('number')
        X_useful=X_numerical.loc[:,X_numerical.iloc[1:5,].nunique() > 1]
        X_useful=X_useful.drop(['ObjectNumber','Number_Object_Number','Parent_PrefilterCellsed','Parent_Cells'],axis=1,errors='ignore')
        df_preprocessed=X_useful.join(y)
        if large_sample:
            df_sample=df_preprocessed.groupby("Treatment").sample(n=np.min(df_preprocessed.Treatment.value_counts()), random_state=1)
        else:
            df_sample=df_preprocessed.groupby("Treatment").sample(n=10000, random_state=1)
        X_useful = df_sample.drop(['Treatment'],axis = 1)
        y_sample = df_sample.Treatment
        return X_useful, y_sample


    large_sample=False
    with PdfPages(sample_folder+'/model_performance_feature_cellpose_texture_noradgra_'+experiment_group+'.pdf') as pdf:

        X_useful,y_sample=data_process(df_twogroup,large_sample)
        x_train, x_test, y_train, y_test = train_test_split(X_useful, 
                                                            y_sample, 
                                                            test_size=0.2, 
                                                            random_state=42)
        scaler = StandardScaler()
        normalized_x_train = pd.DataFrame(
            scaler.fit_transform(x_train),
            columns = x_train.columns
        )
        normalized_x_test = pd.DataFrame(
            scaler.transform(x_test),
            columns = x_test.columns
        )

        #logistic regression for two groups
        logreg = LogisticRegression(max_iter=5000)
        logreg.fit(normalized_x_train, y_train)
        y_pred = logreg.predict(normalized_x_test)
        print()
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        disp=ConfusionMatrixDisplay(conf_mat,display_labels=logreg.classes_)
        disp.plot()
        plt.title('logistic regression accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        pdf.savefig(bbox_inches='tight')
        plt.close()

        #feature importance for two groups lr 
        model_fi = permutation_importance(logreg, normalized_x_train, y_train)
        feature_order=np.argsort(abs(model_fi['importances_mean']))
        feature_importance=np.sort(abs(model_fi['importances_mean']))
        data = pd.DataFrame(data=feature_importance[-10:], index=normalized_x_train.columns[feature_order[-10:]], columns=["score"]).sort_values(by = "score", ascending=False)
        plt.figure(figsize=(6,6))
        data.nlargest(10, columns="score").plot(kind='barh') ## plot top 10 features
        plt.title("feature importance by permutation for logistic regression")
        pdf.savefig(bbox_inches='tight')
        plt.close()
        feature_list.append(list(normalized_x_train.columns[feature_order[-10:]]))


        #xgboost
        le = LabelEncoder()
        le.fit(y_sample)
        xgb = XGBClassifier(random_state =1)
        xgb.fit(normalized_x_train,le.transform(y_train))
        y_pred = xgb.predict(normalized_x_test)
        #print('accuracy of xgboost is {}'.format(accuracy_score(le.transform(y_test), y_pred)))
        conf_mat = confusion_matrix(le.transform(y_test), y_pred)
        plt.figure(figsize=(6,6))
        disp=ConfusionMatrixDisplay(conf_mat,display_labels=xgb.classes_)
        disp.plot()
        plt.title('xgboost accuracy:{}'.format(accuracy_score(le.transform(y_test), y_pred)))
        pdf.savefig(bbox_inches='tight')
        plt.close()

        feature_important = xgb.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        plt.figure(figsize=(6,6))
        data.nlargest(10, columns="score").plot(kind='barh') ## plot top 10 features
        plt.title("feature importance by weight for xgboost")
        pdf.savefig(bbox_inches='tight')
        plt.close()
        feature_list.append(list(data[:10].index))



    with PdfPages(ample_folder+'/mrmr_cellpose_texture_noradgra_'+experiment_group+'.pdf') as pdf:

        X_useful,y_sample=data_process(df_twogroup,large_sample)
        le = LabelEncoder()
        le.fit(y_sample)

        scaler = StandardScaler()
        normalized_X = pd.DataFrame(
            scaler.fit_transform(X_useful),
            columns = X_useful.columns
        )

        #mrmr using f statistic
        selected_features_f = mrmr_classif(X=normalized_X, y=le.transform(y_sample), K=10,return_scores=True,relevance='f')
        sorted_relevance = selected_features_f[1].sort_values(ascending=False)
        min = np.min(sorted_relevance)
        max = np.max(sorted_relevance)
        normalized_sorted_relevance = (sorted_relevance - min) / (max - min)
        sorted_relevance = normalized_sorted_relevance / sum(normalized_sorted_relevance)
        mrmr_feature_vs_score_df = pd.DataFrame({'Features': sorted_relevance.index[:10], 'Values': sorted_relevance[:10]})
        plt.figure(figsize=(6,6))
        mrmr_feature_vs_score_df.plot(kind='barh') ## plot top 10 features
        plt.title("feature importance by f-statistic for mrmr")
        #plt.show()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        feature_list.append(list(sorted_relevance.index[:10]))
        
        #mrmr using random forest
        selected_features_rf = mrmr_classif(X=X_useful, y=le.transform(y_sample), K=10,return_scores=True,relevance='rf')
        sorted_relevance = selected_features_rf[1].sort_values(ascending=False)
        min = np.min(sorted_relevance)
        max = np.max(sorted_relevance)
        normalized_sorted_relevance = (sorted_relevance - min) / (max - min)
        sorted_relevance = normalized_sorted_relevance / sum(normalized_sorted_relevance)
        mrmr_feature_vs_score_df = pd.DataFrame({'Features': sorted_relevance.index[:10], 'Values': sorted_relevance[:10]})
        plt.figure(figsize=(6,6))
        mrmr_feature_vs_score_df.plot(kind='barh') ## plot top 10 features
        plt.title("feature importance by random forest for mrmr")
        pdf.savefig(bbox_inches='tight')
        plt.close()
        feature_list.append(list(sorted_relevance.index[:10]))


    # In[17]:


    #shap for model explanation
    import shap
    import xgboost 

    with PdfPages(sample_folder+'/shap_cellpose_texture_noradgra_'+experiment_group+'.pdf') as pdf:
    
        X_useful,y_sample=data_process(df_twogroup,large_sample)
        
        le = LabelEncoder()
        le.fit(y_sample)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)

        # train an XGBoost model
        model = xgboost.XGBClassifier().fit(X_useful, le.transform(y_sample))

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer = shap.Explainer(model)
        shap_values = explainer(X_useful)

        # show the feature shap value for all the samples 
        shap.plots.beeswarm(shap_values,show=False)
        plt.title('xgboost')
        pdf.savefig(bbox_inches='tight')
        plt.close()
        feature_names = X_useful.columns
        vals = np.abs(shap_values.values).mean(0)
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                        columns=['col_name','feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'],
                                    ascending=False, inplace=True)
        feature_list.append(list(shap_importance.col_name[:10]))



        #logistic regression
        '''
        scaler = StandardScaler()
        normalized_X = pd.DataFrame(
            scaler.fit_transform(X_useful),
            columns = X_useful.columns
        )
        
        logreg = LogisticRegression(max_iter=5000)
        logreg.fit(normalized_X, y_sample)

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer = shap.Explainer(logreg,normalized_X)
        shap_values = explainer(normalized_X)
        '''
        x_train, x_test, y_train, y_test = train_test_split(X_useful, 
                                                            y_sample, 
                                                            test_size=0.2, 
                                                            random_state=42)
        scaler = StandardScaler()
        normalized_x_train = pd.DataFrame(
            scaler.fit_transform(x_train),
            columns = x_train.columns
        )
        normalized_x_test = pd.DataFrame(
            scaler.transform(x_test),
            columns = x_test.columns
        )

        #logistic regression for two groups
        logreg = LogisticRegression(max_iter=5000)
        logreg.fit(normalized_x_train, y_train)

        explainer = shap.Explainer(logreg,normalized_x_train)
        shap_values = explainer(normalized_x_test)

        # show the feature shap value for all the samples 
        shap.plots.beeswarm(shap_values,show=False)
        plt.title('logistic regression')
        pdf.savefig(bbox_inches='tight')
        plt.close()
        feature_names = normalized_X.columns
        vals = np.abs(shap_values.values).mean(0)
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                        columns=['col_name','feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'],
                                    ascending=False, inplace=True)
        feature_list.append(list(shap_importance.col_name[:10]))

    with open(sample_folder+'/featurelist_cellpose_texture_noradgra_'+experiment_group+'.txt', "w") as f:
        for s in feature_list:
            f.write(str(s))