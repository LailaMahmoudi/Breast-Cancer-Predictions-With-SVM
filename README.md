# Breast-Cancer-Predictions-With-SVM : Project Overview


![](https://mlfjqdsf5ptg.i.optimole.com/iQrIoNc-LQvF_N5U/w:800/h:400/q:69/https://nationaldaycalendar.com/wp-content/uploads/2014/10/Breast-Cancer-Awareness-Month-October-1.jpg)



**Implementation of SVM Classifier To Perform Classification on the dataset of Breast Cancer Wisconin; to predict if the tumor is cancer or not**.

* Building some plots and graphs to take an overview about what your data looks like.

* Machine Learning Algorithms used in this Notebook: Logistic Regression, Gradient Boosting Classifier, Random Forest Classifier, Decision Tree Classifier, Kneighbours Classifier, XGB Classifier, Supportr vector Classifier

* Evaluating the performance of SVM Classifier by Differents Metrics.


# Code and Resources Used 

* Python Version: 3.8.3

* Packages : Pandas, Numpy, Matplotlib, Seaborn, Sklearn.

* [Understanding a Classification Report For Your Machine Learning Model](https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397).

* [True Positive Rate](https://www.sciencedirect.com/topics/computer-science/true-positive-rate)

* [How to plot an ROC curve in Python](https://www.sciencedirect.com/topics/computer-science/true-positive-rate)


# Data
 
 [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)


# Look at the dataset

 ![difference between Malignant and Benignt](https://gotalktogetherdotcom.files.wordpress.com/2016/05/cancerbenignmalig1.jpg?w=550)
 
 # EDA
 
 * Checking for the correlation 
 
 
  ![](https://www.kaggleusercontent.com/kf/44797707/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..q-QmVsJmOeECtdjQAtzdlQ.mDkpMK_zPBgsiF3lkc-qPYS9IT-vMT1f_Wc9bu42fUF6YaAaZHSvzfTI8b6CAvhekmx9xNH3UNU2ngGrwLREjyzqMutxGxIRcTjSZLwmxxqIsZj8VS1xX0-wXiJtqeM06NUFQ5UCO4Y0sZapgUhto6yn0JDk1mHnIgHDkuOmwA9V9JXUgEKrUGZXWUlicltHeazooTH_sJ3xH9EzkQjsstfQEkZ9vLP91vE8N9xEtVXmcuxXWkmcvm_VNCgkALxO2GVgF63BqjFt4155ULP_GqC6h7Mjtmb7ehhMAMmFGu20DUBbXp5-xe5wHj0ZRvfhFzjUS1XJi6mPpEJ69kkpiNh3_CckVhpi-__eXYQXnqWKi68gQAqWC1_os8dffLmDwVUqXJ62EHCJlyfUGaMKuj_25td5gmCvw8iH1N-df1wAl66eZulVGWx9Ye70zS45KYnoL7aRgEMRg1J6m2nHYI-vJWttXYJWipdOPzUpw-0XUGoEOaTz9cGOyKqFf-gwpy2r84VDU5Mx2Prh7CMpqKtDI_2bk4cgr14puIkU3bME-kgDhyqxmN_KZh5qMr8RM63NtD2RyWOx4AXL_XHXHHynNLM9Ioc74Waz8-0F11PAYCvZuqsBswHDR_ahXx-qgIK1dkQWzyNoBqdapcc6vlSgreMFQcZ-U1jY2JgVKd4.fgAXN29VHTT2EnnJoETb3w/__results___files/__results___26_0.png)
 
  * plotting the highly correlated pairs
  
  ![](https://www.kaggleusercontent.com/kf/44797707/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..q-QmVsJmOeECtdjQAtzdlQ.mDkpMK_zPBgsiF3lkc-qPYS9IT-vMT1f_Wc9bu42fUF6YaAaZHSvzfTI8b6CAvhekmx9xNH3UNU2ngGrwLREjyzqMutxGxIRcTjSZLwmxxqIsZj8VS1xX0-wXiJtqeM06NUFQ5UCO4Y0sZapgUhto6yn0JDk1mHnIgHDkuOmwA9V9JXUgEKrUGZXWUlicltHeazooTH_sJ3xH9EzkQjsstfQEkZ9vLP91vE8N9xEtVXmcuxXWkmcvm_VNCgkALxO2GVgF63BqjFt4155ULP_GqC6h7Mjtmb7ehhMAMmFGu20DUBbXp5-xe5wHj0ZRvfhFzjUS1XJi6mPpEJ69kkpiNh3_CckVhpi-__eXYQXnqWKi68gQAqWC1_os8dffLmDwVUqXJ62EHCJlyfUGaMKuj_25td5gmCvw8iH1N-df1wAl66eZulVGWx9Ye70zS45KYnoL7aRgEMRg1J6m2nHYI-vJWttXYJWipdOPzUpw-0XUGoEOaTz9cGOyKqFf-gwpy2r84VDU5Mx2Prh7CMpqKtDI_2bk4cgr14puIkU3bME-kgDhyqxmN_KZh5qMr8RM63NtD2RyWOx4AXL_XHXHHynNLM9Ioc74Waz8-0F11PAYCvZuqsBswHDR_ahXx-qgIK1dkQWzyNoBqdapcc6vlSgreMFQcZ-U1jY2JgVKd4.fgAXN29VHTT2EnnJoETb3w/__results___files/__results___28_0.png)
  
  # Model Building
  
  * In this section,  I tried different models and evaluate them using the Accuracy_Score:
  
   + **Logistic Regression** 
   + **Gradient Boosting Classifier**
   + **Random Forest Classifier**
   + **Decision Tree Classifier**
   + **Kneighbours Classifier**
   + **XGB Classifier**
   + **Supportr vector Classifier**


![](https://www.kaggleusercontent.com/kf/44797707/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..q-QmVsJmOeECtdjQAtzdlQ.mDkpMK_zPBgsiF3lkc-qPYS9IT-vMT1f_Wc9bu42fUF6YaAaZHSvzfTI8b6CAvhekmx9xNH3UNU2ngGrwLREjyzqMutxGxIRcTjSZLwmxxqIsZj8VS1xX0-wXiJtqeM06NUFQ5UCO4Y0sZapgUhto6yn0JDk1mHnIgHDkuOmwA9V9JXUgEKrUGZXWUlicltHeazooTH_sJ3xH9EzkQjsstfQEkZ9vLP91vE8N9xEtVXmcuxXWkmcvm_VNCgkALxO2GVgF63BqjFt4155ULP_GqC6h7Mjtmb7ehhMAMmFGu20DUBbXp5-xe5wHj0ZRvfhFzjUS1XJi6mPpEJ69kkpiNh3_CckVhpi-__eXYQXnqWKi68gQAqWC1_os8dffLmDwVUqXJ62EHCJlyfUGaMKuj_25td5gmCvw8iH1N-df1wAl66eZulVGWx9Ye70zS45KYnoL7aRgEMRg1J6m2nHYI-vJWttXYJWipdOPzUpw-0XUGoEOaTz9cGOyKqFf-gwpy2r84VDU5Mx2Prh7CMpqKtDI_2bk4cgr14puIkU3bME-kgDhyqxmN_KZh5qMr8RM63NtD2RyWOx4AXL_XHXHHynNLM9Ioc74Waz8-0F11PAYCvZuqsBswHDR_ahXx-qgIK1dkQWzyNoBqdapcc6vlSgreMFQcZ-U1jY2JgVKd4.fgAXN29VHTT2EnnJoETb3w/__results___files/__results___65_1.png)

# Model Performance

In this step, I evaluate the performance of the models using:

* **Accuracy_Score**
* **Recall**
* **Precision**
* **Classification Report**
* **The ROC Curve**




  
