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
 
 
  ![](https://www.kaggleusercontent.com/kf/44797707/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Jk3GGemFCYYGjCtopgNLCA.5exlAdKvUEdc2V_ONNVHv5s0vkE0uvwIe7LWwx2aUSNKHPfzfeTilXIehjB9KoeMRzHbiOOw-oHXgsQT34B5IYtaxTmw8aOtsVX3h4JAf4okqYlzcAuf-e35-2n0F1QyxLbR8Yz4TOPHxOSiNDogGmvXo39S_c-CsWQRT8PdLV7V8io7XutAcF5oUMeI5qgw-nMjPmHn06pZtP9458cy7B9KVEMkGM9JcSRGnVN38DDJqJoDW-PDyoaVHqKsA47Y_q2DhgeVyDSL3M5aZMCxwqvIxuQo3KRZD3GfxEO_Mcq-7ZLaSYIQSts5HRU48K1VBnbPlFlQQqg7Ji8OwOV4oUIrTrC0j2wo8JmwtiuUHW58COexGzbKog_1hZfaONvhmzwnFFBahfh3mKYpr-yDXUi7g_sbiqX_CJZb9VkMoi6pXIXzt9qz1DCGYTlG07AUJut3UuKBXydjBr10KH8yLIznoQdJbRSf8mIpzD6h_Tzd7o3Zy3BiVyKO8wPceC1slaEyowoSgMTWnAznQdNwp1UceV-9YAOzv7m_3SMEddJpzY5DwqjS6XKw9CPCj_qimIef6b2j-FhNQfA-EFgIelo9nHVjxEaq3rAh-lCnz4MFJ___Q18v6S4JdNFDIttP2zpFLwRmjmzRJGNUltMZr9cTR-Ex9hDOIYc1HpXStPI._CZQXMFUVoiVdwXV_qummQ/__results___files/__results___26_0.png)
 
  * plotting the highly correlated pairs
  
  ![](https://www.kaggleusercontent.com/kf/44797707/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Jk3GGemFCYYGjCtopgNLCA.5exlAdKvUEdc2V_ONNVHv5s0vkE0uvwIe7LWwx2aUSNKHPfzfeTilXIehjB9KoeMRzHbiOOw-oHXgsQT34B5IYtaxTmw8aOtsVX3h4JAf4okqYlzcAuf-e35-2n0F1QyxLbR8Yz4TOPHxOSiNDogGmvXo39S_c-CsWQRT8PdLV7V8io7XutAcF5oUMeI5qgw-nMjPmHn06pZtP9458cy7B9KVEMkGM9JcSRGnVN38DDJqJoDW-PDyoaVHqKsA47Y_q2DhgeVyDSL3M5aZMCxwqvIxuQo3KRZD3GfxEO_Mcq-7ZLaSYIQSts5HRU48K1VBnbPlFlQQqg7Ji8OwOV4oUIrTrC0j2wo8JmwtiuUHW58COexGzbKog_1hZfaONvhmzwnFFBahfh3mKYpr-yDXUi7g_sbiqX_CJZb9VkMoi6pXIXzt9qz1DCGYTlG07AUJut3UuKBXydjBr10KH8yLIznoQdJbRSf8mIpzD6h_Tzd7o3Zy3BiVyKO8wPceC1slaEyowoSgMTWnAznQdNwp1UceV-9YAOzv7m_3SMEddJpzY5DwqjS6XKw9CPCj_qimIef6b2j-FhNQfA-EFgIelo9nHVjxEaq3rAh-lCnz4MFJ___Q18v6S4JdNFDIttP2zpFLwRmjmzRJGNUltMZr9cTR-Ex9hDOIYc1HpXStPI._CZQXMFUVoiVdwXV_qummQ/__results___files/__results___28_0.png)
  
  # Model Building
  
  * In this section,  I tried different models and evaluate them using the Accuracy_Score:
  
   + **Logistic Regression** 
   + **Gradient Boosting Classifier**
   + **Random Forest Classifier**
   + **Decision Tree Classifier**
   + **Kneighbours Classifier**
   + **XGB Classifier**
   + **Supportr vector Classifier**


![](https://www.kaggleusercontent.com/kf/44797707/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Jk3GGemFCYYGjCtopgNLCA.5exlAdKvUEdc2V_ONNVHv5s0vkE0uvwIe7LWwx2aUSNKHPfzfeTilXIehjB9KoeMRzHbiOOw-oHXgsQT34B5IYtaxTmw8aOtsVX3h4JAf4okqYlzcAuf-e35-2n0F1QyxLbR8Yz4TOPHxOSiNDogGmvXo39S_c-CsWQRT8PdLV7V8io7XutAcF5oUMeI5qgw-nMjPmHn06pZtP9458cy7B9KVEMkGM9JcSRGnVN38DDJqJoDW-PDyoaVHqKsA47Y_q2DhgeVyDSL3M5aZMCxwqvIxuQo3KRZD3GfxEO_Mcq-7ZLaSYIQSts5HRU48K1VBnbPlFlQQqg7Ji8OwOV4oUIrTrC0j2wo8JmwtiuUHW58COexGzbKog_1hZfaONvhmzwnFFBahfh3mKYpr-yDXUi7g_sbiqX_CJZb9VkMoi6pXIXzt9qz1DCGYTlG07AUJut3UuKBXydjBr10KH8yLIznoQdJbRSf8mIpzD6h_Tzd7o3Zy3BiVyKO8wPceC1slaEyowoSgMTWnAznQdNwp1UceV-9YAOzv7m_3SMEddJpzY5DwqjS6XKw9CPCj_qimIef6b2j-FhNQfA-EFgIelo9nHVjxEaq3rAh-lCnz4MFJ___Q18v6S4JdNFDIttP2zpFLwRmjmzRJGNUltMZr9cTR-Ex9hDOIYc1HpXStPI._CZQXMFUVoiVdwXV_qummQ/__results___files/__results___65_1.png)

# Model Performance

In this step, i evaluate the performance of the models using:

* **Accuracy_Score**
* **Recall**
* **Precision**
* **Classification Report**
* **The ROC Curve**


  
