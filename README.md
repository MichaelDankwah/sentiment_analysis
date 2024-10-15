# Sentiment Analysis on Customer Reviews for Refurbished Technology

## Project Overview
This project focuses on developing a **Sentiment Analysis Model** to classify customer reviews of **Backmarket** products collected from **Trustpilot** into positive, neutral, or negative sentiments. The insights from this model offer valuable feedback for both consumers and businesses, particularly in the niche market of refurbished technology.

### Key Objectives:
- Scrape customer reviews from **Trustpilot** using web scraping techniques.
- Preprocess and normalize the textual data for effective sentiment classification.
- Develop a robust sentiment classification model, employing various machine learning algorithms.
- Analyze model performance using evaluation metrics and **Receiver Operating Characteristic (ROC) Curves**.

---

## Key Skills and Technologies

### 1. **Web Scraping**
- **Libraries/Tools:** `BeautifulSoup`, `Selenium`
- Scraped customer reviews from **Trustpilot**, specifically targeting **Backmarket** product reviews.
  
### 2. **Data Preprocessing**
- **Text Cleaning:** Removed punctuation, special characters, and stop words to clean raw data.
- **Text Normalization:** Applied **lemmatization** to retain semantic meaning and improve sentiment classification.
- **Vectorization:** Utilized **Term Frequency-Inverse Document Frequency (TF-IDF)** to quantify word importance in reviews.
- **Feature Engineering:** Enriched the dataset with features like sentiment polarity, review length, and word count.

### 3. **Handling Imbalanced Data**
- **Techniques:** Applied **Synthetic Minority Over-sampling Technique (SMOTE)** to address class imbalance, which is critical for sentiment analysis.

### 4. **Machine Learning Models**
- Evaluated multiple machine learning models, including:
  - **Logistic Regression**
  - **Decision Tree**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machines (SVM)**
  - **Naive Bayes** (Bernoulli Naive Bayes)
  
- **Best Model:** The **Bernoulli Naive Bayes** model achieved the highest performance with an accuracy of **94.41%**.

### 5. **Model Optimization and Hyperparameter Tuning**
- **Hyperparameter Tuning:** Optimized the Bernoulli Naive Bayes model using `GridSearchCV` by adjusting key parameters like the smoothing factor (alpha) and binarization threshold.
  
### 6. **Model Evaluation and Performance Metrics**
- **Evaluation Metrics:**
  - Confusion Matrices
  - Classification Reports
  - **ROC Curves:** Detailed analysis of micro-average and macro-average ROC curves
  - **AUC Scores:**
    - Overall sentiment classification AUC: **0.99**
    - Positive and negative sentiment AUC: **0.98**
    - Neutral sentiment AUC: **0.75** (highlighting the challenge of classifying neutral reviews)

### 7. **Insights and Business Impact**
- The insights gained from this sentiment analysis model provide valuable data for businesses to:
  - Understand customer preferences.
  - Tailor marketing strategies.
  - Improve product offerings in the refurbished technology market.

---

## Technologies Used
- **Programming Languages:** Python
- **Libraries:** `BeautifulSoup`, `Selenium`, `scikit-learn`, `pandas`, `numpy`, `nltk`, `matplotlib`, `seaborn`
- **Machine Learning Techniques:** Naive Bayes, Logistic Regression, Decision Trees, KNN, SVM
- **Data Handling:** TF-IDF, SMOTE, GridSearchCV

---

## Project Highlights
- **Achieved 94.41% accuracy** with the Bernoulli Naive Bayes model.
- **Advanced Data Preprocessing:** Text cleaning, lemmatization, and feature engineering for optimal model performance.
- **Comprehensive Model Evaluation:** ROC curves and AUC scores to assess model effectiveness across different sentiment classes.
- **Key Business Insights:** Potential applications in improving product strategies based on customer sentiment.

---

## Future Enhancements
- Improve neutral sentiment classification by refining feature engineering and exploring alternative models like **Deep Learning**.
- Expand the model to include multi-language support for broader analysis of customer reviews.
