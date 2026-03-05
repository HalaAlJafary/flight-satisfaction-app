✈️ FlightVerdict
Predicting Passenger Satisfaction using Machine Learning
FlightVerdict is an end-to-end Machine Learning application designed to analyze and predict passenger satisfaction levels. This project covers the entire data science lifecycle—from Exploratory Data Analysis (EDA) and preprocessing to building a predictive model and deploying it as an interactive web application.
📊 Data Journey
1️⃣ Data Preparation
 * Data Integration: Merged training and testing datasets to increase the sample size and ensure robust evaluation using Cross-Validation.
 * Column Organization: Performed column renaming to ensure consistency and improve code readability.
 * Exploratory Analysis: Utilized Pandas functions (head, info, shape, nunique, describe) to understand data distribution, identify data types, and detect missing values.
2️⃣ Exploratory Data Analysis (EDA) & Visualization
Applied various visualization techniques to extract key insights:
 * Histograms: Analyzed numerical distributions and detected Skewness in delay-related features.
 * Scatter Plots: Confirmed a strong positive correlation between "Departure Delay" and "Arrival Delay."
 * Pie Charts: Verified Data Balance between satisfaction and dissatisfaction classes.
 * Box Plots: Conducted Outlier Analysis; decided to retain outliers as they represent significant real-world scenarios that influence predictions.
 * Heatmaps: Performed Correlation Analysis to guide the Feature Selection process.
🤖 Modeling & Evaluation
Multiple algorithms were tested and compared to ensure peak performance:
 * Tested Models: Linear Regression, Logistic Regression, Decision Tree, SVM, and Random Forest.
 * The Winning Model: Random Forest was selected as the final model due to its superior accuracy and stability.
 * Evaluation Metrics: The model was assessed using Accuracy, Precision, Recall, F1-Score, and AUC.
 * Performance Analysis: Leveraged Confusion Matrix and ROC Curves to ensure high discriminative power and minimize the risk of overfitting.
🚀 Deployment & Web Integration
 * Model Persistence: The final model was exported using the Joblib library for seamless loading.
 * Web Application: Developed an interactive UI using Streamlit, allowing users to:
   * Input flight details and receive an instant FlightVerdict (Satisfaction Status).
   * View the Prediction Probability (confidence level).
   * Explore Feature Importance to understand the key factors driving the model's decision.
 * Hosting: The project is hosted on GitHub and deployed via Streamlit Cloud, making it accessible via a live URL.
🛠️ Tech Stack
 * Language: Python
 * Data Manipulation: Pandas, NumPy
 * ML & Visualization: Scikit-Learn, Matplotlib, Seaborn
 * Deployment: Joblib, Streamlit
🔗 Live Demo: https://flight-satisfaction-app-wgj7ayqxxreodjjwy5poxa.streamlit.app/
