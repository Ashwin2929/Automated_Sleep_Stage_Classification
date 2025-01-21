# Automated Sleep Stage Classification

The project explores the use of various advanced machine learning algorithms, such as XGBoost, Random Forest, and Convolutional Neural Networks, to analyze sleep data from polysomnography (PSG). These models were tested on a dataset containing PSG recordings, focusing on reducing data dimensionality through Principal Component Analysis (PCA) and evaluating the impact of including or excluding statistical features on model performance.

The findings demonstrate that models trained on PCA-transformed data without statistical features performed better, suggesting that simpler feature sets allow for more effective learning and generalization across models. XGBoost, in particular, showed significant improvements in accuracy and robustness, confirming its suitability for handling complex datasets in sleep stage classification.

Additionally, the report discusses the importance of spectral features in capturing the physiological signals' frequency characteristics, which are crucial for accurately identifying sleep stages. Future research is encouraged to delve deeper into spectral feature extraction techniques to further enhance model accuracy and performance in clinical settings.
