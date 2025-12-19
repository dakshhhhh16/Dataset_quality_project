# Dataset Quality Experiments

Final Conclusion

In this project, a series of controlled experiments were conducted using the breast cancer dataset to study how data quality issues and decision policies affect machine learning model performance. A baseline model was first established to provide a reliable reference point. Data quality was then systematically degraded through the introduction of missing values and class imbalance, revealing that model performance is highly sensitive to dataset integrity and that accuracy alone can be a misleading evaluation metric in imbalanced, high-stakes settings.

Attempts to mitigate class imbalance using class weighting demonstrated that corrective techniques do not always yield improvements, particularly when the dataset is already well-separated. Further analysis using prediction probabilities and decision threshold adjustment showed that recall–accuracy trade-offs can be explicitly controlled without altering model architecture, emphasizing the importance of post-model decision policies.

Overall, this work highlights that effective machine learning engineering depends less on model complexity and more on careful data handling, metric selection, and systematic experimentation. Understanding model failure modes and trade-offs is critical for deploying reliable ML systems in real-world applications.