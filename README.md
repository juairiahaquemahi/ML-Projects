**MLP Classifier: Adult Census Income Prediction**


This project focuses on building and understanding a Multi-Layer Perceptron (MLP) Classifier to predict whether an individual earns more than $50,000 per year using the Adult Census Income dataset. The objective was not only to train a high-performing model but also to understand how neural networks like MLPs learn from complex data, and how they compare to more traditional machine learning algorithms such as Logistic Regression and Random Forest.

The Adult dataset includes demographic and occupational information such as education level, working hours, and marital status. These features are highly interactive, making this dataset an excellent ground to explore how nonlinear models perform against linear ones.

The project begins with careful data preprocessing. The dataset is loaded and cleaned to remove extreme outliers and missing entries. Categorical variables are converted into numerical form using one-hot encoding, and numerical features are standardized through feature scaling. This step ensures that the neural network can process all features on a comparable scale and avoids bias in learning.

The cleaned data is then split into three sets: 70 percent for training, 15 percent for validation, and 15 percent for testing. The test set remains unseen until the very end to ensure that the model’s generalization ability can be evaluated fairly.

After preprocessing, a hyperparameter tuning phase is carried out using RandomizedSearchCV. This allows the model to find an optimal combination of hidden layer sizes, activation functions, learning rates, and solvers without overfitting. Early stopping is also used so that the training halts once the model stops improving, ensuring it does not learn noise from the data.

Once the optimized MLP model is trained, it is compared with two other algorithms — Logistic Regression and Random Forest. Logistic Regression serves as a linear baseline, representing the simplest and most interpretable model. It helps identify whether the neural network is truly capturing deeper patterns or merely adding unnecessary complexity. Random Forest, on the other hand, is a robust nonlinear ensemble model. It is capable of modeling interactions between features and serves as a strong benchmark before moving into deep learning territory.

The results show that the MLP Classifier achieves higher accuracy and ROC-AUC scores than both Logistic Regression and Random Forest. This suggests that the MLP successfully learns continuous, nonlinear relationships between socioeconomic factors and income. While Logistic Regression performs well under the assumption of linearity, it cannot model the subtle feature dependencies present in this dataset. Random Forest performs strongly too but tends to produce piecewise decision boundaries, whereas the MLP learns a smoother and more generalizable surface.

In terms of performance, the MLP achieved an accuracy of around 84 percent and an ROC-AUC close to 0.88. Logistic Regression and Random Forest scored slightly lower, around 0.82 and 0.86 respectively. These results confirm that the MLP is not only capable of learning more complex relationships but also avoids overfitting due to careful regularization and early stopping.

Through this project, I learned the importance of proper data preprocessing and feature scaling when working with neural networks. I also realized that while neural networks can be powerful, their true strength lies in the quality of data and the design of the learning process. Comparing the MLP with Logistic Regression and Random Forest allowed me to understand different model families — linear, tree-based, and neural — and appreciate how each interprets the same data differently.

This work was done entirely in Python using libraries such as NumPy, Pandas, and Scikit-learn, and executed in Google Colab.

In the future, I plan to experiment with deeper architectures, dropout regularization, and batch normalization to further improve generalization. I am also interested in using explainable AI tools such as SHAP or LIME to understand how individual features influence model predictions.

The dataset was sourced from the UCI Machine Learning Repository, and the implementation relied on standard Python libraries. References include Dua and Graff (2019), *UCI Machine Learning Repository: Adult Dataset*, and Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research*.

This project was created as part of my coursework to understand and explain the MLP Classifier algorithm. It demonstrates not just how to train a neural model but how to evaluate it critically against established machine learning approaches, ensuring that the algorithm truly learns patterns and not just numbers.

