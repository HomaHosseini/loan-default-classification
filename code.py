import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import time

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for row in X:
            distances = np.sqrt(np.sum((self.X_train - row) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            predictions.append(np.bincount(nearest_labels).argmax())
        return np.array(predictions)



def load_and_preprocess_data(path):
    data = pd.read_csv(path, low_memory=False) 
    relevant_columns = [
        'int_rate', 'dti', 'revol_util', 'total_acc','emp_length_num',
        'sub_grade', 'term','purpose', 'bad_loans', 'payment_inc_ratio', 'installment',
        'inq_last_6mths','is_inc_v', 'annual_inc', 'pub_rec','revol_bal',
         'open_acc','collections_12_mths_ex_med', 'delinq_2yrs', 'home_ownership', 'earliest_cr_line',
         'mths_since_last_delinq'
    ]
    data = data[relevant_columns]

    data['mths_since_last_delinq'] = data['mths_since_last_delinq'].fillna(0)
    data = data.dropna()
    sub_grade_map = {
        'A1': 0, 'A2': 2, 'A3': 4, 'A4': 6, 'A5': 8,
        'B1': 20, 'B2': 22, 'B3': 24, 'B4': 26, 'B5': 28,
        'C1': 40, 'C2': 42, 'C3': 44, 'C4': 46, 'C5': 48,
        'D1': 60, 'D2': 62, 'D3': 64, 'D4': 66, 'D5': 68,
        'E1': 80, 'E2': 82, 'E3': 84, 'E4': 86, 'E5': 88,
        'F1': 100, 'F2': 102, 'F3': 104, 'F4': 106, 'F5': 108,
        'G1': 120, 'G2': 122, 'G3': 124, 'G4': 126, 'G5': 128
    }
    current_year = 2025
    data['credit_age'] = (current_year - pd.to_datetime(data['earliest_cr_line']).dt.year) * 12
    data['credit_age_since_delinq'] = data['credit_age'] - data['mths_since_last_delinq'].fillna(0)
    data['adjusted_annual_inc'] = data.apply(
    lambda row: row['annual_inc'] if row['is_inc_v'] == 'Verified' 
    else (row['annual_inc'] * 0.95 if row['is_inc_v'] == 'Source Verified' 
    else row['annual_inc'] * 0.75), axis=1)
    data['total_credit_used'] = data['revol_bal'] * (data['revol_util'] / 100)
    data['credit_risk_score'] = data['delinq_2yrs'] + data['pub_rec'] + data['collections_12_mths_ex_med']
    data['loan_burden_score'] = data['installment'] / data['adjusted_annual_inc']
    data['employment_stability'] = data.apply(lambda row: 2 if row['emp_length_num'] == 11 else (1 if row['emp_length_num'] > 4 else 0), axis=1)
    data['employment_stability'] += data['home_ownership'].apply(lambda x: 1 if x in ['OWN', 'MORTGAGE'] else 0)
    data['adjusted_dti'] = data['dti'] + (data['revol_util'] / 100) * data['annual_inc']
    data = data.drop(columns=['emp_length_num', 'home_ownership','annual_inc', 'is_inc_v', 'earliest_cr_line', 'credit_age','mths_since_last_delinq'])
    data = data.drop(columns=['delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med', 'dti', 'installment','revol_bal', 'revol_util'])
    data['sub_grade'] = data['sub_grade'].map(sub_grade_map)

    categorical_columns = ['term','purpose']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  

    class_0 = data[data['bad_loans'] == 0]
    class_1 = data[data['bad_loans'] == 1]
    min_samples = min(len(class_0), len(class_1))
    balanced_data = pd.concat([class_0.sample(min_samples), class_1.sample(min_samples)])
    
    X = balanced_data.drop('bad_loans', axis=1)
    y = balanced_data['bad_loans']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    
    return X_train, X_val_scaled, X_test_scaled, y_train, y_val, y_test, X.columns

def train_decision_tree(X_train, y_train, d):
    dt_model = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_knn(X_train, y_train, k):
    knn_model = KNNClassifier(k)
    knn_model.fit(X_train, y_train)
    return knn_model

def train_adaboost(X_train, y_train, n):
    ab_model = AdaBoostClassifier(n_estimators=n, random_state=42)
    ab_model.fit(X_train, y_train)
    return ab_model

def train_rf(X_train, y_train):
    print("Finding best parameters for Random Forest...")
    param_grid_rf = {
        'n_estimators': [100, 150,  200],
        'max_depth': [5, 10, 20]
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    start_time = time.time()
    grid_search_rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")

    return grid_search_rf.best_estimator_, training_time


def compare_models(dt_accuracy, knn_accuracy, ab_accuracy, rf_accuracy, dt_time, knn_time, ab_time, rf_time, dt_pred_time, knn_pred_time, ab_pred_time, rf_pred_time):
    models = ['Decision Tree', 'KNN', 'Adaboost', 'Random Forest']
    accuracies = [dt_accuracy, knn_accuracy, ab_accuracy, rf_accuracy]
    training_times = [dt_time, knn_time, ab_time, rf_time]
    prediction_times = [dt_pred_time, knn_pred_time, ab_pred_time, rf_pred_time]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    colors = ['blue', 'green', 'red', 'orange']
    plt.bar(models, accuracies, color=colors)
    plt.title('Model Comparison - Test Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.subplot(1, 3, 2)
    plt.bar(models, training_times, color=colors)
    plt.title('Model Comparison - Training Time')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(training_times):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.subplot(1, 3, 3)
    plt.bar(models, prediction_times, color=colors)
    plt.title('Model Comparison - Prediction Time')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(prediction_times):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.show()

def main():
    start_program_time = time.time()
    X_train, X_val, X_test, y_train, y_val, y_test, columns = load_and_preprocess_data('loan_sub.csv')
    #tune hyperparameters
    #knn
    print("Finding best k for KNN...")
    knn_accuracies = {}
    for k in [5, 15, 21]:
        model = train_knn(X_train, y_train, k)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        knn_accuracies[k] = val_accuracy
    best_k = max(knn_accuracies, key=knn_accuracies.get)
    print(f"Best k for KNN: {best_k}")
    #decision_tree
    print("Finding best max_depth for Decision Tree...")
    dt_accuracies = {}
    for d in [5, 7, 9]:
        model = train_decision_tree(X_train, y_train, d)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        dt_accuracies[d] = val_accuracy
    best_d = max(dt_accuracies, key=dt_accuracies.get)
    print(f"Best max_depth for Decision Tree: {best_d}")
    #adaBoost
    print("Finding best n_estimators for AdaBoost...")
    ab_accuracies = {}
    for n in [50, 100, 200, 250]:
        model = train_adaboost(X_train, y_train, n)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        ab_accuracies[n] = val_accuracy
    best_n_ab = max(ab_accuracies, key=ab_accuracies.get)
    print(f"Best n_estimators for AdaBoost: {best_n_ab}")
    #rf ---> tuning of hyperparameter is done in the training function

    start_time = time.time()
    dt_model = train_decision_tree(X_train, y_train, best_d)
    dt_time = time.time() - start_time

    start_time = time.time()
    knn_model = train_knn(X_train, y_train, best_k)
    knn_time = time.time() - start_time

    start_time = time.time()
    ab_model = train_adaboost(X_train, y_train, best_n_ab)
    ab_time = time.time() - start_time

    rf_model, rf_time = train_rf(X_train, y_train)
    

    start_time = time.time()
    dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))
    dt_pred_time = time.time() - start_time

    start_time = time.time()
    knn_test_acc = accuracy_score(y_test, knn_model.predict(X_test))
    knn_pred_time = time.time() - start_time

    start_time = time.time()
    ab_test_acc = accuracy_score(y_test, ab_model.predict(X_test))
    ab_pred_time = time.time() - start_time

    start_time = time.time()
    rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))
    rf_pred_time = time.time() - start_time
    
    compare_models(dt_test_acc, knn_test_acc, ab_test_acc, rf_test_acc, dt_time, knn_time, ab_time, rf_time, dt_pred_time, knn_pred_time, ab_pred_time, rf_pred_time)
    program_time = time.time() - start_program_time
    print(f"total_time :{program_time}")
    plt.figure(figsize=(120, 40))
    plot_tree(dt_model, 
              feature_names=columns, 
              class_names=["0", "1"], 
              filled=True, 
              rounded=True, 
              fontsize=5)
    plt.show()


    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': columns,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='orange')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
    

if __name__ == "__main__":
    main()