import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_file():
    file_path = filedialog.askopenfilename()
    df = pd.read_csv(file_path)
    return df

def auc(fpr, tpr):
    # Sortowanie wartości False Positive Rate (fpr) rosnąco
    sort_indices = np.argsort(fpr)
    fpr_sorted = fpr[sort_indices]
    tpr_sorted = tpr[sort_indices]

    # Obliczenie AUC metodą prostokątów
    area = 0
    for i in range(len(fpr_sorted) - 1):
        width = fpr_sorted[i + 1] - fpr_sorted[i]
        avg_height = (tpr_sorted[i] + tpr_sorted[i + 1]) / 2
        area += width * avg_height

    return area

def roc_curve(true_labels, pred_probs):
    thresholds = sorted(set(pred_probs), reverse=True)
    tpr_values = [0]
    fpr_values = [0]

    num_positives = sum(true_labels)
    num_negatives = len(true_labels) - num_positives

    for threshold in thresholds:
        predicted_labels = [1 if pred >= threshold else 0 for pred in pred_probs]

        tp = sum((predicted_labels[i] == 1) and (true_labels[i] == 1) for i in range(len(predicted_labels)))
        fp = sum((predicted_labels[i] == 1) and (true_labels[i] == 0) for i in range(len(predicted_labels)))

        tpr = tp / num_positives if num_positives != 0 else 0
        fpr = fp / num_negatives if num_negatives != 0 else 0

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return np.array(fpr_values), np.array(tpr_values), thresholds


def calculate_classification_metrics(df):
    d = {'<=50K': False, '>50K': True}
    df['income'] = df['income'].map(d)
    df['C50_PV'] = df['C50_PV'].map(d)
    df['rf_PV'] = df['rf_PV'].map(d)
    true_labels = df.iloc[:, 0]
    pred_labels_model1 = df.iloc[:, 1]
    pred_prob_model1 = df.iloc[:, 2]
    pred_labels_model2 = df.iloc[:, 3]
    pred_prob_model2 = df.iloc[:, 4]

    tp = sum((true == 1) and (pred == 1) for true, pred in zip(true_labels, pred_labels_model1))
    tn = sum((true == 0) and (pred == 0) for true, pred in zip(true_labels, pred_labels_model1))
    fp = sum((true == 0) and (pred == 1) for true, pred in zip(true_labels, pred_labels_model1))
    fn = sum((true == 1) and (pred == 0) for true, pred in zip(true_labels, pred_labels_model1))

    confusion_matrix_str = f"Confusion Matrix:\n{tp}  {fn}\n{fp}  {tn}"


    # Obliczanie trafności, czułości, swoistości, precyzji i wskaźnika F1
    true_positives = sum((true_labels[i] == 1) and (pred_labels_model1[i] == 1) for i in range(len(true_labels)))
    true_negatives = sum((true_labels[i] == 0) and (pred_labels_model1[i] == 0) for i in range(len(true_labels)))
    actual_positives = sum(true_labels)
    actual_negatives = len(true_labels) - actual_positives
    predicted_positives = sum(pred_labels_model1)
    f1 = 2 * true_positives / (2 * true_positives + predicted_positives + actual_positives)

    true_positives_model2 = sum((true_labels[i] == 1) and (pred_labels_model2[i] == 1) for i in range(len(true_labels)))
    true_negatives_model2 = sum((true_labels[i] == 0) and (pred_labels_model2[i] == 0) for i in range(len(true_labels)))
    actual_positives_model2 = sum(true_labels)
    actual_negatives_model2 = len(true_labels) - actual_positives_model2
    predicted_positives_model2 = sum(pred_labels_model2)
    f1_model2 = 2 * true_positives_model2 / (2 * true_positives_model2 + predicted_positives_model2 + actual_positives_model2)

    accuracy = np.mean(true_labels == pred_labels_model1)
    sensitivity = true_positives / actual_positives if actual_positives != 0 else 0
    specificity = true_negatives / actual_negatives if actual_negatives != 0 else 0
    precision = true_positives / predicted_positives if predicted_positives != 0 else 0

    accuracy_model2 = np.mean(true_labels == pred_labels_model2)
    sensitivity_model2 = true_positives_model2 / actual_positives_model2 if actual_positives_model2 != 0 else 0
    specificity_model2 = true_negatives_model2 / actual_negatives_model2 if actual_negatives_model2 != 0 else 0
    precision_model2 = true_positives_model2 / predicted_positives_model2 if predicted_positives_model2 != 0 else 0


    # Rysowanie krzywej ROC
    fpr_model1, tpr_model1, _ = roc_curve(true_labels, pred_prob_model1)
    fpr_model2, tpr_model2, _ = roc_curve(true_labels, pred_prob_model2)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_model1, tpr_model1, label='Model 1')
    plt.plot(fpr_model2, tpr_model2, label='Model 2')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywa ROC')
    plt.legend()
    plt.show()
    sensitivity_label.config(text=f"Sensitivity model 1: {sensitivity}")
    specificity_label.config(text=f"Specificity model 1: {specificity}")
    precision_label.config(text=f"Precision model 1: {precision}")
    f1_score_label.config(text=f"F1 Score model 1: {f1}")
    sensitivity_label_2.config(text=f"Sensitivity model 2: {sensitivity_model2}")
    specificity_label_2.config(text=f"Specificity model 2: {specificity_model2}")
    precision_label_2.config(text=f"Precision model 2: {precision_model2}")
    f1_score_label_2.config(text=f"F1 Score model 2: {f1_model2}")


    # Obliczanie AUC
    auc_model1 = auc(fpr_model1, tpr_model1)
    auc_model2 = auc(fpr_model2, tpr_model2)

    print(f"Model 1 AUC: {auc_model1}")
    print(f"Model 2 AUC: {auc_model2}")

    # Wyróżnienie lepszego modelu na podstawie AUC
    better_model = "Model 1" if auc_model1 > auc_model2 else "Model 2"
    print(f"Lepszy model: {better_model}")

def calculate_regression_metrics(df):
    true_values = df['rzeczywista']
    pred_values_model1 = df['przewidywana1']
    pred_values_model2 = df['przewidywana2']

    # Obliczanie współczynników regresyjnych (MAE, MAPE, MSE, RMSE)
    mae_model1 = np.mean(np.abs(true_values - pred_values_model1))
    mae_model2 = np.mean(np.abs(true_values - pred_values_model2))

    mape_model1 = np.mean(np.abs((true_values - pred_values_model1) / true_values)) * 100
    mape_model2 = np.mean(np.abs((true_values - pred_values_model2) / true_values)) * 100

    mse_model1 = np.mean((true_values - pred_values_model1)**2)
    mse_model2 = np.mean((true_values - pred_values_model2)**2)

    rmse_model1 = np.sqrt(mse_model1)
    rmse_model2 = np.sqrt(mse_model2)

    print("Model 1:")
    print(f"MAE: {mae_model1}")
    print(f"MAPE: {mape_model1}")
    print(f"MSE: {mse_model1}")
    print(f"RMSE: {rmse_model1}")

    print("\nModel 2:")
    print(f"MAE: {mae_model2}")
    print(f"MAPE: {mape_model2}")
    print(f"MSE: {mse_model2}")
    print(f"RMSE: {rmse_model2}")

    # Tworzenie histogramu różnic między wartościami rzeczywistymi i przewidywanymi przez modele
    plt.figure(figsize=(8, 6))
    plt.hist(true_values - pred_values_model1, bins=50, alpha=0.5, label='Model 1')
    plt.hist(true_values - pred_values_model2, bins=50, alpha=0.5, label='Model 2')
    plt.xlabel('Różnica między wartościami rzeczywistymi a przewidywanymi')
    plt.ylabel('Liczba przypadków')
    plt.title('Histogram różnic między wartościami rzeczywistymi a przewidywanymi')
    plt.legend()
    plt.show()


def choose_model():
    model_type = var.get()
    df = read_csv_file()
    
    if model_type == 'Classification':
        calculate_classification_metrics(df)
    elif model_type == 'Regression':
        calculate_regression_metrics(df)

# Tworzenie interfejsu graficznego
root = tk.Tk()
root.title("Model Evaluation App")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

label = tk.Label(frame, text="Wybierz rodzaj modelu:")
label.pack()

var = tk.StringVar()
var.set("Classification")

classification_radio = tk.Radiobutton(frame, text="Klasyfikacyjny", variable=var, value="Classification")
classification_radio.pack()

regression_radio = tk.Radiobutton(frame, text="Regresyjny", variable=var, value="Regression")
regression_radio.pack()

browse_button = tk.Button(frame, text="Wybierz plik CSV", command=choose_model)
browse_button.pack()

sensitivity_label = tk.Label(frame, text="")
sensitivity_label.pack()

specificity_label = tk.Label(frame, text="")
specificity_label.pack()

precision_label = tk.Label(frame, text="")
precision_label.pack()

f1_score_label = tk.Label(frame, text="")
f1_score_label.pack()

sensitivity_label_2 = tk.Label(frame, text="")
sensitivity_label_2.pack()

specificity_label_2 = tk.Label(frame, text="")
specificity_label_2.pack()

precision_label_2 = tk.Label(frame, text="")
precision_label_2.pack()

f1_score_label_2 = tk.Label(frame, text="")
f1_score_label_2.pack()

root.mainloop()
