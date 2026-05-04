# ============================================
# AE-QCRSNet COMPLETE IMPLEMENTATION (FINAL)
# ============================================

# !pip install pandas numpy scikit-learn torch pennylane

import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

import torch
import torch.nn as nn
import pennylane as qml

# ============================================
# 1. LOAD DATA
# ============================================
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

df = pd.concat([df1, df2], ignore_index=True)

# Ensure label column exists
if 'label' not in df.columns:
    raise ValueError("Dataset must contain 'label' column")

y = df['label']
X = df.drop(columns=['label'])

# ============================================
# 2. PREPROCESSING
# ============================================
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Missing values
X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])

if len(cat_cols) > 0:
    X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])
    encoded = OneHotEncoder(sparse=False, handle_unknown='ignore').fit_transform(X[cat_cols])
else:
    encoded = np.empty((len(X), 0))

# Normalize
scaled = MinMaxScaler().fit_transform(X[num_cols])

# Combine
X_processed = np.hstack([scaled, encoded])

# ============================================
# 3. CETFT FEATURE EXTRACTION
# ============================================
def CETFT(window):
    cov = np.cov(window, rowvar=False)

    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    phases = np.angle(eigvecs)

    upper = cov[np.triu_indices_from(cov)]

    return np.concatenate([eigvals, phases.flatten(), upper])

def create_windows(X, y, w=10):
    feats, labels = [], []
    for i in range(0, len(X)-w, w):
        feats.append(CETFT(X[i:i+w]))
        labels.append(y.iloc[i])
    return np.array(feats), np.array(labels)

X_cetft, y_cetft = create_windows(X_processed, y)

# ============================================
# 4. BAOSMO FEATURE SELECTION
# ============================================
def fitness(mask, X, y):
    sel = X[:, mask == 1]
    if sel.shape[1] == 0:
        return 0

    Xtr, Xval, ytr, yval = train_test_split(sel, y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xval)

    return accuracy_score(yval, pred)

def BAOSMO(X, y, pop=8, iters=8):
    dim = X.shape[1]
    population = np.random.randint(0, 2, (pop, dim))

    best_mask = population[0]
    best_score = 0

    for _ in range(iters):
        for i in range(pop):
            score = fitness(population[i], X, y)
            if score > best_score:
                best_score = score
                best_mask = population[i]

        # Orbital-inspired update
        for i in range(pop):
            flip = np.random.rand(dim) < 0.1
            population[i][flip] = 1 - population[i][flip]

    return best_mask

mask = BAOSMO(X_cetft, y_cetft)
X_selected = X_cetft[:, mask == 1]

# ============================================
# 5. QUANTUM MODEL (AQCRSN)
# ============================================
n_qubits = min(6, X_selected.shape[1])
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_layer(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

    for i in range(n_qubits):
        qml.RX(weights[i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class AQCRSN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, n_qubits)
        self.q_weights = nn.Parameter(torch.randn(n_qubits))
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))

        q_out = []
        for i in range(x.shape[0]):
            q_out.append(torch.tensor(quantum_layer(x[i], self.q_weights)))

        q_out = torch.stack(q_out)
        return torch.softmax(self.fc2(q_out), dim=1)

# ============================================
# 6. TRAINING + TIMING
# ============================================
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_cetft, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = AQCRSN(X_selected.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

start_train = time.time()

for epoch in range(10):
    optimizer.zero_grad()
    out = model(X_train)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()

end_train = time.time()
training_time = end_train - start_train

# ============================================
# 7. INFERENCE + METRICS
# ============================================
start_inf = time.time()
with torch.no_grad():
    preds = model(X_test)
end_inf = time.time()

inference_time = (end_inf - start_inf) / len(X_test)

pred_class = torch.argmax(preds, dim=1)

accuracy = accuracy_score(y_test, pred_class)
precision = precision_score(y_test, pred_class)
recall = recall_score(y_test, pred_class)
f1 = f1_score(y_test, pred_class)
auc = roc_auc_score(y_test, preds[:,1])

tn, fp, fn, tp = confusion_matrix(y_test, pred_class).ravel()

FAR = fp / (fp + tn + 1e-10)
FRR = np.sum(mask) / len(mask)

total_params = sum(p.numel() for p in model.parameters())
computational_complexity = total_params

# ============================================
# 8. RESULTS
# ============================================
print("\n========== AE-QCRSNet FINAL RESULTS ==========")
print(f"Accuracy                : {accuracy:.4f}")
print(f"Precision               : {precision:.4f}")
print(f"Recall                  : {recall:.4f}")
print(f"F1 Score                : {f1:.4f}")
print(f"AUC                     : {auc:.4f}")
print(f"False Alarm Rate (FAR)  : {FAR:.4f}")
print(f"Feature Retention (FRR) : {FRR:.4f}")
print(f"Inference Time (sec/sample): {inference_time:.6f}")
print(f"Training Time (sec)     : {training_time:.4f}")
print(f"Total Parameters        : {total_params}")
print(f"Computational Complexity: {computational_complexity}")