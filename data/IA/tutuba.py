# -*- coding: utf-8 -*-
"""Tutuba_training.ipynb - Versão Corrigida

Script corrigido para tratar valores nulos e melhorar a estabilidade do treinamento.
"""

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# Carregar os dados
file_name = "data/dados_unificados_final.csv"
df = pd.read_csv(file_name)

print("Colunas disponíveis:", df.columns.tolist())
print(f"Shape original: {df.shape}")
print(f"Valores nulos por coluna:")
print(df.isnull().sum())

# Limpar dados - remover linhas com valores nulos críticos
print("\nRemovendo linhas com valores nulos...")
df_clean = df.dropna(
    subset=[
        "comportamento",
        "p_forrageio",
        "timestamp",
        "lat",
        "lon",
        "depth_dm",
        "temp_cC",
    ]
)
print(f"Shape após remoção de nulos: {df_clean.shape}")

# Preencher valores nulos restantes com estratégias apropriadas
print("\nPreenchendo valores nulos restantes...")
# Para variáveis ambientais, usar interpolação ou valores médios
df_clean["ssha_ambiente"] = df_clean["ssha_ambiente"].fillna(
    df_clean["ssha_ambiente"].median()
)
df_clean["chlor_a_ambiente"] = df_clean["chlor_a_ambiente"].fillna(
    df_clean["chlor_a_ambiente"].median()
)

# Para acelerômetro, preencher com 0 (sem movimento)
df_clean["acc_x"] = df_clean["acc_x"].fillna(0)
df_clean["acc_y"] = df_clean["acc_y"].fillna(0)
df_clean["acc_z"] = df_clean["acc_z"].fillna(0)

print(f"Valores nulos após limpeza: {df_clean.isnull().sum().sum()}")

# Preparar os dados
# Inputs: timestamp, lat, lon, ssha_ambiente, chlor_a_ambiente
acc_total = np.sqrt(
    df_clean["acc_x"] ** 2 + df_clean["acc_y"] ** 2 + df_clean["acc_z"] ** 2
)

X = df_clean[
    [
        "timestamp",
        "lat",
        "lon",
        "depth_dm",
        "temp_cC",
        "ssha_ambiente",
        "chlor_a_ambiente",
    ]
].copy()
X["acc_total"] = acc_total

# Verificar se há valores infinitos ou muito grandes
print(f"Valores infinitos em X: {np.isinf(X).sum().sum()}")
print(f"Valores muito grandes (>1e10): {(np.abs(X) > 1e10).sum().sum()}")

# Outputs: comportamento (one-hot) + p_forrageio
# Converter comportamento para one-hot encoding
comportamento_mapping = {"busca": 0, "forrageando": 1, "transitando": 2}
y_comportamento = df_clean["comportamento"].map(comportamento_mapping).values
y_comportamento_onehot = tf.keras.utils.to_categorical(y_comportamento, num_classes=3)

# p_forrageio como saída de regressão - garantir que está entre 0 e 1
y_forrageio = np.clip(df_clean["p_forrageio"].values, 0, 1).reshape(-1, 1)

# Combinar as saídas
y = np.concatenate([y_comportamento_onehot, y_forrageio], axis=1)

print(f"Shape dos inputs: {X.shape}")
print(f"Shape dos outputs: {y.shape}")
print(
    f"Distribuição dos comportamentos: {np.unique(y_comportamento, return_counts=True)}"
)
print(f"Range de p_forrageio: {y_forrageio.min():.4f} - {y_forrageio.max():.4f}")

# Normalizar os dados de entrada usando RobustScaler (mais robusto a outliers)
scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)

print("Estatísticas dos dados normalizados:")
print(f"Inputs - Média: {X_scaled.mean(axis=0)}, Std: {X_scaled.std(axis=0)}")
print(f"Range dos inputs normalizados: {X_scaled.min(axis=0)} - {X_scaled.max(axis=0)}")

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y_comportamento
)

print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")


def create_model():
    # Input layer
    inputs = layers.Input(shape=(8,), name="input")

    # Hidden layers
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Hidden layers com arquitetura mais conservadora
    for i in range(4):  # Reduzido de 8 para 4 camadas
        x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

    # Camadas de saída separadas
    # Para comportamento (classificação)
    behavior_output = layers.Dense(3, activation="softmax", name="behavior")(x)

    # Para p_forrageio (regressão)
    forage_output = layers.Dense(1, activation="sigmoid", name="forage")(x)

    # Criar modelo com múltiplas saídas
    model = keras.Model(inputs=inputs, outputs=[behavior_output, forage_output])

    return model


# Criar e compilar o modelo
model = create_model()

# Preparar dados para modelo com múltiplas saídas
y_train_behavior = y_train[:, :3]
y_train_forage = y_train[:, 3]
y_test_behavior = y_test[:, :3]
y_test_forage = y_test[:, 3]

# Compilar o modelo com losses separadas
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001, clipnorm=1.0
    ),  # Gradient clipping
    loss={"behavior": "categorical_crossentropy", "forage": "mse"},
    loss_weights={"behavior": 1.0, "forage": 0.5},
    metrics={"behavior": "accuracy", "forage": "mse"},
)


# Mostrar arquitetura do modelo
model.summary()

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7
)

# Treinar o modelo
print("Iniciando treinamento...")
history = model.fit(
    X_train,
    {"behavior": y_train_behavior, "forage": y_train_forage},
    epochs=200,  # Reduzido de 200 para 100
    batch_size=64,  # Aumentado de 32 para 64
    validation_data=(X_test, {"behavior": y_test_behavior, "forage": y_test_forage}),
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
)

# Avaliar o modelo
print("\nAvaliação final do modelo:")
test_results = model.evaluate(
    X_test, {"behavior": y_test_behavior, "forage": y_test_forage}, verbose=0
)
print(f"Loss total no teste: {test_results[0]:.4f}")
print(f"Loss comportamento: {test_results[1]:.4f}")
print(f"Loss forrageio: {test_results[2]:.4f}")
print(f"Acurácia do comportamento: {test_results[3]:.4f}")
print(f"MSE do p_forrageio: {test_results[4]:.4f}")

# Fazer previsões
y_pred = model.predict(X_test)
y_pred_behavior = y_pred[0]  # Primeira saída (comportamento)
y_pred_forage = y_pred[1].flatten()  # Segunda saída (forrageio)

y_true_behavior = y_test_behavior
y_true_forage = y_test_forage

# Calcular acurácia detalhada
from sklearn.metrics import classification_report, confusion_matrix

# Converter one-hot para labels
y_true_behavior_labels = np.argmax(y_true_behavior, axis=1)
y_pred_behavior_labels = np.argmax(y_pred_behavior, axis=1)

print("\nRelatório de classificação para comportamento:")
print(
    classification_report(
        y_true_behavior_labels,
        y_pred_behavior_labels,
        target_names=["busca", "forrageando", "transitando"],
    )
)

print("\nMatriz de confusão:")
print(confusion_matrix(y_true_behavior_labels, y_pred_behavior_labels))

# Métricas para p_forrageio
from sklearn.metrics import mean_squared_error, r2_score

forage_mse = mean_squared_error(y_true_forage, y_pred_forage)
forage_r2 = r2_score(y_true_forage, y_pred_forage)

print(f"\nMétricas para p_forrageio:")
print(f"MSE: {forage_mse:.4f}")
print(f"R²: {forage_r2:.4f}")

# Visualizar histórico de treinamento
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss durante o treinamento")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()

# Acurácia do comportamento
plt.subplot(1, 3, 2)
plt.plot(history.history["behavior_accuracy"], label="Train Acc")
plt.plot(history.history["val_behavior_accuracy"], label="Val Acc")
plt.title("Acurácia do comportamento")
plt.xlabel("Época")
plt.ylabel("Acurácia")
plt.legend()

# MSE do p_forrageio
plt.subplot(1, 3, 3)
plt.plot(history.history["forage_mse"], label="Train MSE")
plt.plot(history.history["val_forage_mse"], label="Val MSE")
plt.title("MSE do p_forrageio")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.legend()

plt.tight_layout()
plt.show()


# Função para fazer previsões em novos dados
def predict_behavior_and_forrage(model, scaler, timestamp, lat, lon, ssha, chlor_a):
    # Calcular acc_total (assumindo aceleração zero para exemplo)
    acc_total = 0.0

    # Preparar input
    input_data = np.array([[timestamp, lat, lon, ssha, chlor_a, acc_total]])
    input_scaled = scaler.transform(input_data)

    # Fazer previsão
    prediction = model.predict(input_scaled, verbose=0)

    # Processar saída
    behavior_probs = prediction[0][0]  # Primeira saída (comportamento)
    forage_prob = prediction[1][0][0]  # Segunda saída (forrageio)

    # Resultados
    behavior_labels = ["busca", "forrageando", "transitando"]
    predicted_behavior = behavior_labels[np.argmax(behavior_probs)]

    return {
        "comportamento": predicted_behavior,
        "probabilidades_comportamento": dict(zip(behavior_labels, behavior_probs)),
        "p_forrageio": float(forage_prob),
    }


# Salvar o modelo
model.save("data/IA_TREINADA/tubarao_comportamento_model.h5")
print("\nModelo salvo como 'tubarao_comportamento_model.h5'")

# Salvar o scaler
import joblib

joblib.dump(scaler_X, "data/IA_TREINADA/scaler.pkl")
print("Scaler salvo como 'data/IA_TREINADA/scaler.pkl'")
