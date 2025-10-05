# -*- coding: utf-8 -*-
import json
import os

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm  # barra de progresso

# ------------------------------
# Carregar modelo e scaler com custom_objects
# ------------------------------
custom_objects = {"mse": keras.metrics.MeanSquaredError()}

model = keras.models.load_model(
    "data/IA/IA_TREINADA/tubarao_comportamento_model.h5", custom_objects=custom_objects
)
scaler = joblib.load("data/IA/IA_TREINADA/scaler.pkl")


# ------------------------------
# Função de inferência para uma linha
# ------------------------------
def predict_behavior_and_forrage(model, scaler, input_data):
    acc_total = np.sqrt(
        input_data["acc_x"] ** 2 + input_data["acc_y"] ** 2 + input_data["acc_z"] ** 2
    )
    arr = np.array(
        [
            [
                input_data["timestamp"],
                input_data["lat"],
                input_data["lon"],
                input_data["depth_dm"],
                input_data["temp_cC"],
                input_data["ssha_ambiente"],
                input_data["chlor_a_ambiente"],
                acc_total,
            ]
        ]
    )
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled, verbose=0)
    behavior_probs = pred[0][0]
    forage_prob = pred[1][0][0]
    behavior_labels = ["busca", "forrageando", "transitando"]
    predicted_behavior = behavior_labels[np.argmax(behavior_probs)]
    return {
        "inputs": input_data,
        "outputs": {
            "comportamento": predicted_behavior,
            "probabilidades_comportamento": dict(
                zip(behavior_labels, [float(x) for x in behavior_probs])
            ),
            "p_forrageio": float(forage_prob),
        },
    }


# ------------------------------
# Carregar CSV de entrada
# ------------------------------
input_csv = "data/IA/IA_TREINADA/dados_unificados_final_inferencia.csv"
df_input = pd.read_csv(input_csv)

# ------------------------------
# Preencher valores nulos automaticamente
# ------------------------------
df_input["ssha_ambiente"] = df_input["ssha_ambiente"].fillna(
    df_input["ssha_ambiente"].median()
)
df_input["chlor_a_ambiente"] = df_input["chlor_a_ambiente"].fillna(
    df_input["chlor_a_ambiente"].median()
)
df_input["acc_x"] = df_input["acc_x"].fillna(0)
df_input["acc_y"] = df_input["acc_y"].fillna(0)
df_input["acc_z"] = df_input["acc_z"].fillna(0)
df_input["depth_dm"] = df_input["depth_dm"].fillna(df_input["depth_dm"].median())
df_input["temp_cC"] = df_input["temp_cC"].fillna(df_input["temp_cC"].median())
df_input["timestamp"] = df_input["timestamp"].fillna(method="ffill")
df_input["lat"] = df_input["lat"].fillna(method="ffill")
df_input["lon"] = df_input["lon"].fillna(method="ffill")

# ------------------------------
# Executar inferência com status
# ------------------------------
results = []
for idx, row in enumerate(tqdm(df_input.to_dict(orient="records"), desc="Inferência")):
    result = predict_behavior_and_forrage(model, scaler, row)
    results.append(result)
    # opcional: print a cada 100 registros
    if (idx + 1) % 100 == 0:
        print(f"{idx + 1} / {len(df_input)} registros processados...")

# ------------------------------
# Salvar resultados em JSON
# ------------------------------
# Criar diretório de saída se não existir
output_dir = "data/IA/IA_TREINADA/OUTPUT"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "inferencia_result.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Inferência concluída! {len(results)} registros salvos em '{output_file}'")
