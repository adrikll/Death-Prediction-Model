import os
import re
import numpy as np
import torch
import pandas as pd
import neurokit2 as nk
from torch.utils.data import Dataset
import json

class ChagasDataset(Dataset):
    def __init__(self, data_dir, patients_df, weights=None, return_ids=True, max_minutes=27):
        self.data_dir       = data_dir
        self.patients_df    = self._filter_patients(patients_df)
        self.weights        = weights or {}
        self.return_ids     = return_ids
        self.max_minutes    = max_minutes
        self.sampling_rate  = 480  

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.txt')
        ])

        sample_signal = np.loadtxt(self.files[0], dtype=np.float32)
        self.feature_names = self._extract_ecg_features(sample_signal).index.tolist()

    def _filter_patients(self, df):
        c1 = df[(df['Obito_MS'] == 1) & (df['Time'] < 5)]
        c0 = df[(df['Obito_MS'] == 0) | (df['Time'] == 5)]
        return pd.concat([c1, c0])

    def _extract_ecg_features(self, signal: np.ndarray) -> pd.Series:
        max_samples = int(self.max_minutes * 60 * self.sampling_rate)

        if len(signal) < max_samples:
            raise ValueError(f"Sinal muito curto: tem {len(signal)/self.sampling_rate/60:.1f} min, necessário >= {self.max_minutes} min.")
        
        segment = signal[:max_samples]

        #limpeza e detecção de picos com método pantompkins1985
        clean_ecg = nk.ecg_clean(segment, sampling_rate=self.sampling_rate, method="pantompkins1985")
        _, info = nk.ecg_peaks(clean_ecg, sampling_rate=self.sampling_rate, method="pantompkins1985")

        try:
            hrv_time = nk.hrv_time(info['ECG_R_Peaks'], sampling_rate=self.sampling_rate, show=False)
        except Exception as e:
            print("Erro na extração de feature: HRV_TIME")
            print(e)
            hrv_time = pd.DataFrame()

        try:
            hrv_freq = nk.hrv_frequency(info['ECG_R_Peaks'], sampling_rate=self.sampling_rate, show=False)
        except Exception as e:
            print("Erro na extração de feature: HRV_FREQUENCY")
            print(e)
            hrv_freq = pd.DataFrame()

        try:
            hrv_nonlinear = nk.hrv_nonlinear(info['ECG_R_Peaks'], sampling_rate=self.sampling_rate, show=False)
        except Exception as e:
            print("Erro na extração de feature: HRV_NONLINEAR")
            print(e)
            hrv_nonlinear = pd.DataFrame()

        
        if 'HRV_ULF' in hrv_freq.columns:
            hrv_freq = hrv_freq.drop(columns=['HRV_ULF'])
        

        features_df = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

        if features_df.empty:
            raise ValueError("Nenhuma feature extraída para esse sinal.")

        return features_df.iloc[0]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        basename = os.path.basename(path)
        m = re.match(r'sinal(\d+)\.txt', basename)
        if not m:
            raise ValueError(f"Arquivo inválido: {basename}")
        patient_id = int(m.group(1))

        info = self.patients_df[self.patients_df['ID'] == patient_id]
        if info.empty:
            raise ValueError(f"ID {patient_id} não está na tabela de pacientes.")
        obito = info['Obito_MS'].iloc[0]
        tempo = info['Time'].iloc[0]
        event = 1 if (obito == 1 and tempo < 5) else 0

        signal = np.loadtxt(path, dtype=np.float32)
        feats = self._extract_ecg_features(signal)
        x = torch.tensor(feats.values, dtype=torch.float32)
        y = torch.tensor([event], dtype=torch.float32)
        w = torch.tensor([self.weights.get(str(patient_id), 1.0)], dtype=torch.float32)

        out = [x, y, w]
        if self.return_ids:
            out.append(patient_id)
        return tuple(out)


def save_features(dataset, output_file):
    header_written = os.path.exists(output_file)

    for idx in range(len(dataset)):
        try:
            features, event, weight, patient_id = dataset[idx]
            patient_info = dataset.patients_df[dataset.patients_df['ID'] == patient_id]

            if patient_info.empty:
                continue

            obito = patient_info['Obito_MS'].values[0]
            tempo = patient_info['Time'].values[0]

            feature_dict = {name: features[i].item() for i, name in enumerate(dataset.feature_names)}
            feature_dict.update({
                'event': event.item(),
                'weight': weight.item(),
                'patient_id': patient_id,
                'Obito_MS': obito,
                'Time': tempo,
                'class_1': event.item()  
            })

            df_row = pd.DataFrame([feature_dict])

            col_order = ['patient_id', 'weight', 'Obito_MS', 'Time', 'class_1'] + \
                        dataset.feature_names + ['event']
            df_row = df_row[col_order]

            df_row.to_csv(output_file, mode='a', header=not header_written, index=False)
            header_written = True

            print(f"Paciente {patient_id} ok.")

        except Exception as e:
            print(f"Erro ao processar paciente {patient_id} (idx={idx}): {e}")


if __name__ == "__main__":
    from utils import compute_chagas_weights

    df_chagas = pd.read_excel('chagas_idades.xlsx')
    compute_chagas_weights()
    with open('chagas_weights.json', 'r') as f:
        weights = json.load(f)

    ds = ChagasDataset('base_chagas', df_chagas, weights=weights, return_ids=True)

    save_features(ds, 'features_chagas.csv')

   



