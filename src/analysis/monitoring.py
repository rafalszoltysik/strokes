#!/usr/bin/env python3

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MonitorModelu:
    
    def __init__(self, model, nazwa_modelu: str = "StrokePredictor", prog_driftu: float = 0.05):
        self.model = model
        self.nazwa_modelu = nazwa_modelu
        self.historia_monitoringu = []
        self.bazowa_wydajnosc = None
        self.prog_driftu = prog_driftu  # Próg driftu z konfiguracji
        
    def oblicz_metryki_modelu(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray) -> Dict[str, float]:
        metryki = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc_score': roc_auc_score(y_true, y_pred_proba),
            'timestamp': datetime.now().isoformat()
        }
        return metryki
    
    def wykryj_drift_danych(self, X_new: np.ndarray, X_baseline: np.ndarray) -> Dict[str, Any]:
        logger.info("Sprawdzanie driftu danych")
        
        wyniki_driftu = {}
        
        # Porównanie statystyk
        statystyki_bazowe = {
            'mean': np.mean(X_baseline, axis=0),
            'std': np.std(X_baseline, axis=0),
            'min': np.min(X_baseline, axis=0),
            'max': np.max(X_baseline, axis=0)
        }
        
        statystyki_nowe = {
            'mean': np.mean(X_new, axis=0),
            'std': np.std(X_new, axis=0),
            'min': np.min(X_new, axis=0),
            'max': np.max(X_new, axis=0)
        }
        
        # Obliczenie różnic z obsługą dzielenia przez zero
        # Unikamy dzielenia przez zero używając np.where
        drift_sredniej = np.where(
            np.abs(statystyki_bazowe['mean']) > 1e-10,  # Jeśli baseline nie jest bliski zero
            np.abs(statystyki_nowe['mean'] - statystyki_bazowe['mean']) / np.abs(statystyki_bazowe['mean']),
            0.0  # Jeśli baseline jest bliski zero, drift = 0
        )
        
        drift_std = np.where(
            np.abs(statystyki_bazowe['std']) > 1e-10,  # Jeśli baseline nie jest bliski zero
            np.abs(statystyki_nowe['std'] - statystyki_bazowe['std']) / np.abs(statystyki_bazowe['std']),
            0.0  # Jeśli baseline jest bliski zero, drift = 0
        )
        
        wyniki_driftu = {
            'mean_drift': drift_sredniej.tolist(),
            'std_drift': drift_std.tolist(),
            'max_mean_drift': float(np.max(drift_sredniej)),
            'max_std_drift': float(np.max(drift_std)),
            'drift_detected': np.max(drift_sredniej) > self.prog_driftu or np.max(drift_std) > self.prog_driftu,
            'timestamp': datetime.now().isoformat()
        }
        
        if wyniki_driftu['drift_detected']:
            logger.warning(f"Wykryto drift danych: max_mean_drift={wyniki_driftu['max_mean_drift']:.3f}, max_std_drift={wyniki_driftu['max_std_drift']:.3f}")
            logger.info(f"Próg driftu: {self.prog_driftu}")
            logger.info(f"Liczba cech z driftem mean: {np.sum(drift_sredniej > self.prog_driftu)}")
            logger.info(f"Liczba cech z driftem std: {np.sum(drift_std > self.prog_driftu)}")
        else:
            logger.info("Nie wykryto znaczącego driftu danych")
            logger.info(f"Max mean drift: {wyniki_driftu['max_mean_drift']:.6f}, Max std drift: {wyniki_driftu['max_std_drift']:.6f}")
        
        return wyniki_driftu
    
    def wykryj_drift_wydajnosci(self, aktualne_metryki: Dict[str, float]) -> Dict[str, Any]:
        logger.info("Sprawdzanie driftu wydajności")
        
        if self.bazowa_wydajnosc is None:
            self.bazowa_wydajnosc = aktualne_metryki
            logger.info("Ustawiono wartość bazową wydajności")
            return {'drift_detected': False, 'message': 'Wartość bazowa ustawiona'}
        
        drift_wydajnosci = {}
        
        for metryka in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']:
            if metryka in self.bazowa_wydajnosc and metryka in aktualne_metryki:
                wartosc_bazowa = self.bazowa_wydajnosc[metryka]
                wartosc_aktualna = aktualne_metryki[metryka]
                drift = (wartosc_bazowa - wartosc_aktualna) / wartosc_bazowa
                drift_wydajnosci[metryka] = {
                    'baseline': wartosc_bazowa,
                    'current': wartosc_aktualna,
                    'drift': drift,
                    'significant_drift': drift > self.prog_driftu
                }
        
        znaczący_drift = any(drift['significant_drift'] for drift in drift_wydajnosci.values())
        
        wyniki_driftu = {
            'performance_drift': drift_wydajnosci,
            'significant_drift': znaczący_drift,
            'timestamp': datetime.now().isoformat()
        }
        
        if znaczący_drift:
            logger.warning("Wykryto znaczący drift wydajności!")
            for metryka, info_driftu in drift_wydajnosci.items():
                if info_driftu['significant_drift']:
                    logger.warning(f"{metryka}: {info_driftu['baseline']:.3f} -> {info_driftu['current']:.3f} (drift: {info_driftu['drift']:.3f})")
        else:
            logger.info("Wydajność modelu pozostaje stabilna")
        
        return wyniki_driftu
    
    def wykryj_wzorce_halucynacji(self, y_pred_proba: np.ndarray, 
                                    prog_pewnosci: float = 0.8) -> Dict[str, Any]:
        logger.info("Sprawdzanie wzorców halucynacji")
        
        # Analiza rozkładu pewności predykcji
        wysoka_pewnosc = np.sum(y_pred_proba > prog_pewnosci)
        niska_pewnosc = np.sum(y_pred_proba < (1 - prog_pewnosci))
        srednia_pewnosc = len(y_pred_proba) - wysoka_pewnosc - niska_pewnosc
        
        # Wykrywanie ekstremalnych wartości pewności
        ekstremalnie_wysoka = np.sum(y_pred_proba > 0.95)
        ekstremalnie_niska = np.sum(y_pred_proba < 0.05)
        
        # Analiza wariancji pewności
        wariancja_pewnosci = np.var(y_pred_proba)
        
        wyniki_halucynacji = {
            'confidence_distribution': {
                'high_confidence': int(wysoka_pewnosc),
                'medium_confidence': int(srednia_pewnosc),
                'low_confidence': int(niska_pewnosc),
                'extreme_high': int(ekstremalnie_wysoka),
                'extreme_low': int(ekstremalnie_niska)
            },
            'confidence_variance': float(wariancja_pewnosci),
            'potential_hallucination': ekstremalnie_wysoka > len(y_pred_proba) * 0.1 or ekstremalnie_niska > len(y_pred_proba) * 0.1,
            'timestamp': datetime.now().isoformat()
        }
        
        if wyniki_halucynacji['potential_hallucination']:
            logger.warning("Wykryto potencjalne wzorce halucynacji!")
            logger.warning(f"Ekstremalnie wysokie pewności: {ekstremalnie_wysoka}, ekstremalnie niskie: {ekstremalnie_niska}")
        else:
            logger.info("Nie wykryto wzorców halucynacji")
        
        return wyniki_halucynacji
    
    def analizuj_spojnosc_predykcji(self, X: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_proba: np.ndarray) -> Dict[str, Any]:
        logger.info("Analiza spójności predykcji")
        
        # Analiza rozkładu predykcji
        rozklad_predykcji = {
            'positive_predictions': int(np.sum(y_pred == 1)),
            'negative_predictions': int(np.sum(y_pred == 0)),
            'positive_rate': float(np.mean(y_pred))
        }
        
        # Analiza pewności predykcji
        statystyki_pewnosci = {
            'mean_confidence': float(np.mean(y_pred_proba)),
            'std_confidence': float(np.std(y_pred_proba)),
            'min_confidence': float(np.min(y_pred_proba)),
            'max_confidence': float(np.max(y_pred_proba))
        }
        
        # Wykrywanie anomalii w predykcjach
        prog_anomalii = 2.0  # 2 odchylenia standardowe
        z_scores_pewnosci = np.abs((y_pred_proba - statystyki_pewnosci['mean_confidence']) / statystyki_pewnosci['std_confidence'])
        anomalie = np.sum(z_scores_pewnosci > prog_anomalii)
        
        wyniki_spojnosci = {
            'prediction_distribution': rozklad_predykcji,
            'confidence_stats': statystyki_pewnosci,
            'anomalies_detected': int(anomalie),
            'anomaly_rate': float(anomalie / len(y_pred_proba)),
            'timestamp': datetime.now().isoformat()
        }
        
        if wyniki_spojnosci['anomaly_rate'] > 0.05:  # 5% anomalii
            logger.warning(f"Wykryto {anomalie} anomalii w predykcjach ({wyniki_spojnosci['anomaly_rate']:.1%})")
        else:
            logger.info("Predykcje są spójne")
        
        return wyniki_spojnosci
    
    def utworz_dashboard_monitoringu(self, wyniki_monitoringu: Dict[str, Any]) -> None:
        logger.info("Tworzenie dashboardu monitoringu")
        
        fig, osie = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dashboard Monitoringu Modelu - {self.nazwa_modelu}', fontsize=16, fontweight='bold')
        
        # 1. Historia wydajności
        if 'performance_history' in wyniki_monitoringu:
            historia = wyniki_monitoringu['performance_history']
            metryki = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            
            for i, metryka in enumerate(metryki):
                if metryka in historia:
                    osie[0,0].plot(historia[metryka], label=metryka, marker='o')
            
            osie[0,0].set_title('Historia Wydajności Modelu')
            osie[0,0].set_xlabel('Czas')
            osie[0,0].set_ylabel('Wartość metryki')
            osie[0,0].legend()
            osie[0,0].grid(True, alpha=0.3)
        
        # 2. Rozkład pewności predykcji
        if 'hallucination_analysis' in wyniki_monitoringu:
            halucynacje = wyniki_monitoringu['hallucination_analysis']
            rozklad_pewnosci = halucynacje['confidence_distribution']
            
            kategorie = ['Niska', 'Średnia', 'Wysoka', 'Ekstremalnie wysoka']
            wartosci = [rozklad_pewnosci['low_confidence'], rozklad_pewnosci['medium_confidence'], 
                    rozklad_pewnosci['high_confidence'], rozklad_pewnosci['extreme_high']]
            
            osie[0,1].bar(kategorie, wartosci, color=['red', 'yellow', 'green', 'purple'], alpha=0.7)
            osie[0,1].set_title('Rozkład Pewności Predykcji')
            osie[0,1].set_ylabel('Liczba predykcji')
            osie[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Analiza driftu danych
        if 'data_drift' in wyniki_monitoringu:
            drift = wyniki_monitoringu['data_drift']
            if 'mean_drift' in drift:
                drift_sredniej = np.array(drift['mean_drift'])
                osie[1,0].bar(range(len(drift_sredniej)), drift_sredniej, alpha=0.7, color='orange')
                osie[1,0].set_title('Drift Średnich Cech')
                osie[1,0].set_xlabel('Indeks cechy')
                osie[1,0].set_ylabel('Wielkość driftu')
                osie[1,0].axhline(y=self.prog_driftu, color='red', linestyle='--', label='Próg alarmu')
                osie[1,0].legend()
        
        # 4. Status monitoringu
        osie[1,1].axis('off')
        
        # Tworzenie tabeli statusu
        dane_statusu = []
        if 'performance_drift' in wyniki_monitoringu:
            dane_statusu.append(['Drift Wydajności', 'OK' if not wyniki_monitoringu.get('significant_drift', False) else 'OSTRZEŻENIE'])
        if 'data_drift' in wyniki_monitoringu:
            dane_statusu.append(['Drift Danych', 'OK' if not wyniki_monitoringu['data_drift']['drift_detected'] else 'OSTRZEŻENIE'])
        if 'hallucination_analysis' in wyniki_monitoringu:
            dane_statusu.append(['Halucynacje', 'OK' if not wyniki_monitoringu['hallucination_analysis']['potential_hallucination'] else 'OSTRZEŻENIE'])
        
        if dane_statusu:
            tabela = osie[1,1].table(cellText=dane_statusu, 
                                  colLabels=['Komponent', 'Status'],
                                  cellLoc='center', loc='center')
            tabela.auto_set_font_size(False)
            tabela.set_fontsize(12)
            tabela.scale(1.2, 1.5)
        
        osie[1,1].set_title('Status Monitoringu')
        
        plt.tight_layout()
        plt.savefig('results/plots/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Zapisano dashboard monitoringu")
    
    def monitoruj_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                     X_baseline: Optional[np.ndarray] = None) -> Dict[str, Any]:
        logger.info("=== ROZPOCZĘCIE MONITORINGU MODELU ===")
        
        # Predykcje
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Obliczenie metryk
        aktualne_metryki = self.oblicz_metryki_modelu(y_test, y_pred, y_pred_proba)
        
        wyniki_monitoringu = {
            'model_name': self.nazwa_modelu,
            'timestamp': datetime.now().isoformat(),
            'current_metrics': aktualne_metryki
        }
        
        # Wykrywanie driftu danych
        if X_baseline is not None:
            drift_danych = self.wykryj_drift_danych(X_test, X_baseline)
            wyniki_monitoringu['data_drift'] = drift_danych
        
        # Wykrywanie driftu wydajności
        drift_wydajnosci = self.wykryj_drift_wydajnosci(aktualne_metryki)
        wyniki_monitoringu['performance_drift'] = drift_wydajnosci
        
        # Wykrywanie halucynacji
        analiza_halucynacji = self.wykryj_wzorce_halucynacji(y_pred_proba)
        wyniki_monitoringu['hallucination_analysis'] = analiza_halucynacji
        
        # Analiza spójności
        analiza_spojnosci = self.analizuj_spojnosc_predykcji(X_test, y_pred, y_pred_proba)
        wyniki_monitoringu['consistency_analysis'] = analiza_spojnosci
        
        # Aktualizacja historii
        self.historia_monitoringu.append(wyniki_monitoringu)
        
        # Tworzenie dashboardu
        self.utworz_dashboard_monitoringu(wyniki_monitoringu)
        
        # Zapisanie wyników
        self.zapisz_wyniki_monitoringu(wyniki_monitoringu)
        
        logger.info("=== ZAKOŃCZENIE MONITORINGU MODELU ===")
        return wyniki_monitoringu
    
    def zapisz_wyniki_monitoringu(self, wyniki: Dict[str, Any]) -> None:
        znacznik_czasu = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Zapisanie JSON
        sciezka_wynikow = Path('results/monitoring') / f'monitoring_results_{znacznik_czasu}.json'
        sciezka_wynikow.parent.mkdir(parents=True, exist_ok=True)
        
        with open(sciezka_wynikow, 'w', encoding='utf-8') as plik:
            json.dump(wyniki, plik, indent=2, ensure_ascii=False, default=str)
        
        # Zapisanie raportu tekstowego
        sciezka_raportu = Path('results/monitoring') / f'monitoring_report_{znacznik_czasu}.txt'
        with open(sciezka_raportu, 'w', encoding='utf-8') as plik:
            plik.write(f"=== RAPORT MONITORINGU MODELU - {self.nazwa_modelu} ===\n")
            plik.write(f"Data: {wyniki['timestamp']}\n\n")
            
            # Metryki wydajności
            plik.write("=== METRYKI WYDAJNOŚCI ===\n")
            for metryka, wartosc in wyniki['current_metrics'].items():
                if metryka != 'timestamp':
                    plik.write(f"{metryka}: {wartosc:.4f}\n")
            
            # Status komponentów
            plik.write("\n=== STATUS KOMPONENTÓW ===\n")
            if 'data_drift' in wyniki:
                plik.write(f"Drift danych: {'WYKRYTO' if wyniki['data_drift']['drift_detected'] else 'BRAK'}\n")
            if 'performance_drift' in wyniki:
                plik.write(f"Drift wydajności: {'WYKRYTO' if wyniki.get('significant_drift', False) else 'BRAK'}\n")
            if 'hallucination_analysis' in wyniki:
                plik.write(f"Halucynacje: {'WYKRYTO' if wyniki['hallucination_analysis']['potential_hallucination'] else 'BRAK'}\n")
        
        logger.info(f"Zapisano wyniki monitoringu w: {sciezka_wynikow}")

if __name__ == "__main__":
    # Przykład użycia
    # monitor = MonitorModelu(model, "StrokePredictor")
    # wyniki = monitor.monitoruj_model(X_test, y_test, X_baseline)
    pass
