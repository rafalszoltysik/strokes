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

class ModelMonitor:
    """
    Model monitoring and hallucination detection
    """
    
    def __init__(self, model, model_name: str = "StrokePredictor", drift_threshold: float = 0.05):
        """Inicjalizacja monitora modelu"""
        self.model = model
        self.model_name = model_name
        self.monitoring_history = []
        self.performance_baseline = None
        self.drift_threshold = drift_threshold  # Próg driftu z konfiguracji
        
    def calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Obliczenie metryk wydajności modelu"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc_score': roc_auc_score(y_true, y_pred_proba),
            'timestamp': datetime.now().isoformat()
        }
        return metrics
    
    def detect_data_drift(self, X_new: np.ndarray, X_baseline: np.ndarray) -> Dict[str, Any]:
        """Wykrywanie driftu danych"""
        logger.info("Sprawdzanie driftu danych")
        
        drift_results = {}
        
        # Porównanie statystyk
        baseline_stats = {
            'mean': np.mean(X_baseline, axis=0),
            'std': np.std(X_baseline, axis=0),
            'min': np.min(X_baseline, axis=0),
            'max': np.max(X_baseline, axis=0)
        }
        
        new_stats = {
            'mean': np.mean(X_new, axis=0),
            'std': np.std(X_new, axis=0),
            'min': np.min(X_new, axis=0),
            'max': np.max(X_new, axis=0)
        }
        
        # Obliczenie różnic z obsługą dzielenia przez zero
        # Unikamy dzielenia przez zero używając np.where
        mean_drift = np.where(
            np.abs(baseline_stats['mean']) > 1e-10,  # Jeśli baseline nie jest bliski zero
            np.abs(new_stats['mean'] - baseline_stats['mean']) / np.abs(baseline_stats['mean']),
            0.0  # Jeśli baseline jest bliski zero, drift = 0
        )
        
        std_drift = np.where(
            np.abs(baseline_stats['std']) > 1e-10,  # Jeśli baseline nie jest bliski zero
            np.abs(new_stats['std'] - baseline_stats['std']) / np.abs(baseline_stats['std']),
            0.0  # Jeśli baseline jest bliski zero, drift = 0
        )
        
        drift_results = {
            'mean_drift': mean_drift.tolist(),
            'std_drift': std_drift.tolist(),
            'max_mean_drift': float(np.max(mean_drift)),
            'max_std_drift': float(np.max(std_drift)),
            'drift_detected': np.max(mean_drift) > self.drift_threshold or np.max(std_drift) > self.drift_threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_results['drift_detected']:
            logger.warning(f"Wykryto drift danych: max_mean_drift={drift_results['max_mean_drift']:.3f}, max_std_drift={drift_results['max_std_drift']:.3f}")
            logger.info(f"Próg driftu: {self.drift_threshold}")
            logger.info(f"Liczba cech z driftem mean: {np.sum(mean_drift > self.drift_threshold)}")
            logger.info(f"Liczba cech z driftem std: {np.sum(std_drift > self.drift_threshold)}")
        else:
            logger.info("Nie wykryto znaczącego driftu danych")
            logger.info(f"Max mean drift: {drift_results['max_mean_drift']:.6f}, Max std drift: {drift_results['max_std_drift']:.6f}")
        
        return drift_results
    
    def detect_performance_drift(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Wykrywanie driftu wydajności"""
        logger.info("Sprawdzanie driftu wydajności")
        
        if self.performance_baseline is None:
            self.performance_baseline = current_metrics
            logger.info("Ustawiono baseline wydajności")
            return {'drift_detected': False, 'message': 'Baseline ustawiony'}
        
        performance_drift = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']:
            if metric in self.performance_baseline and metric in current_metrics:
                baseline_value = self.performance_baseline[metric]
                current_value = current_metrics[metric]
                drift = (baseline_value - current_value) / baseline_value
                performance_drift[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'drift': drift,
                    'significant_drift': drift > self.drift_threshold
                }
        
        significant_drift = any(drift['significant_drift'] for drift in performance_drift.values())
        
        drift_results = {
            'performance_drift': performance_drift,
            'significant_drift': significant_drift,
            'timestamp': datetime.now().isoformat()
        }
        
        if significant_drift:
            logger.warning("Wykryto znaczący drift wydajności!")
            for metric, drift_info in performance_drift.items():
                if drift_info['significant_drift']:
                    logger.warning(f"{metric}: {drift_info['baseline']:.3f} -> {drift_info['current']:.3f} (drift: {drift_info['drift']:.3f})")
        else:
            logger.info("Wydajność modelu pozostaje stabilna")
        
        return drift_results
    
    def detect_hallucination_patterns(self, y_pred_proba: np.ndarray, 
                                    confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """Wykrywanie wzorców halucynacji"""
        logger.info("Sprawdzanie wzorców halucynacji")
        
        # Analiza rozkładu pewności predykcji
        high_confidence = np.sum(y_pred_proba > confidence_threshold)
        low_confidence = np.sum(y_pred_proba < (1 - confidence_threshold))
        medium_confidence = len(y_pred_proba) - high_confidence - low_confidence
        
        # Wykrywanie ekstremalnych wartości pewności
        extreme_high = np.sum(y_pred_proba > 0.95)
        extreme_low = np.sum(y_pred_proba < 0.05)
        
        # Analiza wariancji pewności
        confidence_variance = np.var(y_pred_proba)
        
        hallucination_results = {
            'confidence_distribution': {
                'high_confidence': int(high_confidence),
                'medium_confidence': int(medium_confidence),
                'low_confidence': int(low_confidence),
                'extreme_high': int(extreme_high),
                'extreme_low': int(extreme_low)
            },
            'confidence_variance': float(confidence_variance),
            'potential_hallucination': extreme_high > len(y_pred_proba) * 0.1 or extreme_low > len(y_pred_proba) * 0.1,
            'timestamp': datetime.now().isoformat()
        }
        
        if hallucination_results['potential_hallucination']:
            logger.warning("Wykryto potencjalne wzorce halucynacji!")
            logger.warning(f"Ekstremalnie wysokie pewności: {extreme_high}, ekstremalnie niskie: {extreme_low}")
        else:
            logger.info("Nie wykryto wzorców halucynacji")
        
        return hallucination_results
    
    def analyze_prediction_consistency(self, X: np.ndarray, y_pred: np.ndarray, 
                                     y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analiza spójności predykcji"""
        logger.info("Analiza spójności predykcji")
        
        # Analiza rozkładu predykcji
        prediction_distribution = {
            'positive_predictions': int(np.sum(y_pred == 1)),
            'negative_predictions': int(np.sum(y_pred == 0)),
            'positive_rate': float(np.mean(y_pred))
        }
        
        # Analiza pewności predykcji
        confidence_stats = {
            'mean_confidence': float(np.mean(y_pred_proba)),
            'std_confidence': float(np.std(y_pred_proba)),
            'min_confidence': float(np.min(y_pred_proba)),
            'max_confidence': float(np.max(y_pred_proba))
        }
        
        # Wykrywanie anomalii w predykcjach
        anomaly_threshold = 2.0  # 2 odchylenia standardowe
        confidence_z_scores = np.abs((y_pred_proba - confidence_stats['mean_confidence']) / confidence_stats['std_confidence'])
        anomalies = np.sum(confidence_z_scores > anomaly_threshold)
        
        consistency_results = {
            'prediction_distribution': prediction_distribution,
            'confidence_stats': confidence_stats,
            'anomalies_detected': int(anomalies),
            'anomaly_rate': float(anomalies / len(y_pred_proba)),
            'timestamp': datetime.now().isoformat()
        }
        
        if consistency_results['anomaly_rate'] > 0.05:  # 5% anomalii
            logger.warning(f"Wykryto {anomalies} anomalii w predykcjach ({consistency_results['anomaly_rate']:.1%})")
        else:
            logger.info("Predykcje są spójne")
        
        return consistency_results
    
    def create_monitoring_dashboard(self, monitoring_results: Dict[str, Any]) -> None:
        """Tworzenie dashboardu monitoringu"""
        logger.info("Tworzenie dashboardu monitoringu")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dashboard Monitoringu Modelu - {self.model_name}', fontsize=16, fontweight='bold')
        
        # 1. Historia wydajności
        if 'performance_history' in monitoring_results:
            history = monitoring_results['performance_history']
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            
            for i, metric in enumerate(metrics):
                if metric in history:
                    axes[0,0].plot(history[metric], label=metric, marker='o')
            
            axes[0,0].set_title('Historia Wydajności Modelu')
            axes[0,0].set_xlabel('Czas')
            axes[0,0].set_ylabel('Wartość metryki')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Rozkład pewności predykcji
        if 'hallucination_analysis' in monitoring_results:
            hallucination = monitoring_results['hallucination_analysis']
            conf_dist = hallucination['confidence_distribution']
            
            categories = ['Niska', 'Średnia', 'Wysoka', 'Ekstremalnie wysoka']
            values = [conf_dist['low_confidence'], conf_dist['medium_confidence'], 
                    conf_dist['high_confidence'], conf_dist['extreme_high']]
            
            axes[0,1].bar(categories, values, color=['red', 'yellow', 'green', 'purple'], alpha=0.7)
            axes[0,1].set_title('Rozkład Pewności Predykcji')
            axes[0,1].set_ylabel('Liczba predykcji')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Analiza driftu danych
        if 'data_drift' in monitoring_results:
            drift = monitoring_results['data_drift']
            if 'mean_drift' in drift:
                mean_drift = np.array(drift['mean_drift'])
                axes[1,0].bar(range(len(mean_drift)), mean_drift, alpha=0.7, color='orange')
                axes[1,0].set_title('Drift Średnich Cech')
                axes[1,0].set_xlabel('Indeks cechy')
                axes[1,0].set_ylabel('Wielkość driftu')
                axes[1,0].axhline(y=self.drift_threshold, color='red', linestyle='--', label='Próg alarmu')
                axes[1,0].legend()
        
        # 4. Status monitoringu
        axes[1,1].axis('off')
        
        # Tworzenie tabeli statusu
        status_data = []
        if 'performance_drift' in monitoring_results:
            status_data.append(['Drift Wydajności', 'OK' if not monitoring_results.get('significant_drift', False) else 'WARNING'])
        if 'data_drift' in monitoring_results:
            status_data.append(['Drift Danych', 'OK' if not monitoring_results['data_drift']['drift_detected'] else 'WARNING'])
        if 'hallucination_analysis' in monitoring_results:
            status_data.append(['Halucynacje', 'OK' if not monitoring_results['hallucination_analysis']['potential_hallucination'] else 'WARNING'])
        
        if status_data:
            table = axes[1,1].table(cellText=status_data, 
                                  colLabels=['Komponent', 'Status'],
                                  cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
        
        axes[1,1].set_title('Status Monitoringu')
        
        plt.tight_layout()
        plt.savefig('results/plots/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Zapisano dashboard monitoringu")
    
    def monitor_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                     X_baseline: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Główna metoda monitoringu modelu"""
        logger.info("=== ROZPOCZĘCIE MONITORINGU MODELU ===")
        
        # Predykcje
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Obliczenie metryk
        current_metrics = self.calculate_model_metrics(y_test, y_pred, y_pred_proba)
        
        monitoring_results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics
        }
        
        # Wykrywanie driftu danych
        if X_baseline is not None:
            data_drift = self.detect_data_drift(X_test, X_baseline)
            monitoring_results['data_drift'] = data_drift
        
        # Wykrywanie driftu wydajności
        performance_drift = self.detect_performance_drift(current_metrics)
        monitoring_results['performance_drift'] = performance_drift
        
        # Wykrywanie halucynacji
        hallucination_analysis = self.detect_hallucination_patterns(y_pred_proba)
        monitoring_results['hallucination_analysis'] = hallucination_analysis
        
        # Analiza spójności
        consistency_analysis = self.analyze_prediction_consistency(X_test, y_pred, y_pred_proba)
        monitoring_results['consistency_analysis'] = consistency_analysis
        
        # Aktualizacja historii
        self.monitoring_history.append(monitoring_results)
        
        # Tworzenie dashboardu
        self.create_monitoring_dashboard(monitoring_results)
        
        # Zapisanie wyników
        self.save_monitoring_results(monitoring_results)
        
        logger.info("=== ZAKOŃCZENIE MONITORINGU MODELU ===")
        return monitoring_results
    
    def save_monitoring_results(self, results: Dict[str, Any]) -> None:
        """Zapisanie wyników monitoringu"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Zapisanie JSON
        results_path = Path('results/monitoring') / f'monitoring_results_{timestamp}.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Zapisanie raportu tekstowego
        report_path = Path('results/monitoring') / f'monitoring_report_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== RAPORT MONITORINGU MODELU - {self.model_name} ===\n")
            f.write(f"Data: {results['timestamp']}\n\n")
            
            # Metryki wydajności
            f.write("=== METRYKI WYDAJNOŚCI ===\n")
            for metric, value in results['current_metrics'].items():
                if metric != 'timestamp':
                    f.write(f"{metric}: {value:.4f}\n")
            
            # Status komponentów
            f.write("\n=== STATUS KOMPONENTÓW ===\n")
            if 'data_drift' in results:
                f.write(f"Drift danych: {'WYKRYTO' if results['data_drift']['drift_detected'] else 'BRAK'}\n")
            if 'performance_drift' in results:
                f.write(f"Drift wydajności: {'WYKRYTO' if results.get('significant_drift', False) else 'BRAK'}\n")
            if 'hallucination_analysis' in results:
                f.write(f"Halucynacje: {'WYKRYTO' if results['hallucination_analysis']['potential_hallucination'] else 'BRAK'}\n")
        
        logger.info(f"Zapisano wyniki monitoringu w: {results_path}")

if __name__ == "__main__":
    # Przykład użycia
    # monitor = ModelMonitor(model, "StrokePredictor")
    # results = monitor.monitor_model(X_test, y_test, X_baseline)
    pass
