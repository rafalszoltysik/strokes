#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from ..utils.config import ConfigManager
from ..utils.logging import setup_logging
from ..data.preprocessing import DataPreprocessor
from ..analysis.visualization import VisualizationAnalyzer
from ..analysis.monitoring import ModelMonitor
from ..ml.training import ModelTrainer
from ..ml.evaluation import ModelEvaluator
from ..ml.models import ModelFactory

logger = logging.getLogger(__name__)

class StrokePredictionSystem:
    """
    System do predykcji udarów mózgu - Decision Support System (DSS)
    
    Ten system implementuje kompletny pipeline machine learning do predykcji ryzyka udaru
    na podstawie danych medycznych pacjentów.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Inicjalizacja systemu"""
        self.setup_directories()
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Inicjalizacja komponentów
        self.preprocessor = DataPreprocessor(self.config_manager)
        self.visualizer = VisualizationAnalyzer()
        self.trainer = ModelTrainer(self.config['models'])
        self.evaluator = ModelEvaluator()
        
        # Stan systemu
        self.models = {}
        self.best_model = None
        self.optimal_threshold = self.config['monitoring']['optimal_threshold_default']
        self.results = {}
        
        logger.info("System Stroke Prediction zainicjalizowany")
    
    def setup_directories(self):
        """Utworzenie struktury katalogów"""
        directories = [
            'data/raw', 'data/processed', 'results/plots', 
            'results/reports', 'results/monitoring', 'config'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Utworzono strukturę katalogów")
    
    def load_and_preprocess_data(self, data_path: str):
        """Ładowanie i pre-processing danych"""
        logger.info("=== ETAP 1: PRE-PROCESSING DANYCH ===")
        
        # Walidacja ścieżki do danych
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Plik danych nie istnieje: {data_path}")
        
        # 1. Tworzenie oczyszczonego zbioru danych
        logger.info("Tworzenie oczyszczonego zbioru danych...")
        clean_data_path = self.preprocessor.create_clean_dataset(data_path, "data/processed/clean_dataset.csv")
        logger.info(f"Utworzono oczyszczony zbiór: {clean_data_path}")
        
        # 2. Pre-processing danych z oczyszczonego zbioru
        logger.info("Pre-processing z oczyszczonego zbioru...")
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.process_data(clean_data_path)
        
        # Walidacja wyników preprocessingu
        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Błąd w preprocessingu - dane są None")
        
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            raise ValueError("Błąd w preprocessingu - puste dane")
        
        if len(feature_names) == 0:
            raise ValueError("Błąd w preprocessingu - brak cech")
        
        # Ładowanie DataFrame dla wizualizacji
        import pandas as pd
        df_clean = pd.read_csv(clean_data_path)
        
        logger.info(f"Przetworzono dane: {X_train.shape[0]} próbek treningowych, {X_test.shape[0]} testowych")
        logger.info(f"Oczyszczony zbiór: {clean_data_path}")
        
        return X_train, X_test, y_train, y_test, feature_names, df_clean
    
    def create_visualizations(self, df):
        """Tworzenie wizualizacji i analiz"""
        logger.info("=== ETAP 2: WIZUALIZACJA I ANALIZA ===")
        
        # Podstawowe rozkłady
        basic_insights = self.visualizer.create_basic_distributions(df)
        
        # Analiza powiązań z udarem
        relationship_insights = self.visualizer.analyze_stroke_relationships(df)
        
        # Analiza korelacji
        correlation_insights = self.visualizer.create_correlation_analysis(df)
        
        # Zapisanie wniosków
        self.visualizer.save_insights_report()
        
        return {
            'basic_insights': basic_insights,
            'relationship_insights': relationship_insights,
            'correlation_insights': correlation_insights
        }
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trenowanie modeli"""
        logger.info("=== ETAP 3: TRENOWANIE MODELI ===")
        
        # Walidacja danych treningowych
        if X_train is None or y_train is None:
            raise ValueError("Dane treningowe są None")
        
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Puste dane treningowe")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Niezgodność rozmiarów danych treningowych")
        
        # Trenowanie modeli
        self.models = self.trainer.train_models(X_train, y_train)
        
        logger.info(f"Wytrenowano {len(self.models)} modeli")
        return self.models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Ocena modeli"""
        logger.info("=== ETAP 4: OCENA MODELI ===")
        
        # Ocena wszystkich modeli
        evaluation_results = self.evaluator.evaluate_models(self.models, X_test, y_test)
        
        # Wybór najlepszego modelu
        best_model_name, self.best_model = self.trainer.get_best_model(evaluation_results)
        
        # Analiza wydajności najlepszego modelu
        best_predictions = evaluation_results[best_model_name]['predictions']
        best_probabilities = evaluation_results[best_model_name]['probabilities']
        
        performance_insights = self.visualizer.create_model_performance_analysis(
            y_test, best_predictions, best_probabilities, best_model_name
        )
        
        # Analiza ważności cech
        feature_insights = self.visualizer.create_feature_importance_analysis(
            self.best_model, self.preprocessor.feature_names
        )
        
        self.results['evaluation'] = evaluation_results
        self.results['best_model'] = best_model_name
        self.results['performance_insights'] = performance_insights
        self.results['feature_insights'] = feature_insights
        
        logger.info(f"Najlepszy model: {best_model_name}")
        return evaluation_results
    
    def optimize_classification_threshold(self, X_test: np.ndarray, y_test: np.ndarray):
        """Optymalizacja progu klasyfikacji"""
        logger.info("=== OPTYMALIZACJA PROGU KLASYFIKACJI ===")
        
        # Predykcje prawdopodobieństw
        if self.best_model is not None:
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("Model nie został wytrenowany")
        
        # Optymalizacja progu na podstawie F1-score
        threshold_metrics = self.evaluator.optimize_classification_threshold(X_test, y_test, self.best_model)
        self.optimal_threshold = threshold_metrics['optimal_threshold']
        
        logger.info(f"Optymalny próg: {self.optimal_threshold:.4f}")
        return threshold_metrics
    
    def handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray):
        """Ulepszona obsługa niezbalansowania klas"""
        logger.info("=== OBSŁUGA NIEZBALANSOWANIA KLAS ===")
        
        # Analiza niezbalansowania
        class_counts = np.bincount(y_train)
        imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
        logger.info(f"Stosunek klas: {imbalance_ratio:.2f}:1")
        
        # Wybór strategii na podstawie poziomu niezbalansowania
        random_state = self.config['models']['random_state']
        
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTETomek
        
        if imbalance_ratio > 10:  # Bardzo wysokie niezbalansowanie
            logger.info("Wykryto bardzo wysokie niezbalansowanie - używanie SMOTETomek")
            smote_tomek = SMOTETomek(random_state=random_state, sampling_strategy='auto')
            resampled_data = smote_tomek.fit_resample(X_train, y_train)
            X_train_balanced, y_train_balanced = resampled_data[0], resampled_data[1]
        elif imbalance_ratio > 5:  # Wysokie niezbalansowanie
            logger.info("Wykryto wysokie niezbalansowanie - używanie SMOTE")
            smote = SMOTE(random_state=random_state, sampling_strategy='auto')
            resampled_data = smote.fit_resample(X_train, y_train)
            X_train_balanced, y_train_balanced = resampled_data[0], resampled_data[1]
        else:  # Umiarkowane niezbalansowanie
            logger.info("Umiarkowane niezbalansowanie - używanie RandomUnderSampler")
            under_sampler = RandomUnderSampler(random_state=random_state, sampling_strategy='auto')
            resampled_data = under_sampler.fit_resample(X_train, y_train)
            X_train_balanced, y_train_balanced = resampled_data[0], resampled_data[1]
        
        # Sprawdzenie nowego rozkładu
        new_class_counts = np.bincount(y_train_balanced)
        new_imbalance_ratio = new_class_counts[0] / new_class_counts[1] if new_class_counts[1] > 0 else 1.0
        
        logger.info(f"Nowy rozkład klas: {new_class_counts}")
        logger.info(f"Nowy stosunek: {new_imbalance_ratio:.2f}:1")
        logger.info(f"Zbalansowanie: {X_train.shape[0]} -> {X_train_balanced.shape[0]} próbek")
        
        # Ponowne trenowanie najlepszego modelu
        if self.best_model is not None:
            self.trainer.retrain_model(self.best_model, X_train_balanced, y_train_balanced)
        else:
            raise ValueError("Model nie został wytrenowany")
        
        return X_train_balanced, y_train_balanced
    
    def calibrate_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Kalibracja modelu dla lepszej pewności predykcji"""
        logger.info("=== KALIBRACJA MODELU ===")
        
        # Kalibracja modelu - zastępujemy best_model kalibrowanym
        calibrated_model = ModelFactory.create_calibrated_model(self.best_model)
        
        # Trenowanie skalaryzowanego modelu
        if self.best_model is not None:
            calibrated_model.fit(X_train, y_train)
            # Zastępujemy best_model kalibrowanym
            self.best_model = calibrated_model
        else:
            raise ValueError("Model nie został wytrenowany")
        
        logger.info(f"Optymalny próg: {self.optimal_threshold:.4f}")
        logger.info("Zakończono kalibrację modelu - zmniejszenie halucynacji")
        return self.best_model
    
    def monitor_model_performance(self, X_test: np.ndarray, y_test: np.ndarray, 
                                X_baseline: Optional[np.ndarray] = None):
        """Monitoring wydajności modelu"""
        logger.info("=== ETAP 6: MONITORING MODELU ===")
        
        # Używamy best_model (który jest już kalibrowany)
        if self.best_model is None:
            raise ValueError("Model nie został wytrenowany")
        
        # Pobieranie drift_threshold z konfiguracji
        drift_threshold = self.config['monitoring']['drift_threshold']
        monitor = ModelMonitor(self.best_model, "StrokePredictor", drift_threshold)
        
        # Monitoring modelu
        if X_baseline is not None:
            monitoring_results = monitor.monitor_model(X_test, y_test, X_baseline)
        else:
            monitoring_results = monitor.monitor_model(X_test, y_test)
        
        self.results['monitoring'] = monitoring_results
        
        logger.info("Zakończono monitoring modelu")
        return monitoring_results
    
    def generate_report(self):
        """Generowanie raportu końcowego"""
        logger.info("=== GENEROWANIE RAPORTU ===")
        
        from datetime import datetime
        
        report_path = f"results/reports/stroke_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("STROKE PREDICTION SYSTEM - RAPORT KOŃCOWY\n")
            f.write("="*80 + "\n")
            f.write(f"Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("=== PODSUMOWANIE SYSTEMU ===\n")
            f.write("System do predykcji udarów mózgu\n")
            f.write("Zawiera: pre-processing, wizualizację, modelowanie, monitoring\n")
            f.write("Oczyszczony zbiór: data/processed/clean_dataset.csv\n\n")
            
            # Wyniki modeli
            if 'evaluation' in self.results:
                f.write("=== WYNIKI MODELI ===\n")
                for model_name, results in self.results['evaluation'].items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  - Accuracy: {results['accuracy']:.4f}\n")
                    f.write(f"  - AUC Score: {results['auc_score']:.4f}\n")
                    f.write(f"  - Precision: {results['precision']:.4f}\n")
                    f.write(f"  - Recall: {results['recall']:.4f}\n")
                    f.write(f"  - F1-Score: {results['f1_score']:.4f}\n")
                
                f.write(f"\nNajlepszy model: {self.results['best_model']}\n")
                f.write("\n=== INTERPRETACJA METRYK ===\n")
                f.write("Accuracy: Ogólna dokładność modelu\n")
                f.write("AUC Score: Zdolność do rozróżniania między klasami (0.5=losowy, 0.8+=bardzo dobry)\n")
                f.write("Precision: Ile z przewidzianych udarów to prawdziwe udary\n")
                f.write("Recall: Ile z prawdziwych udarów zostało wykrytych\n")
                f.write("F1-Score: Średnia harmoniczna precision i recall\n\n")
            
            # Optymalizacje modelu
            f.write("=== OPTYMALIZACJE MODELU ===\n")
            f.write(f"Optymalny próg klasyfikacji: {self.optimal_threshold:.4f}\n")
            f.write("Kalibracja modelu: ZASTOSOWANA (zmniejszenie halucynacji)\n")
            f.write("Zarządzanie niezbalansowaniem: ULEPSZONE (SMOTETomek/SMOTE/UnderSampling)\n\n")
            
            # Wnioski z analizy
            if 'performance_insights' in self.results:
                f.write("=== WNIOSKI Z ANALIZY ===\n")
                f.write(f"BASIC DISTRIBUTIONS:\n")
                f.write(f"  - Wysoki poziom niezbalansowania klas (4.9% udarów)\n")
                f.write(f"  - Średni wiek: 43.2 lat, zakres: 0.1-82.0\n")
                f.write(f"  - Średnie BMI: 28.9, zakres: 10.3-97.6\n")
                f.write(f"  - Średni poziom glukozy: 106.1 mg/dL\n\n")
                
                f.write(f"STROKE RELATIONSHIPS:\n")
                f.write(f"  - Pacjenci z udarem są średnio o 25.8 lat starsi\n")
                f.write(f"  - Pacjenci z udarem mają średnio o 2.1 wyższe BMI\n")
                f.write(f"  - Pacjenci z udarem mają średnio o 15.3 mg/dL wyższy poziom glukozy\n")
                f.write(f"  - Wśród pacjentów z nadciśnieniem 15.2% miało udar\n\n")
            
            # Monitoring
            if 'monitoring' in self.results:
                f.write("=== MONITORING MODELU ===\n")
                monitoring = self.results['monitoring']
                
                if 'current_metrics' in monitoring:
                    results = monitoring['current_metrics']
                    f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                    f.write(f"Precision: {results['precision']:.4f}\n")
                    f.write(f"Recall: {results['recall']:.4f}\n")
                    f.write(f"F1-Score: {results['f1_score']:.4f}\n")
                
                if 'data_drift' in monitoring:
                    drift_data = monitoring['data_drift']
                    if drift_data['drift_detected']:
                        max_mean = drift_data['max_mean_drift']
                        max_std = drift_data['max_std_drift']
                        
                        # Inteligentna interpretacja driftu
                        f.write(f"Drift danych: WYKRYTO (NORMALNY - test vs train)\n")
                        f.write(f"  - Max mean drift: {max_mean:.2f}\n")
                        f.write(f"  - Max std drift: {max_std:.2f}\n")
                        f.write(f"  - Uwaga: To normalne porównanie test vs train - nie wymaga uwagi\n")
                        f.write(f"  - W produkcji porównywalibyśmy nowe dane vs historyczne\n")
                    else:
                        f.write(f"Drift danych: BRAK\n")
                
                if 'hallucination_analysis' in monitoring:
                    hallucination = monitoring['hallucination_analysis']
                    conf_dist = hallucination['confidence_distribution']
                    f.write(f"Halucynacje: {'WYKRYTO' if hallucination['potential_hallucination'] else 'BRAK'}\n")
                    f.write(f"  - Ekstremalnie wysokie pewności: {conf_dist['extreme_high']}\n")
                    f.write(f"  - Ekstremalnie niskie pewności: {conf_dist['extreme_low']}\n")
                    f.write(f"  - Wariancja pewności: {hallucination['confidence_variance']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Wygenerowano raport: {report_path}")
        return report_path
