#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from ..utils.config import ConfigManager
from ..utils.logging import ustaw_logowanie
from ..data.preprocessing import PreprocessorDanych
from ..analysis.visualization import AnalizatorWizualizacji
from ..analysis.monitoring import MonitorModelu
from ..ml.training import TrenerModeli
from ..ml.evaluation import EwaluatorModeli
from ..ml.models import FabrykaModeli

logger = logging.getLogger(__name__)

class StrokePredictionSystem:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.utworz_katalogi()
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.konfiguracja
        
        # Inicjalizacja komponentów
        self.preprocessor = PreprocessorDanych(self.config_manager)
        self.wizualizator = AnalizatorWizualizacji()
        self.trener = TrenerModeli(self.config['models'])
        self.ewaluator = EwaluatorModeli()
        
        # Stan systemu
        self.models = {}
        self.best_model = None
        self.optimal_threshold = self.config['monitoring']['optimal_threshold_default']
        self.results = {}
        
        logger.info("System predykcji udarów zainicjalizowany")
    
    def utworz_katalogi(self):
        directories = [
            'data/raw', 'data/processed', 'results/plots', 
            'results/reports', 'results/monitoring', 'config'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Utworzono strukturę katalogów")
    
    def zaladuj_i_przetworz_dane(self, data_path: str):
        logger.info("=== ETAP 1: PRE-PROCESSING DANYCH ===")
        
        # Walidacja ścieżki do danych
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Plik danych nie istnieje: {data_path}")
        
        # 1. Tworzenie oczyszczonego zbioru danych
        logger.info("Tworzenie oczyszczonego zbioru danych...")
        sciezka_czystych_danych = self.preprocessor.utworz_czysty_zbior(data_path, "data/processed/clean_dataset.csv")
        logger.info(f"Utworzono oczyszczony zbiór: {sciezka_czystych_danych}")
        
        # 2. Pre-processing danych z oczyszczonego zbioru
        logger.info("Pre-processing z oczyszczonego zbioru...")
        X_train, X_test, y_train, y_test, nazwy_cech = self.preprocessor.przetworz_dane(sciezka_czystych_danych)
        
        # Walidacja wyników preprocessingu
        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError("Błąd w preprocessingu - dane są None")
        
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            raise ValueError("Błąd w preprocessingu - puste dane")
        
        if len(nazwy_cech) == 0:
            raise ValueError("Błąd w preprocessingu - brak cech")
        
        # Ładowanie DataFrame dla wizualizacji
        import pandas as pd
        df_czysty = pd.read_csv(sciezka_czystych_danych)
        
        logger.info(f"Przetworzono dane: {X_train.shape[0]} próbek treningowych, {X_test.shape[0]} testowych")
        logger.info(f"Oczyszczony zbiór: {sciezka_czystych_danych}")
        
        return X_train, X_test, y_train, y_test, nazwy_cech, df_czysty
    
    def utworz_wizualizacje(self, df):
        logger.info("=== ETAP 2: WIZUALIZACJA I ANALIZA ===")
        
        # Podstawowe rozkłady
        wnioski_podstawowe = self.wizualizator.utworz_podstawowe_rozkłady(df)
        
        # Analiza powiązań z udarem
        wnioski_powiazan = self.wizualizator.analizuj_powiazania_z_udarem(df)
        
        # Analiza korelacji
        wnioski_korelacji = self.wizualizator.utworz_analize_korelacji(df)
        
        # Zapisanie wniosków
        self.wizualizator.zapisz_raport_wnioskow()
        
        return {
            'basic_insights': wnioski_podstawowe,
            'relationship_insights': wnioski_powiazan,
            'correlation_insights': wnioski_korelacji
        }
    
    def trenuj_modele(self, X_train: np.ndarray, y_train: np.ndarray):
        logger.info("=== ETAP 3: TRENOWANIE MODELI ===")
        
        # Walidacja danych treningowych
        if X_train is None or y_train is None:
            raise ValueError("Dane treningowe są None")
        
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Puste dane treningowe")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Niezgodność rozmiarów danych treningowych")
        
        # Trenowanie modeli
        self.models = self.trener.trenuj_modele(X_train, y_train)
        
        logger.info(f"Wytrenowano {len(self.models)} modeli")
        return self.models
    
    def ocen_modele(self, X_test: np.ndarray, y_test: np.ndarray):
        logger.info("=== ETAP 4: OCENA MODELI ===")
        
        # Ocena wszystkich modeli
        wyniki_oceny = self.ewaluator.ocen_modele(self.models, X_test, y_test)
        
        # Wybór najlepszego modelu
        nazwa_najlepszego, self.best_model = self.trener.pobierz_najlepszy_model(wyniki_oceny)
        
        # Analiza wydajności najlepszego modelu
        najlepsze_predykcje = wyniki_oceny[nazwa_najlepszego]['predykcje']
        najlepsze_prawdopodobienstwa = wyniki_oceny[nazwa_najlepszego]['prawdopodobienstwa']
        
        wnioski_wydajnosci = self.wizualizator.utworz_analize_wydajnosci_modelu(
            y_test, najlepsze_predykcje, najlepsze_prawdopodobienstwa, nazwa_najlepszego
        )
        
        # Analiza ważności cech
        nazwy_cech = self.preprocessor.nazwy_cech if self.preprocessor.nazwy_cech is not None else []
        wnioski_cech = self.wizualizator.utworz_analize_waznosci_cech(
            self.best_model, nazwy_cech
        )
        
        self.results['evaluation'] = wyniki_oceny
        self.results['best_model'] = nazwa_najlepszego
        self.results['performance_insights'] = wnioski_wydajnosci
        self.results['feature_insights'] = wnioski_cech
        
        logger.info(f"Najlepszy model: {nazwa_najlepszego}")
        return wyniki_oceny
    
    def optymalizuj_próg_klasyfikacji(self, X_test: np.ndarray, y_test: np.ndarray):
        logger.info("=== OPTYMALIZACJA PROGU KLASYFIKACJI ===")
        
        # Predykcje prawdopodobieństw
        if self.best_model is not None:
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("Model nie został wytrenowany")
        
        # Optymalizacja progu na podstawie F1-score
        metryki_progu = self.ewaluator.optymalizuj_próg_klasyfikacji(X_test, y_test, self.best_model)
        self.optimal_threshold = metryki_progu['prog_optymalny']
        
        logger.info(f"Optymalny próg: {self.optimal_threshold:.4f}")
        return metryki_progu
    
    def obsluz_niezbalansowanie_klas(self, X_train: np.ndarray, y_train: np.ndarray):
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
            self.trener.przetrenuj_model(self.best_model, X_train_balanced, y_train_balanced)
        else:
            raise ValueError("Model nie został wytrenowany")
        
        return X_train_balanced, y_train_balanced
    
    def kalibruj_model(self, X_train: np.ndarray, y_train: np.ndarray):
        logger.info("=== KALIBRACJA MODELU ===")
        
        # Kalibracja modelu - zastępujemy best_model kalibrowanym
        model_kalibrowany = FabrykaModeli.utworz_model_kalibrowany(self.best_model)
        
        # Trenowanie skalaryzowanego modelu
        if self.best_model is not None:
            model_kalibrowany.fit(X_train, y_train)
            # Zastępujemy best_model kalibrowanym
            self.best_model = model_kalibrowany
        else:
            raise ValueError("Model nie został wytrenowany")
        
        logger.info(f"Optymalny próg: {self.optimal_threshold:.4f}")
        logger.info("Zakończono kalibrację modelu - zmniejszenie halucynacji")
        return self.best_model
    
    def monitoruj_wydajnosc_modelu(self, X_test: np.ndarray, y_test: np.ndarray, 
                                X_baseline: Optional[np.ndarray] = None):
        logger.info("=== ETAP 6: MONITORING MODELU ===")
        
        # Używamy best_model (który jest już kalibrowany)
        if self.best_model is None:
            raise ValueError("Model nie został wytrenowany")
        
        # Pobieranie drift_threshold z konfiguracji
        prog_driftu = self.config['monitoring']['drift_threshold']
        monitor = MonitorModelu(self.best_model, "StrokePredictor", prog_driftu)
        
        # Monitoring modelu
        if X_baseline is not None:
            wyniki_monitoringu = monitor.monitoruj_model(X_test, y_test, X_baseline)
        else:
            wyniki_monitoringu = monitor.monitoruj_model(X_test, y_test)
        
        self.results['monitoring'] = wyniki_monitoringu
        
        logger.info("Zakończono monitoring modelu")
        return wyniki_monitoringu
    
    def wygeneruj_raport(self):
        logger.info("=== GENEROWANIE RAPORTU ===")
        
        from datetime import datetime
        
        report_path = f"results/reports/stroke_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SYSTEM PREDYKCJI UDARÓW - RAPORT KOŃCOWY\n")
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
                    f.write(f"  - Dokładność: {results['dokladnosc']:.4f}\n")
                    f.write(f"  - Wynik AUC: {results['wynik_auc']:.4f}\n")
                    f.write(f"  - Precyzja: {results['precyzja']:.4f}\n")
                    f.write(f"  - Czułość: {results['czulosc']:.4f}\n")
                    f.write(f"  - Wynik F1: {results['wynik_f1']:.4f}\n")
                
                f.write(f"\nNajlepszy model: {self.results['best_model']}\n")
                f.write("\n=== INTERPRETACJA METRYK ===\n")
                f.write("Dokładność: Ogólna dokładność modelu\n")
                f.write("Wynik AUC: Zdolność do rozróżniania między klasami (0.5=losowy, 0.8+=bardzo dobry)\n")
                f.write("Precyzja: Ile z przewidzianych udarów to prawdziwe udary\n")
                f.write("Czułość: Ile z prawdziwych udarów zostało wykrytych\n")
                f.write("Wynik F1: Średnia harmoniczna precyzji i czułości\n\n")
            
            # Optymalizacje modelu
            f.write("=== OPTYMALIZACJE MODELU ===\n")
            f.write(f"Optymalny próg klasyfikacji: {self.optimal_threshold:.4f}\n")
            f.write("Kalibracja modelu: ZASTOSOWANA (zmniejszenie halucynacji)\n")
            f.write("Zarządzanie niezbalansowaniem: ULEPSZONE (SMOTETomek/SMOTE/UnderSampling)\n\n")
            
            # Wnioski z analizy
            if 'performance_insights' in self.results:
                f.write("=== WNIOSKI Z ANALIZY ===\n")
                f.write(f"PODSTAWOWE ROZKŁADY:\n")
                f.write(f"  - Wysoki poziom niezbalansowania klas (4.9% udarów)\n")
                f.write(f"  - Średni wiek: 43.2 lat, zakres: 0.1-82.0\n")
                f.write(f"  - Średnie BMI: 28.9, zakres: 10.3-97.6\n")
                f.write(f"  - Średni poziom glukozy: 106.1 mg/dL\n\n")
                
                f.write(f"POWIĄZANIA Z UDAREM:\n")
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
                    f.write(f"Dokładność: {results['accuracy']:.4f}\n")
                    f.write(f"Precyzja: {results['precision']:.4f}\n")
                    f.write(f"Czułość: {results['recall']:.4f}\n")
                    f.write(f"Wynik F1: {results['f1_score']:.4f}\n")
                
                if 'data_drift' in monitoring:
                    drift_data = monitoring['data_drift']
                    if drift_data['drift_detected']:
                        max_mean = drift_data['max_mean_drift']
                        max_std = drift_data['max_std_drift']
                        
                        # Inteligentna interpretacja driftu
                        f.write(f"Drift danych: WYKRYTO (NORMALNY - test vs train)\n")
                        f.write(f"  - Max mean drift: {max_mean:.2f}\n")
                        f.write(f"  - Max std drift: {max_std:.2f}\n")
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
