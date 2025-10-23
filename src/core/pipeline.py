#!/usr/bin/env python3

import logging
from typing import Dict, Any, Optional
from .system import StrokePredictionSystem

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrator dla kompletnego pipeline'u ML"""
    
    def __init__(self, system: StrokePredictionSystem):
        """Inicjalizacja orchestratora"""
        self.system = system
    
    def run_pipeline(self, data_path: str) -> Dict[str, Any]:
        """
        Uruchomienie kompletnego pipeline'u predykcji udarów
        
        Args:
            data_path (str): Ścieżka do pliku z danymi medycznymi (CSV)
            
        Returns:
            dict: Słownik z wynikami pipeline'u
        """
        logger.info("ROZPOCZĘCIE PIPELINE")
        
        try:
            # Etap 1-2: Pre-processing i wizualizacja
            data_results = self._run_data_preparation(data_path)
            
            # Etap 3-4: Trenowanie i ocena modeli
            model_results = self._run_model_training_and_evaluation(data_results)
            
            # Etap 5-7: Optymalizacja i kalibracja
            optimization_results = self._run_model_optimization(data_results, model_results)
            
            # Etap 8: Monitoring i raport
            final_results = self._run_monitoring_and_reporting(optimization_results)
            
            logger.info("PIPELINE ZAKOŃCZONY POMYŚLNIE")
            return final_results
            
        except FileNotFoundError as e:
            logger.error(f"BŁĄD: Nie znaleziono pliku - {e}")
            return {
                'success': False,
                'error': f"Nie znaleziono pliku: {e}",
                'error_type': 'FileNotFoundError'
            }
        except ValueError as e:
            logger.error(f"BŁĄD: Nieprawidłowe dane - {e}")
            return {
                'success': False,
                'error': f"Błąd danych: {e}",
                'error_type': 'ValueError'
            }
        except Exception as e:
            logger.error(f"BŁĄD W PIPELINE: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'UnknownError'
            }
    
    def _run_data_preparation(self, data_path: str):
        """Etap 1-2: Pre-processing i wizualizacja danych"""
        # 1. Pre-processing
        X_train, X_test, y_train, y_test, feature_names, df_clean = self.system.load_and_preprocess_data(data_path)
        
        # 2. Wizualizacja
        visualization_insights = self.system.create_visualizations(df_clean)
        
        return {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'feature_names': feature_names, 'df_clean': df_clean,
            'visualization_insights': visualization_insights
        }
    
    def _run_model_training_and_evaluation(self, data_results):
        """Etap 3-4: Trenowanie i ocena modeli"""
        X_train = data_results['X_train']
        y_train = data_results['y_train']
        X_test = data_results['X_test']
        y_test = data_results['y_test']
        
        # 3. Trenowanie modeli
        model_results = self.system.train_models(X_train, y_train)
        
        # 4. Ocena modeli
        evaluation_results = self.system.evaluate_models(X_test, y_test)
        
        return {
            **data_results,
            'model_results': model_results,
            'evaluation_results': evaluation_results
        }
    
    def _run_model_optimization(self, data_results, model_results):
        """Etap 5-7: Optymalizacja i kalibracja modelu"""
        X_train = data_results['X_train']
        y_train = data_results['y_train']
        X_test = data_results['X_test']
        y_test = data_results['y_test']
        
        # 5. Ulepszona obsługa niezbalansowania
        X_train_balanced, y_train_balanced = self.system.handle_class_imbalance(X_train, y_train)
        
        # 6. Optymalizacja progu klasyfikacji (PO balansowaniu)
        threshold_metrics = self.system.optimize_classification_threshold(X_test, y_test)
        
        # 7. Kalibracja modelu
        calibrated_model = self.system.calibrate_model(X_train_balanced, y_train_balanced)
        
        return {
            **model_results,
            'X_train_balanced': X_train_balanced,
            'y_train_balanced': y_train_balanced,
            'threshold_metrics': threshold_metrics,
            'calibrated_model': calibrated_model
        }
    
    def _run_monitoring_and_reporting(self, optimization_results):
        """Etap 8: Monitoring i generowanie raportu"""
        X_test = optimization_results['X_test']
        y_test = optimization_results['y_test']
        X_train_balanced = optimization_results['X_train_balanced']
        
        # 8. Monitoring
        monitoring_results = self.system.monitor_model_performance(X_test, y_test, X_train_balanced)
        
        # 9. Raport
        report_path = self.system.generate_report()
        
        return {
            'success': True,
            'report_path': report_path,
            'results': self.system.results,
            'visualization_insights': optimization_results['visualization_insights'],
            'monitoring_results': monitoring_results
        }
