#!/usr/bin/env python3

import logging
from typing import Dict, Any, Optional
from .system import StrokePredictionSystem

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    
    def __init__(self, system: StrokePredictionSystem):
        self.system = system
    
    def uruchom_pipeline(self, data_path: str) -> Dict[str, Any]:
        logger.info("ROZPOCZĘCIE PIPELINE")
        
        try:
            # Etap 1-2: Pre-processing i wizualizacja
            data_results = self._uruchom_przygotowanie_danych(data_path)
            
            # Etap 3-4: Trenowanie i ocena modeli
            model_results = self._uruchom_trenowanie_i_ocena_modeli(data_results)
            
            # Etap 5-7: Optymalizacja i kalibracja
            optimization_results = self._uruchom_optymalizacja_modelu(data_results, model_results)
            
            # Etap 8: Monitoring i raport
            final_results = self._uruchom_monitoring_i_raportowanie(optimization_results)
            
            logger.info("PIPELINE ZAKOŃCZONY POMYŚLNIE")
            return final_results
            
        except FileNotFoundError as e:
            logger.error(f"BŁĄD: Nie znaleziono pliku - {e}")
            return {
                'sukces': False,
                'blad': f"Nie znaleziono pliku: {e}",
                'typ_bledu': 'NieZnalezionoPliku'
            }
        except ValueError as e:
            logger.error(f"BŁĄD: Nieprawidłowe dane - {e}")
            return {
                'sukces': False,
                'blad': f"Błąd danych: {e}",
                'typ_bledu': 'BłądDanych'
            }
        except Exception as e:
            logger.error(f"BŁĄD W PIPELINE: {e}")
            return {
                'sukces': False,
                'blad': str(e),
                'typ_bledu': 'NieznanyBłąd'
            }
    
    def _uruchom_przygotowanie_danych(self, data_path: str):
        # 1. Pre-processing
        X_train, X_test, y_train, y_test, nazwy_cech, df_czysty = self.system.zaladuj_i_przetworz_dane(data_path)
        
        # 2. Wizualizacja
        wnioski_wizualizacji = self.system.utworz_wizualizacje(df_czysty)
        
        return {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'nazwy_cech': nazwy_cech, 'df_czysty': df_czysty,
            'wnioski_wizualizacji': wnioski_wizualizacji
        }
    
    def _uruchom_trenowanie_i_ocena_modeli(self, wyniki_danych):
        X_train = wyniki_danych['X_train']
        y_train = wyniki_danych['y_train']
        X_test = wyniki_danych['X_test']
        y_test = wyniki_danych['y_test']
        
        # 3. Trenowanie modeli
        wyniki_modeli = self.system.trenuj_modele(X_train, y_train)
        
        # 4. Ocena modeli
        wyniki_oceny = self.system.ocen_modele(X_test, y_test)
        
        return {
            **wyniki_danych,
            'wyniki_modeli': wyniki_modeli,
            'wyniki_oceny': wyniki_oceny
        }
    
    def _uruchom_optymalizacja_modelu(self, wyniki_danych, wyniki_modeli):
        X_train = wyniki_danych['X_train']
        y_train = wyniki_danych['y_train']
        X_test = wyniki_danych['X_test']
        y_test = wyniki_danych['y_test']
        
        # 5. Ulepszona obsługa niezbalansowania
        X_train_balanced, y_train_balanced = self.system.obsluz_niezbalansowanie_klas(X_train, y_train)
        
        # 6. Optymalizacja progu klasyfikacji (PO balansowaniu)
        metryki_progu = self.system.optymalizuj_próg_klasyfikacji(X_test, y_test)
        
        # 7. Kalibracja modelu
        model_kalibrowany = self.system.kalibruj_model(X_train_balanced, y_train_balanced)
        
        return {
            **wyniki_modeli,
            'X_train_balanced': X_train_balanced,
            'y_train_balanced': y_train_balanced,
            'metryki_progu': metryki_progu,
            'model_kalibrowany': model_kalibrowany
        }
    
    def _uruchom_monitoring_i_raportowanie(self, wyniki_optymalizacji):
        X_test = wyniki_optymalizacji['X_test']
        y_test = wyniki_optymalizacji['y_test']
        X_train_balanced = wyniki_optymalizacji['X_train_balanced']
        
        # 8. Monitoring
        wyniki_monitoringu = self.system.monitoruj_wydajnosc_modelu(X_test, y_test, X_train_balanced)
        
        # 9. Raport
        sciezka_raportu = self.system.wygeneruj_raport()
        
        return {
            'sukces': True,
            'sciezka_raportu': sciezka_raportu,
            'wyniki': self.system.results,
            'wnioski_wizualizacji': wyniki_optymalizacji['wnioski_wizualizacji'],
            'wyniki_monitoringu': wyniki_monitoringu
        }
