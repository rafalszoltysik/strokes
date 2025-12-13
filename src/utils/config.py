#!/usr/bin/env python3

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, cast

logger = logging.getLogger(__name__)

class ConfigManager:
    
    def __init__(self, sciezka_konfiguracji: str = "config/config.yaml"):
        self.sciezka_konfiguracji = Path(sciezka_konfiguracji)
        self.konfiguracja = self._zaladuj_konfiguracje()
    
    def _zaladuj_konfiguracje(self) -> Dict[str, Any]:
        if self.sciezka_konfiguracji.exists():
            try:
                with open(self.sciezka_konfiguracji, 'r', encoding='utf-8') as plik:
                    konfiguracja = yaml.safe_load(plik)
                logger.info(f"Zaaladowano konfigurację z: {self.sciezka_konfiguracji}")
                return konfiguracja
            except Exception as e:
                logger.error(f"Błąd ładowania konfiguracji: {e}")
                return self._pobierz_domyslna_konfiguracje()
        else:
            logger.warning(f"Plik konfiguracji {self.sciezka_konfiguracji} nie istnieje, używanie domyślnej konfiguracji")
            return self._pobierz_domyslna_konfiguracje()
    
    def _pobierz_domyslna_konfiguracje(self) -> Dict[str, Any]:
        return {
            'preprocessing': {
                'test_size': 0.2,
                'random_state': 42,
                'stratify': True
            },
            'models': {
                'random_state': 42,
                'n_estimators': 100,
                'max_iter': 1000
            },
            'monitoring': {
                'drift_threshold': 0.05,
                'optimal_threshold_default': 0.5
            },
            'data_quality': {
                'problematic_values': ['#N/D', 'N/A', 'n/a', 'NULL', 'null', '', ' ', '?', '-', 'unknown', 'Unknown'],
                'invalid_ranges': {
                    'age': {'min': 0, 'max': 120},
                    'bmi': {'min': 10, 'max': 100},
                    'avg_glucose_level': {'min': 50, 'max': 600}
                }
            },
            'output': {
                'processed_data_path': 'data/processed/',
                'save_intermediate': True
            }
        }
    
    def pobierz(self, klucz: str, domyslna=None):
        klucze = klucz.split('.')
        wartosc = self.konfiguracja
        
        try:
            for k in klucze:
                wartosc = wartosc[k]
            return wartosc
        except (KeyError, TypeError):
            return domyslna
    
    def pobierz_konfiguracje_preprocessingu(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.pobierz('preprocessing', {}))
    
    def pobierz_konfiguracje_modeli(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.pobierz('models', {}))
    
    def pobierz_konfiguracje_monitoringu(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.pobierz('monitoring', {}))
    
    def pobierz_konfiguracje_jakosci_danych(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.pobierz('data_quality', {}))
