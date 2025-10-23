#!/usr/bin/env python3

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manager konfiguracji systemu"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Inicjalizacja managera konfiguracji"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Ładowanie konfiguracji z pliku YAML"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Zaaladowano konfigurację z: {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Błąd ładowania konfiguracji: {e}")
                return self._get_default_config()
        else:
            logger.warning(f"Plik konfiguracji {self.config_path} nie istnieje, używanie domyślnej konfiguracji")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Domyślna konfiguracja systemu"""
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
    
    def get(self, key: str, default=None):
        """Pobieranie wartości z konfiguracji"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Pobieranie konfiguracji preprocessingu"""
        return self.get('preprocessing', {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """Pobieranie konfiguracji modeli"""
        return self.get('models', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Pobieranie konfiguracji monitoringu"""
        return self.get('monitoring', {})
    
    def get_data_quality_config(self) -> Dict[str, Any]:
        """Pobieranie konfiguracji jakości danych"""
        return self.get('data_quality', {})
