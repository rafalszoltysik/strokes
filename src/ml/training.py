#!/usr/bin/env python3

import numpy as np
import logging
from typing import Dict, Any, Tuple
from .models import ModelFactory

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Klasa do trenowania modeli ML"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicjalizacja trenera modeli"""
        self.config = config
        self.models = {}
        self.training_results = {}
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Trenowanie wszystkich modeli"""
        logger.info("=== TRENOWANIE MODELI ===")
        
        # Walidacja danych treningowych
        if X_train is None or y_train is None:
            raise ValueError("Dane treningowe są None")
        
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Puste dane treningowe")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Niezgodność rozmiarów danych treningowych")
        
        # Tworzenie modeli
        models_config = ModelFactory.create_models(self.config)
        
        # Trenowanie każdego modelu
        for name, model in models_config.items():
            logger.info(f"Trenowanie modelu: {name}")
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                logger.info(f"Model {name} wytrenowany pomyślnie")
            except Exception as e:
                logger.error(f"Błąd trenowania modelu {name}: {e}")
                raise
        
        logger.info(f"Wytrenowano {len(self.models)} modeli")
        return self.models
    
    def get_best_model(self, evaluation_results: Dict[str, Any]) -> Tuple[str, Any]:
        """Wybór najlepszego modelu na podstawie wyników oceny"""
        if not evaluation_results:
            raise ValueError("Brak wyników oceny modeli")
        
        # Wybór modelu z najwyższym AUC
        best_model_name = max(evaluation_results.keys(), 
                            key=lambda x: evaluation_results[x]['auc_score'])
        best_model = self.models[best_model_name]
        
        logger.info(f"Wybrano najlepszy model: {best_model_name}")
        return best_model_name, best_model
    
    def retrain_model(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """Ponowne trenowanie modelu na nowych danych"""
        if model is None:
            raise ValueError("Model nie został wytrenowany")
        
        logger.info("Ponowne trenowanie modelu...")
        model.fit(X_train, y_train)
        logger.info("Model wytrenowany ponownie")
