#!/usr/bin/env python3

import numpy as np
import logging
from typing import Dict, Any, Tuple
from .models import FabrykaModeli

logger = logging.getLogger(__name__)

class TrenerModeli:
    
    def __init__(self, konfiguracja: Dict[str, Any]):
        self.konfiguracja = konfiguracja
        self.modele = {}
        self.wyniki_treningu = {}
    
    def trenuj_modele(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        logger.info("=== TRENOWANIE MODELI ===")
        
        # Walidacja danych treningowych
        if X_train is None or y_train is None:
            raise ValueError("Dane treningowe są None")
        
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Puste dane treningowe")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Niezgodność rozmiarów danych treningowych")
        
        # Tworzenie modeli
        konfiguracja_modeli = FabrykaModeli.utworz_modele(self.konfiguracja)
        
        # Trenowanie każdego modelu
        for nazwa, model in konfiguracja_modeli.items():
            logger.info(f"Trenowanie modelu: {nazwa}")
            try:
                model.fit(X_train, y_train)
                self.modele[nazwa] = model
                logger.info(f"Model {nazwa} wytrenowany pomyślnie")
            except Exception as e:
                logger.error(f"Błąd trenowania modelu {nazwa}: {e}")
                raise
        
        logger.info(f"Wytrenowano {len(self.modele)} modeli")
        return self.modele
    
    def pobierz_najlepszy_model(self, wyniki_oceny: Dict[str, Any]) -> Tuple[str, Any]:
        if not wyniki_oceny:
            raise ValueError("Brak wyników oceny modeli")
        
        # Wybór modelu z najwyższym AUC
        nazwa_najlepszego = max(wyniki_oceny.keys(), 
                            key=lambda x: wyniki_oceny[x]['wynik_auc'])
        najlepszy_model = self.modele[nazwa_najlepszego]
        
        logger.info(f"Wybrano najlepszy model: {nazwa_najlepszego}")
        return nazwa_najlepszego, najlepszy_model
    
    def przetrenuj_model(self, model, X_train: np.ndarray, y_train: np.ndarray):
        if model is None:
            raise ValueError("Model nie został wytrenowany")
        
        logger.info("Ponowne trenowanie modelu...")
        model.fit(X_train, y_train)
        logger.info("Model wytrenowany ponownie")
