#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

# Dodanie ścieżki do modułów
sys.path.append('src')

from src.core.system import StrokePredictionSystem
from src.core.pipeline import PipelineOrchestrator
from src.utils.logging import ustaw_logowanie

# Konfiguracja logowania
logger = ustaw_logowanie(poziom="INFO")

def main():
    logger.info("SYSTEM PREDYKCJI UDARÓW")
    logger.info("="*60)
    
    # Inicjalizacja systemu
    system = StrokePredictionSystem()
    orkiestrator = PipelineOrchestrator(system)
    
    # Ścieżka do danych
    sciezka_danych = "data/raw/healthcare-dataset-stroke-data.csv"
    
    if not Path(sciezka_danych).exists():
        logger.error(f"BŁĄD: Plik danych nie istnieje: {sciezka_danych}")
        logger.error("Upewnij się, że plik healthcare-dataset-stroke-data.csv znajduje się w katalogu data/raw/")
        return
    
    logger.info(f"Uruchamianie pipeline'u z danymi: {sciezka_danych}")
    logger.info("="*60)
    
    # Uruchomienie pipeline'u
    wyniki = orkiestrator.uruchom_pipeline(sciezka_danych)
    
    if wyniki['sukces']:
        logger.info("\n" + "="*60)
        logger.info("PIPELINE ZAKOŃCZONY POMYŚLNIE!")
        logger.info("="*60)
        logger.info(f"Raport: {wyniki['sciezka_raportu']}")
        logger.info(f"Wyniki: {len(wyniki['wyniki'])} komponentów")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("PIPELINE ZAKOŃCZONY BŁĘDEM!")
        logger.error("="*60)
        logger.error(f"Błąd: {wyniki['blad']}")
        logger.error(f"Typ błędu: {wyniki['typ_bledu']}")
        logger.error("="*60)

if __name__ == "__main__":
    main()