#!/usr/bin/env python3

import logging
import sys
from pathlib import Path
from typing import Optional

def ustaw_logowanie(poziom: str = "INFO", plik_logu: Optional[str] = None) -> logging.Logger:
    
    # Tworzenie katalogu logs jeśli nie istnieje
    if plik_logu:
        Path(plik_logu).parent.mkdir(parents=True, exist_ok=True)
    
    # Konfiguracja formatu logowania
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Konfiguracja handlerów
    handlery = [logging.StreamHandler(sys.stdout)]
    
    if plik_logu:
        handler_pliku = logging.FileHandler(plik_logu, encoding='utf-8')
        handler_pliku.setFormatter(formatter)
        handlery.append(handler_pliku)
    
    # Konfiguracja głównego loggera
    logging.basicConfig(
        level=getattr(logging, poziom.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlery
    )
    
    # Wyłączenie niepotrzebnych logów
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def pobierz_logger(nazwa: str) -> logging.Logger:
    return logging.getLogger(nazwa)
