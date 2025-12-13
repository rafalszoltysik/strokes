# SYSTEM WSPIERANIA DECYZJI - PREDYKCJA UDARÓW MÓZGU

## SPIS TREŚCI

1. [Wprowadzenie](#wprowadzenie)
2. [Cel Systemu](#cel-systemu)
3. [Architektura Systemu](#architektura-systemu)
4. [Komponenty Systemu](#komponenty-systemu)
5. [Proces Decyzyjny](#proces-decyzyjny)
6. [Algorytmy i Metody](#algorytmy-i-metody)
7. [Interfejs Użytkownika](#interfejs-użytkownika)
8. [Wyniki i Ocena](#wyniki-i-ocena)
9. [Monitoring i Kontrola](#monitoring-i-kontrola)
10. [Wnioski i Rekomendacje](#wnioski-i-rekomendacje)

---

## WPROWADZENIE

### Definicja Systemu Wspierania Decyzji

System Wspierania Decyzji (DSS - Decision Support System) to zintegrowany system informatyczny, który wspomaga proces podejmowania decyzji poprzez:
- **Analizę danych** - przetwarzanie i analiza informacji
- **Modelowanie** - tworzenie modeli predykcyjnych
- **Wizualizację** - prezentacja wyników w zrozumiałej formie
- **Monitoring** - ciągłe śledzenie wydajności systemu

### Kontekst Medyczny

**Problem:** Udar mózgu to druga najczęstsza przyczyna zgonów na świecie i główna przyczyna niepełnosprawności. Wczesne wykrycie czynników ryzyka może znacząco poprawić rokowania pacjentów.

**Rozwiązanie:** System predykcji udarów wykorzystujący uczenie maszynowe do identyfikacji pacjentów wysokiego ryzyka.

---

## CEL SYSTEMU

### Główne Cele

1. **Predykcja Ryzyka** - Identyfikacja pacjentów z wysokim ryzykiem udaru
2. **Wsparcie Decyzyjne** - Dostarczenie lekarzom obiektywnych danych do podejmowania decyzji
3. **Optymalizacja Zasobów** - Skierowanie uwagi na pacjentów wymagających najpilniejszej interwencji
4. **Ciągłe Uczenie** - System adaptuje się do nowych danych i poprawia swoją dokładność

### Użytkownicy Systemu

- **Lekarze** - podejmowanie decyzji terapeutycznych
- **Personel medyczny** - planowanie opieki
- **Administratorzy** - zarządzanie zasobami szpitalnymi
- **Pacjenci** - świadomość własnego ryzyka

---

## ARCHITEKTURA SYSTEMU

### Struktura Projektu

```
strokes/
├── src/                          # Moduły systemu
│   ├── core/                     # Główne komponenty systemu
│   │   ├── system.py            # StrokePredictionSystem
│   │   └── pipeline.py          # PipelineOrchestrator
│   ├── data/                     # Przetwarzanie danych
│   │   └── preprocessing.py     # DataPreprocessor
│   ├── analysis/                # Analiza i monitoring
│   │   ├── visualization.py     # VisualizationAnalyzer
│   │   └── monitoring.py       # ModelMonitor
│   ├── ml/                      # Machine Learning
│   │   ├── models.py           # ModelFactory
│   │   ├── training.py         # ModelTrainer
│   │   └── evaluation.py       # ModelEvaluator
│   └── utils/                   # Narzędzia pomocnicze
│       ├── config.py           # ConfigManager
│       └── logging.py          # setup_logging
├── data/                         # Warstwa danych
│   ├── raw/                     # Dane surowe
│   └── processed/               # Dane przetworzone
├── results/                     # Warstwa wyników
│   ├── plots/                   # Wizualizacje
│   ├── models/                  # Modele ML
│   ├── monitoring/              # Raporty monitoringu
│   └── reports/                 # Raporty końcowe
├── config/                      # Konfiguracja
├── logs/                        # Logi systemu
├── requirements.txt             # Zależności Python
└── main.py                      # Główny punkt wejścia
```

### Architektura Warstwowa

```
┌─────────────────────────────────────┐
│           WARSTWA PREZENTACJI       │
│     (Raporty, Wykresy, Dashboard)   │
├─────────────────────────────────────┤
│           WARSTWA LOGICZNA          │
│   (Algorytmy ML, Monitoring, DSS)   │
├─────────────────────────────────────┤
│           WARSTWA DANYCH            │
│    (Pre-processing, Storage)        │
└─────────────────────────────────────┘
```

---

## KOMPONENTY SYSTEMU

### 1. System Główny (`src/core/system.py`)

**Klasa:** `StrokePredictionSystem`

**Funkcje:**
- Główna klasa systemu DSS
- Inicjalizacja wszystkich komponentów
- Zarządzanie konfiguracją
- Koordynacja procesów

**Kluczowe Metody:**
```python
def __init__(config_path)      # Inicjalizacja systemu
def setup_directories()        # Tworzenie struktury katalogów
def run_complete_pipeline()    # Uruchomienie pełnego pipeline'u
```

### 2. Orchestrator Pipeline (`src/core/pipeline.py`)

**Klasa:** `PipelineOrchestrator`

**Funkcje:**
- Orkiestracja całego procesu ML
- Koordynacja między modułami
- Zarządzanie błędami
- Generowanie raportów końcowych

### 3. Pre-processing Danych (`src/data/preprocessing.py`)

**Klasa:** `DataPreprocessor`

**Funkcje:**
- Oczyszczanie danych z problematycznych wartości
- Wykrywanie i obsługa anomalii
- Imputacja brakujących wartości
- Normalizacja i skalowanie cech
- Kodowanie zmiennych kategorycznych

### 4. Wizualizacja i Analiza (`src/analysis/visualization.py`)

**Klasa:** `VisualizationAnalyzer`

**Funkcje:**
- Analiza rozkładu cech
- Identyfikacja korelacji
- Wizualizacja wyników modelu
- Generowanie raportów analitycznych

### 5. Monitoring Modelu (`src/analysis/monitoring.py`)

**Klasa:** `ModelMonitor`

**Funkcje:**
- Wykrywanie driftu danych
- Monitoring wydajności modelu
- Wykrywanie halucynacji
- Analiza spójności predykcji

### 6. Machine Learning (`src/ml/`)

**Komponenty:**
- `models.py` - `ModelFactory` - Fabryka modeli ML
- `training.py` - `ModelTrainer` - Trenowanie modeli
- `evaluation.py` - `ModelEvaluator` - Ocena modeli

### 7. Narzędzia (`src/utils/`)

**Komponenty:**
- `config.py` - `ConfigManager` - Zarządzanie konfiguracją
- `logging.py` - `setup_logging` - Konfiguracja logowania

---

## PROCES DECYZYJNY

### Etapy Procesu Decyzyjnego

#### 1. **IDENTYFIKACJA PROBLEMU**
- Analiza danych pacjenta
- Identyfikacja czynników ryzyka
- Ocena aktualnego stanu zdrowia

#### 2. **GROMADZENIE INFORMACJI**
- Pre-processing danych medycznych
- Walidacja jakości danych
- Przygotowanie cech predykcyjnych

#### 3. **ANALIZA ALTERNATYW**
- Porównanie różnych modeli ML
- Ocena metryk wydajności
- Wybór najlepszego modelu

#### 4. **PODEJMOWANIE DECYZJI**
- Generowanie predykcji ryzyka
- Interpretacja wyników
- Rekomendacje terapeutyczne

#### 5. **MONITORING I KONTROLA**
- Śledzenie wydajności systemu
- Wykrywanie problemów
- Ciągłe ulepszanie

### Schemat Procesu Decyzyjnego

```
DANE PACJENTA → PRE-PROCESSING → MODEL ML → PREDYKCJA → REKOMENDACJA
     ↓              ↓              ↓           ↓            ↓
  Wiek, BMI,    Oczyszczenie,   Random      Prawdopod.   Interwencja
  Ciśnienie,    Normalizacja,  Forest,     ryzyka       medyczna
  Glukoza,      Kodowanie      Logistic    udaru        lub monitoring
  Choroby       kategorycznych Regression
```

---

## ALGORYTMY I METODY

### Algorytmy Uczenia Maszynowego

#### 1. **Random Forest**
- **Zalety:** Odporny na overfitting, interpretowalny
- **Zastosowanie:** Klasyfikacja, analiza ważności cech
- **Parametry:** n_estimators=100, random_state=42

#### 2. **Gradient Boosting**
- **Zalety:** Wysoka dokładność, obsługa nieliniowości
- **Zastosowanie:** Predykcja ryzyka, optymalizacja
- **Parametry:** random_state=42

#### 3. **Logistic Regression**
- **Zalety:** Interpretowalny, szybki, stabilny
- **Zastosowanie:** Analiza czynników ryzyka
- **Parametry:** max_iter=1000, random_state=42

#### 4. **Support Vector Machine (SVM)**
- **Zalety:** Skuteczny na małych zbiorach, odporny na wymiarowość
- **Zastosowanie:** Klasyfikacja binarna
- **Parametry:** probability=True, random_state=42

### Metody Oceny Modeli

#### Metryki Wydajności
- **Accuracy:** Ogólna dokładność klasyfikacji
- **AUC Score:** Zdolność rozróżniania między klasami (0.5=losowy, 0.8+=bardzo dobry)
- **Precision:** Ile z przewidzianych udarów to prawdziwe udary
- **Recall:** Ile z prawdziwych udarów zostało wykrytych
- **F1-Score:** Średnia harmoniczna precision i recall

#### Metody Walidacji
- **Train-Test Split:** Podział 80/20
- **Stratified Sampling:** Zachowanie proporcji klas
- **Cross-Validation:** Walidacja krzyżowa
- **SMOTE:** Zbalansowanie klas

### Techniki Pre-processing

#### Obsługa Braków Danych
- **Mediana według grupy wiekowej** (BMI)
- **Mediana ogólna** (inne cechy)
- **Walidacja zakresów** (wiek, BMI, glukoza)

#### Normalizacja Cech
- **StandardScaler:** Standaryzacja do średniej=0, odchylenie=1
- **LabelEncoder:** Kodowanie zmiennych kategorycznych
- **Feature Engineering:** Tworzenie nowych cech

---

## INTERFEJS UŻYTKOWNIKA

### Typy Interfejsów

#### 1. **Interfejs Programistyczny (API)**
```python
# Przykład użycia systemu
from src.core.system import StrokePredictionSystem
from src.core.pipeline import PipelineOrchestrator

# Inicjalizacja systemu
system = StrokePredictionSystem()
orchestrator = PipelineOrchestrator(system)

# Uruchomienie pipeline'u
results = orchestrator.run_pipeline("data/raw/healthcare-dataset-stroke-data.csv")
```

#### 2. **Interfejs Raportowy**
- **Raporty tekstowe:** Szczegółowe analizy
- **Wykresy:** Wizualizacje danych i wyników
- **Dashboard:** Monitoring w czasie rzeczywistym

#### 3. **Interfejs Konfiguracyjny**
- **Pliki YAML:** Konfiguracja parametrów
- **Logi systemu:** Śledzenie operacji
- **Struktura katalogów:** Organizacja wyników

### Struktura Wyjściowa

```
results/
├── plots/                    # Wizualizacje
│   ├── basic_distributions.png
│   ├── stroke_relationships.png
│   ├── correlation_matrix.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── monitoring_dashboard.png
├── models/                   # Modele ML
│   └── best_model_*.pkl
├── monitoring/              # Raporty monitoringu
│   ├── drift_analysis.json
│   ├── performance_metrics.json
│   └── hallucination_detection.json
└── reports/                 # Raporty końcowe
    ├── stroke_report.txt
    └── insights_report.json
```

---

## WYNIKI I OCENA

### Wydajność Systemu

#### Najlepsze Wyniki
- **Najlepszy Model:** Logistic Regression
- **AUC Score:** 0.8455 (bardzo dobra wydajność)
- **Accuracy:** 75.44%
- **Precision:** Wysoka precyzja w wykrywaniu udarów
- **Recall:** Dobra czułość w identyfikacji ryzyka

#### Interpretacja Metryk
- **AUC > 0.8:** Bardzo dobra zdolność rozróżniania
- **Accuracy 75%:** Dobra ogólna dokładność
- **Balanced Precision/Recall:** Optymalne dla medycyny

### Kluczowe Czynniki Ryzyka

#### Ranking Ważności Cech
1. **Wiek** - Najsilniejszy predyktor ryzyka
2. **Nadciśnienie** - Kluczowy czynnik kardiologiczny
3. **Choroby serca** - Bezpośredni wskaźnik ryzyka
4. **Poziom glukozy** - Wskaźnik metaboliczny
5. **BMI** - Wskaźnik otyłości
6. **Płeć** - Różnice w ryzyku między płciami

### Analiza Korelacji

#### Silne Korelacje
- Wiek ↔ Ryzyko udaru (pozytywna)
- Nadciśnienie ↔ Ryzyko udaru (pozytywna)
- Choroby serca ↔ Ryzyko udaru (pozytywna)

#### Słabe Korelacje
- Płeć ↔ Ryzyko udaru (umiarkowana)
- BMI ↔ Ryzyko udaru (umiarkowana)

---

## MONITORING I KONTROLA

### System Monitoringu

#### 1. **Monitoring Danych**
- **Drift Detection:** Wykrywanie zmian w rozkładzie cech
- **Data Quality:** Kontrola jakości nowych danych
- **Anomaly Detection:** Identyfikacja nietypowych przypadków

#### 2. **Monitoring Modelu**
- **Performance Drift:** Spadek wydajności w czasie
- **Prediction Drift:** Zmiany w rozkładzie predykcji
- **Feature Drift:** Zmiany w ważności cech

#### 3. **Wykrywanie Halucynacji**
- **Extreme Confidence:** Ekstremalne wartości pewności
- **Inconsistent Predictions:** Niespójne predykcje
- **Outlier Analysis:** Analiza wartości odstających

### Progi Alertów

#### Krytyczne Alerty
- **Drift danych > 5%:** Znaczące zmiany w danych
- **Spadek wydajności > 5%:** Pogorszenie modelu
- **Wykrycie halucynacji:** Potencjalne błędy systemu
- **Wysoki poziom anomalii:** Nietypowe przypadki

#### Monitoring Dashboard
- **Wykresy czasowe:** Trendy wydajności
- **Metryki w czasie rzeczywistym:** Aktualne wyniki
- **Alerty:** Powiadomienia o problemach
- **Rekomendacje:** Sugestie ulepszeń

---

## WNIOSKI I REKOMENDACJE

### Główne Wnioski

#### 1. **Skuteczność Systemu**
- System osiąga wysoką dokładność predykcji (AUC = 0.8455)
- Logistic Regression okazał się najlepszym modelem
- Kluczowe czynniki ryzyka zostały poprawnie zidentyfikowane

#### 2. **Wartość dla Decyzji Medycznych**
- System dostarcza obiektywne dane do podejmowania decyzji
- Identyfikuje pacjentów wysokiego ryzyka
- Wspomaga optymalizację zasobów medycznych

#### 3. **Jakość Implementacji**
- Profesjonalna struktura kodu
- Zaawansowany system monitoringu
- Kompleksowa dokumentacja

### Rekomendacje

#### 1. **Rozwój Systemu**
- **Integracja z systemami szpitalnymi:** Bezpośrednie połączenie z EHR
- **API REST:** Umożliwienie integracji z innymi systemami
- **Dashboard webowy:** Interfejs użytkownika dla lekarzy
- **Mobile app:** Dostęp dla personelu medycznego

#### 2. **Ulepszenia Algorytmiczne**
- **Ensemble Methods:** Kombinacja wielu modeli
- **Deep Learning:** Sieci neuronowe dla złożonych wzorców
- **Feature Engineering:** Dodatkowe cechy medyczne
- **Real-time Learning:** Adaptacja do nowych danych

#### 3. **Aspekty Praktyczne**
- **Walidacja kliniczna:** Testy z prawdziwymi pacjentami
- **Zgodność regulacyjna:** Spełnienie standardów medycznych
- **Szkolenie personelu:** Edukacja użytkowników
- **Dokumentacja medyczna:** Standardy dokumentacji

### Wartość Biznesowa

#### Korzyści Medyczne
- **Wczesne wykrycie:** Identyfikacja ryzyka przed wystąpieniem udaru
- **Optymalizacja opieki:** Skierowanie uwagi na pacjentów wysokiego ryzyka
- **Redukcja kosztów:** Zapobieganie kosztownym interwencjom
- **Poprawa wyników:** Lepsze rokowania pacjentów

#### Korzyści Operacyjne
- **Automatyzacja:** Redukcja pracy manualnej
- **Skalowalność:** Obsługa dużych populacji pacjentów
- **Monitoring:** Ciągłe śledzenie wydajności
- **Raportowanie:** Automatyczne generowanie raportów

---

## TECHNOLOGIE I NARZĘDZIA

### Stack Technologiczny

#### Języki i Frameworki
- **Python 3.8+:** Główny język programowania
- **Pandas:** Manipulacja danymi
- **NumPy:** Obliczenia numeryczne
- **Scikit-learn:** Uczenie maszynowe
- **Matplotlib/Seaborn:** Wizualizacja

#### Biblioteki Specjalistyczne
- **Imbalanced-learn:** Obsługa niezbalansowania klas (SMOTE, SMOTETomek)
- **Joblib:** Serializacja modeli
- **PyYAML:** Konfiguracja systemu


#### Zależności
```

# Core Data Science Libraries (używane w projekcie)
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization (używane w visualization.py)
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning - Imbalanced Data (używane w system.py)
imbalanced-learn>=0.11.0

# Configuration (używane w config.py)
pyyaml>=6.0

# Model Persistence (używane w system.py)
joblib>=1.3.0
```

---

## DOKUMENTACJA I WSPARCIE

### Struktura Dokumentacji

#### 1. **Dokumentacja Techniczna**
- **README.md:** Przewodnik użytkownika
- **Kod źródłowy:** Komentarze i docstringi
- **Konfiguracja:** Pliki YAML z opisami
- **Logi:** Szczegółowe logi systemu

#### 2. **Dokumentacja Medyczna**
- **Raporty analityczne:** Wnioski z analiz
- **Wizualizacje:** Wykresy i diagramy
- **Metryki wydajności:** Wyniki modeli
- **Rekomendacje:** Sugestie terapeutyczne

### Wsparcie i Utrzymanie

#### Monitoring Systemu
- **Logi aplikacji:** Śledzenie operacji
- **Metryki wydajności:** Monitoring modelu
- **Alerty:** Powiadomienia o problemach
- **Backup:** Regularne kopie zapasowe

#### Aktualizacje
- **Wersjonowanie:** Kontrola wersji kodu
- **Migracje:** Aktualizacja modeli
- **Testy:** Walidacja zmian
- **Deployment:** Wdrożenie nowych wersji

---

## PODSUMOWANIE DLA PRZEDMIOTU "SYSTEMY WSPIERANIA DECYZJI"

### Charakterystyka Systemu DSS

#### 1. **Typ Systemu**
- **System Wspierania Decyzji Medycznych**
- **System Predykcyjny** (Predictive DSS)
- **System Analityczny** (Analytical DSS)

#### 2. **Komponenty DSS**
- **Baza Danych:** Przetworzone dane medyczne
- **Model Decyzyjny:** Algorytmy uczenia maszynowego
- **Interfejs Użytkownika:** Raporty i wizualizacje
- **System Monitoringu:** Kontrola wydajności

#### 3. **Proces Decyzyjny**
- **Identyfikacja problemu:** Analiza ryzyka udaru
- **Gromadzenie danych:** Pre-processing medyczny
- **Analiza:** Algorytmy ML
- **Decyzja:** Rekomendacje terapeutyczne
- **Monitoring:** Kontrola wyników

### Wartość Edukacyjna

#### Aspekty Teoretyczne
- **Teoria DSS:** Praktyczna implementacja
- **Algorytmy ML:** Rzeczywiste zastosowania
- **Analiza danych:** Metody statystyczne
- **Systemy informacyjne:** Architektura systemu

#### Aspekty Praktyczne
- **Programowanie:** Kod w Python
- **Analiza danych:** Pandas, NumPy
- **Uczenie maszynowe:** Scikit-learn
- **Wizualizacja:** Matplotlib, Seaborn

### Zastosowania w Praktyce

#### Medycyna
- **Predykcja ryzyka:** Identyfikacja pacjentów wysokiego ryzyka
- **Wsparcie diagnostyczne:** Obiektywne dane dla lekarzy
- **Optymalizacja opieki:** Skierowanie zasobów
- **Badania kliniczne:** Analiza czynników ryzyka

#### Inne Dziedziny
- **Finanse:** Ocena ryzyka kredytowego
- **Marketing:** Segmentacja klientów
- **Produkcja:** Predykcja awarii
- **Logistyka:** Optymalizacja tras

