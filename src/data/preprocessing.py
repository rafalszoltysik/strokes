#!/usr/bin/env python3

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PreprocessorDanych:
    
    def __init__(self, menedzer_konfiguracji=None):
        if menedzer_konfiguracji is None:
            from ..utils.config import ConfigManager
            menedzer_konfiguracji = ConfigManager()
        self.konfiguracja = menedzer_konfiguracji.konfiguracja
        self.skalownik = StandardScaler()
        self.kodery_etykiet = {}
        self.nazwy_cech = None
        self.statystyki_preprocessingu = {}
        
    
    def zaladuj_dane(self, sciezka_danych: str) -> pd.DataFrame:
        logger.info(f"Ładowanie danych z: {sciezka_danych}")
        try:
            df = pd.read_csv(sciezka_danych)
            logger.info(f"Załadowano {df.shape[0]} obserwacji z {df.shape[1]} cechami")
            return df
        except Exception as e:
            logger.error(f"Błąd podczas ładowania danych: {e}")
            raise
    
    def obsluz_problemy_jakosci_danych(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Rozpoczęcie obsługi problemów z jakością danych")
        df_przetworzony = df.copy()
        
        # 1. Obsługa problematycznych wartości
        wartosci_problematyczne = self.konfiguracja['data_quality']['problematic_values']
        for kolumna in df_przetworzony.columns:
            if df_przetworzony[kolumna].dtype == 'object':
                liczba_przed = df_przetworzony[kolumna].isnull().sum()
                df_przetworzony[kolumna] = df_przetworzony[kolumna].replace(wartosci_problematyczne, np.nan)
                liczba_po = df_przetworzony[kolumna].isnull().sum()
                if liczba_po > liczba_przed:
                    logger.info(f"Kolumna {kolumna}: znaleziono {liczba_po - liczba_przed} problematycznych wartości")
        
        # Obsługa NaN w kolumnach kategorycznych - zastąpienie przed dalszym przetwarzaniem
        for kolumna in df_przetworzony.columns:
            if df_przetworzony[kolumna].dtype == 'object' and df_przetworzony[kolumna].isnull().any():
                df_przetworzony[kolumna] = df_przetworzony[kolumna].fillna('Unknown')
                logger.info(f"Zastąpiono NaN w kolumnie {kolumna} wartością 'Unknown'")
        
        # 2. Obsługa nieprawidłowych wartości 0
        problemy_zerowe = {}
        for kolumna in ['age', 'bmi', 'avg_glucose_level']:
            if kolumna in df_przetworzony.columns:
                zera = (df_przetworzony[kolumna] == 0).sum()
                if zera > 0:
                    logger.info(f"Znaleziono {zera} przypadków {kolumna} = 0, zastępowanie medianą")
                    df_przetworzony.loc[df_przetworzony[kolumna] == 0, kolumna] = df_przetworzony[kolumna].median()
                    problemy_zerowe[kolumna] = zera
        
        # 3. Walidacja zakresów wartości
        nieprawidlowe_zakresy = self.konfiguracja['data_quality']['invalid_ranges']
        for kolumna, zakresy in nieprawidlowe_zakresy.items():
            if kolumna in df_przetworzony.columns:
                # Sprawdź tylko wartości numeryczne (nie NaN)
                maska_numeryczna = pd.notna(df_przetworzony[kolumna])
                liczba_nieprawidlowych = ((df_przetworzony[kolumna] < zakresy['min']) | 
                               (df_przetworzony[kolumna] > zakresy['max'])).sum()
                if liczba_nieprawidlowych > 0:
                    logger.info(f"Znaleziono {liczba_nieprawidlowych} przypadków nieprawidłowego {kolumna}, zastępowanie medianą")
                    maska_nieprawidlowa = (df_przetworzony[kolumna] < zakresy['min']) | (df_przetworzony[kolumna] > zakresy['max'])
                    df_przetworzony.loc[maska_nieprawidlowa, kolumna] = df_przetworzony[kolumna].median()
                    logger.info(f"Zastąpiono {liczba_nieprawidlowych} nieprawidłowych wartości {kolumna} medianą: {df_przetworzony[kolumna].median():.2f}")
        
        self.statystyki_preprocessingu['data_quality'] = {
            'zero_issues': problemy_zerowe,
            'invalid_ranges': nieprawidlowe_zakresy
        }
        
        logger.info("Zakończono obsługę problemów z jakością danych")
        return df_przetworzony
    
    def obsluz_braki_danych(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Rozpoczęcie obsługi braków danych")
        df_processed = df.copy()
        
        statystyki_brakow = {}
        for col in df_processed.columns:
            liczba_brakow = df_processed[col].isnull().sum()
            if liczba_brakow > 0:
                procent_brakow = (liczba_brakow / len(df_processed)) * 100
                statystyki_brakow[col] = {'count': liczba_brakow, 'percent': procent_brakow}
                logger.info(f"Kolumna {col}: {liczba_brakow} braków ({procent_brakow:.2f}%)")
                
                # Strategia obsługi braków danych
                if col == 'bmi':
                    # BMI to kluczowa cecha - imputujemy medianą według grupy wiekowej
                    logger.info(f"Imputacja BMI według grup wiekowych...")
                    
                    # Tworzenie grup wiekowych
                    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                                      bins=[0, 30, 50, 70, 120], 
                                                      labels=['young', 'middle', 'senior', 'elderly'])
                    
                    # Imputacja medianą według grupy wiekowej
                    for age_group in df_processed['age_group'].unique():
                        if pd.notna(age_group):
                            mask = (df_processed['age_group'] == age_group) & df_processed['bmi'].isna()
                            group_median = df_processed[df_processed['age_group'] == age_group]['bmi'].median()
                            if pd.notna(group_median):
                                df_processed.loc[mask, 'bmi'] = group_median
                                logger.info(f"Imputowano BMI dla grupy {age_group}: {mask.sum()} przypadków -> {group_median:.2f}")
                    
                    # Usunięcie pomocniczej kolumny
                    df_processed = df_processed.drop('age_group', axis=1)
                    
                    # Jeśli nadal są braki, użyj mediany ogólnej
                    remaining_nulls = df_processed['bmi'].isna().sum()
                    if remaining_nulls > 0:
                        overall_median = df_processed['bmi'].median()
                        df_processed['bmi'] = df_processed['bmi'].fillna(overall_median)
                        logger.info(f"Imputowano pozostałe {remaining_nulls} braków BMI medianą ogólną: {overall_median:.2f}")
                elif col == 'smoking_status':
                    # Specjalna obsługa dla smoking_status - imputacja według wieku i płci
                    logger.info(f"Imputacja smoking_status według wieku i płci...")
                    
                    # Tworzenie grup wiekowych dla smoking_status
                    df_processed['age_group_smoking'] = pd.cut(df_processed['age'], 
                                                               bins=[0, 30, 50, 70, 120], 
                                                               labels=['young', 'middle', 'senior', 'elderly'])
                    
                    # Imputacja według grupy wiekowej i płci
                    for age_group in df_processed['age_group_smoking'].unique():
                        if pd.notna(age_group):
                            for gender in df_processed['gender'].unique():
                                if pd.notna(gender):
                                    mask = ((df_processed['age_group_smoking'] == age_group) & 
                                           (df_processed['gender'] == gender) & 
                                           df_processed['smoking_status'].isna())
                                    
                                    if mask.sum() > 0:
                                        # Znajdź najczęstszy status w tej grupie
                                        group_data = df_processed[(df_processed['age_group_smoking'] == age_group) & 
                                                                (df_processed['gender'] == gender) & 
                                                                df_processed['smoking_status'].notna()]
                                        
                                        if len(group_data) > 0:
                                            most_common = group_data['smoking_status'].mode()
                                            if len(most_common) > 0:
                                                df_processed.loc[mask, 'smoking_status'] = most_common.iloc[0]
                                                logger.info(f"Imputowano smoking_status dla {age_group}/{gender}: {mask.sum()} przypadków -> {most_common.iloc[0]}")
                    
                    # Usunięcie pomocniczej kolumny
                    df_processed = df_processed.drop('age_group_smoking', axis=1)
                    
                    # Jeśli nadal są braki, użyj najczęstszego statusu ogólnie
                    remaining_nulls = df_processed['smoking_status'].isna().sum()
                    if remaining_nulls > 0:
                        most_common_overall = df_processed['smoking_status'].mode()
                        if len(most_common_overall) > 0:
                            df_processed['smoking_status'] = df_processed['smoking_status'].fillna(most_common_overall.iloc[0])
                            logger.info(f"Imputowano pozostałe {remaining_nulls} braków smoking_status najczęstszym statusem: {most_common_overall.iloc[0]}")
                        else:
                            df_processed['smoking_status'] = df_processed['smoking_status'].fillna('Unknown')
                            logger.info(f"Imputowano pozostałe {remaining_nulls} braków smoking_status jako 'Unknown'")
                            
                elif col in ['work_type', 'gender', 'ever_married', 'Residence_type']:
                    # Dla innych kolumn kategorycznych - zostaw NaN, obsłużymy w kodowaniu
                    logger.info(f"Kolumna kategoryczna {col} - pomijamy imputację, obsłużymy w kodowaniu")
                else:
                    # Dla innych kolumn numerycznych - mediana
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Aktualizacja statystyk po imputacji
        liczba_koncowa = len(df_processed)
        self.statystyki_preprocessingu['missing_values'] = statystyki_brakow
        self.statystyki_preprocessingu['data_retention'] = {
            'initial_rows': len(df),
            'final_rows': liczba_koncowa,
            'removed_rows': len(df) - liczba_koncowa,
            'retention_rate': (liczba_koncowa / len(df)) * 100
        }
        
        # Walidacja po imputacji
        logger.info("=== WALIDACJA PO IMPUTACJI ===")
        for col in df_processed.columns:
            remaining_nulls = df_processed[col].isnull().sum()
            if remaining_nulls > 0:
                logger.warning(f"UWAGA: Pozostało {remaining_nulls} braków w kolumnie {col}")
            else:
                logger.info(f"Kolumna {col}: brak braków danych")
        
        # Sprawdzenie zakresów wartości po imputacji
        for col, ranges in self.konfiguracja['data_quality']['invalid_ranges'].items():
            if col in df_processed.columns:
                invalid_count = ((df_processed[col] < ranges['min']) | 
                               (df_processed[col] > ranges['max'])).sum()
                if invalid_count > 0:
                    logger.warning(f"UWAGA: {invalid_count} wartości {col} poza zakresem [{ranges['min']}, {ranges['max']}]")
                else:
                    logger.info(f"Kolumna {col}: wszystkie wartości w prawidłowym zakresie")
        
        logger.info(f"Zakończono obsługę braków danych")
        logger.info(f"Zachowano {liczba_koncowa} wierszy z {len(df)} ({self.statystyki_preprocessingu['data_retention']['retention_rate']:.1f}%)")
        return df_processed
    
    def zakoduj_zmienne_kategoryczne(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Rozpoczęcie kodowania zmiennych kategorycznych")
        df_processed = df.copy()
        
        # Label encoding dla zmiennych binarnych
        binary_columns = ['gender', 'ever_married', 'Residence_type']
        for col in binary_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                # Obsługa wartości NaN przed kodowaniem
                df_processed[col] = df_processed[col].fillna('Unknown')
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.kodery_etykiet[col] = le
                logger.info(f"Zakodowano kolumnę {col} (Label Encoding)")
        
        # One-hot encoding dla zmiennych nominalnych
        nominal_columns = ['work_type', 'smoking_status']
        for col in nominal_columns:
            if col in df_processed.columns:
                # Obsługa wartości NaN przed one-hot encoding
                df_processed[col] = df_processed[col].fillna('Unknown')
                dummies = pd.get_dummies(df_processed[col], prefix=col)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed = df_processed.drop(col, axis=1)
                logger.info(f"Zakodowano kolumnę {col} (One-Hot Encoding) -> {dummies.shape[1]} nowych kolumn")
        
        logger.info("Zakończono kodowanie zmiennych kategorycznych")
        return df_processed
    
    def przygotuj_cechy_i_cel(self, df: pd.DataFrame, kolumna_celu: str = 'stroke') -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Przygotowanie cech i zmiennej docelowej (cel: {kolumna_celu})")
        
        # Usunięcie kolumny ID
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            logger.info("Usunięto kolumnę ID")
        
        # Podział na cechy i zmienną docelową
        X = df.drop(kolumna_celu, axis=1)
        y = df[kolumna_celu]
        
        self.nazwy_cech = X.columns.tolist()
        logger.info(f"Przygotowano {X.shape[1]} cech i {len(y)} obserwacji")
        
        return X, y
    
    def podziel_i_skaluj_dane(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Rozpoczęcie podziału danych i normalizacji")
        
        # Podział na zbiory treningowe i testowe
        rozmiar_testu = self.konfiguracja['preprocessing']['test_size']
        losowy_stan = self.konfiguracja['preprocessing']['random_state']
        stratyfikacja = self.konfiguracja['preprocessing']['stratify']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=rozmiar_testu, random_state=losowy_stan, stratify=y if stratyfikacja else None
        )
        
        # Normalizacja cech
        X_train_scaled = self.skalownik.fit_transform(X_train)
        X_test_scaled = self.skalownik.transform(X_test)
        
        logger.info(f"Podzielono dane: treningowe {X_train.shape[0]}, testowe {X_test.shape[0]}")
        logger.info("Zakończono normalizację cech")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    
    def utworz_czysty_zbior(self, sciezka_danych: str, sciezka_wyjscia: str = "data/processed/clean_dataset.csv") -> str:
        logger.info("=== TWORZENIE OCZYSZCZONEGO ZBIORU DANYCH ===")
        
        # 1. Ładowanie danych
        df = self.zaladuj_dane(sciezka_danych)
        logger.info(f"Załadowano {len(df)} wierszy")
        
        # 2. Obsługa problemów z jakością
        df = self.obsluz_problemy_jakosci_danych(df)
        
        # 3. Obsługa braków danych
        df = self.obsluz_braki_danych(df)
        logger.info(f"Po obsłudze braków: {len(df)} wierszy")
        
        # 4. Kodowanie zmiennych kategorycznych
        df = self.zakoduj_zmienne_kategoryczne(df)
        
        # 5. Zapisanie oczyszczonego zbioru
        df.to_csv(sciezka_wyjscia, index=False)
        logger.info(f"Zapisano oczyszczony zbiór: {sciezka_wyjscia}")
        logger.info(f"Kolumny: {list(df.columns)}")
        logger.info(f"Rozmiar: {df.shape}")
        
        return sciezka_wyjscia

    def waliduj_jakosc_danych(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("=== WALIDACJA JAKOŚCI DANYCH ===")
        
        validation_results = {
            'total_rows': len(df),
            'missing_values': {},
            'data_types': {},
            'value_ranges': {},
            'categorical_values': {},
            'warnings': [],
            'errors': []
        }
        
        # Sprawdzenie braków danych
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            validation_results['missing_values'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }
            if missing_count > 0:
                validation_results['warnings'].append(f"Kolumna {col}: {missing_count} braków danych")
        
        # Sprawdzenie typów danych
        for col in df.columns:
            validation_results['data_types'][col] = str(df[col].dtype)
        
        # Sprawdzenie zakresów wartości dla kolumn numerycznych
        for col in ['age', 'bmi', 'avg_glucose_level']:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                validation_results['value_ranges'][col] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': df[col].mean(),
                    'median': df[col].median()
                }
                
                # Sprawdzenie czy wartości są w rozsądnych zakresach
                if col == 'age' and (min_val < 0 or max_val > 120):
                    validation_results['errors'].append(f"Wiek poza zakresem: {min_val}-{max_val}")
                elif col == 'bmi' and (min_val < 10 or max_val > 100):
                    validation_results['warnings'].append(f"BMI poza zakresem: {min_val}-{max_val}")
                elif col == 'avg_glucose_level' and (min_val < 50 or max_val > 600):
                    validation_results['warnings'].append(f"Glukoza poza zakresem: {min_val}-{max_val}")
        
        # Sprawdzenie wartości kategorycznych
        for col in ['gender', 'smoking_status', 'work_type']:
            if col in df.columns:
                unique_values = df[col].unique()
                validation_results['categorical_values'][col] = {
                    'unique_count': len(unique_values),
                    'values': list(unique_values)
                }
        
        # Podsumowanie
        if validation_results['warnings']:
            logger.warning(f"Znaleziono {len(validation_results['warnings'])} ostrzeżeń")
        if validation_results['errors']:
            logger.error(f"Znaleziono {len(validation_results['errors'])} błędów")
        
        logger.info("Zakończono walidację jakości danych")
        return validation_results

    def przetworz_dane(self, sciezka_danych: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        logger.info("=== ROZPOCZĘCIE PRZETWARZANIA DANYCH ===")
        
        # 1. Ładowanie danych
        df = self.zaladuj_dane(sciezka_danych)
        
        # 2. Obsługa problemów z jakością
        df = self.obsluz_problemy_jakosci_danych(df)
        
        # 3. Obsługa braków danych
        df = self.obsluz_braki_danych(df)
        
        # 4. Kodowanie zmiennych kategorycznych
        df = self.zakoduj_zmienne_kategoryczne(df)
        
        # 5. Walidacja jakości danych po preprocessingu
        wyniki_walidacji = self.waliduj_jakosc_danych(df)
        self.statystyki_preprocessingu['validation'] = wyniki_walidacji
        
        # 6. Przygotowanie cech i zmiennej docelowej
        X, y = self.przygotuj_cechy_i_cel(df)
        
        # 7. Podział i normalizacja
        X_train, X_test, y_train, y_test = self.podziel_i_skaluj_dane(X, y)
        
        # 8. Zapisanie przetworzonych danych - TYLKO STATYSTYKI
        if self.konfiguracja['output']['save_intermediate']:
            # Zapisanie tylko statystyk preprocessingu
            import json
            sciezka_wyjscia = Path(self.konfiguracja['output']['processed_data_path'])
            sciezka_wyjscia.mkdir(parents=True, exist_ok=True)
            
            with open(sciezka_wyjscia / 'preprocessing_stats.json', 'w') as plik:
                json.dump(self.statystyki_preprocessingu, plik, indent=2, default=str)
            
            logger.info(f"Zapisano statystyki preprocessingu w: {sciezka_wyjscia}")
        
        logger.info("=== ZAKOŃCZENIE PRZETWARZANIA DANYCH ===")
        
        # Zwróć nazwy_cech jako listę lub pustą listę jeśli None
        lista_nazw_cech = self.nazwy_cech if self.nazwy_cech is not None else []
        return X_train, X_test, y_train, y_test, lista_nazw_cech

if __name__ == "__main__":
    # Przykład użycia
    preprocessor = PreprocessorDanych()
    X_train, X_test, y_train, y_test, nazwy_cech = preprocessor.przetworz_dane("data/raw/healthcare-dataset-stroke-data.csv")
    logger.info(f"Przetworzono dane: {X_train.shape[0]} próbek treningowych, {X_test.shape[0]} testowych")
