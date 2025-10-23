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

class DataPreprocessor:
    """
    Data preprocessor for stroke prediction dataset
    """
    
    def __init__(self, config_manager=None):
        """Inicjalizacja preprocessora z konfiguracją"""
        if config_manager is None:
            from ..utils.config import ConfigManager
            config_manager = ConfigManager()
        self.config = config_manager.config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.preprocessing_stats = {}
        
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Ładowanie danych z pliku CSV"""
        logger.info(f"Ładowanie danych z: {data_path}")
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Załadowano {df.shape[0]} obserwacji z {df.shape[1]} cechami")
            return df
        except Exception as e:
            logger.error(f"Błąd podczas ładowania danych: {e}")
            raise
    
    def handle_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Obsługa problemów z jakością danych"""
        logger.info("Rozpoczęcie obsługi problemów z jakością danych")
        df_processed = df.copy()
        
        # 1. Obsługa problematycznych wartości
        problematic_values = self.config['data_quality']['problematic_values']
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                before_count = df_processed[col].isnull().sum()
                df_processed[col] = df_processed[col].replace(problematic_values, np.nan)
                after_count = df_processed[col].isnull().sum()
                if after_count > before_count:
                    logger.info(f"Kolumna {col}: znaleziono {after_count - before_count} problematycznych wartości")
        
        # Obsługa NaN w kolumnach kategorycznych - zastąpienie przed dalszym przetwarzaniem
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna('Unknown')
                logger.info(f"Zastąpiono NaN w kolumnie {col} wartością 'Unknown'")
        
        # 2. Obsługa nieprawidłowych wartości 0
        zero_issues = {}
        for col in ['age', 'bmi', 'avg_glucose_level']:
            if col in df_processed.columns:
                zeros = (df_processed[col] == 0).sum()
                if zeros > 0:
                    logger.info(f"Znaleziono {zeros} przypadków {col} = 0, zastępowanie medianą")
                    df_processed.loc[df_processed[col] == 0, col] = df_processed[col].median()
                    zero_issues[col] = zeros
        
        # 3. Walidacja zakresów wartości
        invalid_ranges = self.config['data_quality']['invalid_ranges']
        for col, ranges in invalid_ranges.items():
            if col in df_processed.columns:
                # Sprawdź tylko wartości numeryczne (nie NaN)
                numeric_mask = pd.notna(df_processed[col])
                invalid_count = ((df_processed[col] < ranges['min']) | 
                               (df_processed[col] > ranges['max'])).sum()
                if invalid_count > 0:
                    logger.info(f"Znaleziono {invalid_count} przypadków nieprawidłowego {col}, zastępowanie medianą")
                    invalid_mask = (df_processed[col] < ranges['min']) | (df_processed[col] > ranges['max'])
                    df_processed.loc[invalid_mask, col] = df_processed[col].median()
                    logger.info(f"Zastąpiono {invalid_count} nieprawidłowych wartości {col} medianą: {df_processed[col].median():.2f}")
        
        self.preprocessing_stats['data_quality'] = {
            'zero_issues': zero_issues,
            'invalid_ranges': invalid_ranges
        }
        
        logger.info("Zakończono obsługę problemów z jakością danych")
        return df_processed
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Obsługa braków danych"""
        logger.info("Rozpoczęcie obsługi braków danych")
        df_processed = df.copy()
        
        missing_stats = {}
        for col in df_processed.columns:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                missing_percent = (missing_count / len(df_processed)) * 100
                missing_stats[col] = {'count': missing_count, 'percent': missing_percent}
                logger.info(f"Kolumna {col}: {missing_count} braków ({missing_percent:.2f}%)")
                
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
        final_count = len(df_processed)
        self.preprocessing_stats['missing_values'] = missing_stats
        self.preprocessing_stats['data_retention'] = {
            'initial_rows': len(df),
            'final_rows': final_count,
            'removed_rows': len(df) - final_count,
            'retention_rate': (final_count / len(df)) * 100
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
        for col, ranges in self.config['data_quality']['invalid_ranges'].items():
            if col in df_processed.columns:
                invalid_count = ((df_processed[col] < ranges['min']) | 
                               (df_processed[col] > ranges['max'])).sum()
                if invalid_count > 0:
                    logger.warning(f"UWAGA: {invalid_count} wartości {col} poza zakresem [{ranges['min']}, {ranges['max']}]")
                else:
                    logger.info(f"Kolumna {col}: wszystkie wartości w prawidłowym zakresie")
        
        logger.info(f"Zakończono obsługę braków danych")
        logger.info(f"Zachowano {final_count} wierszy z {len(df)} ({self.preprocessing_stats['data_retention']['retention_rate']:.1f}%)")
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kodowanie zmiennych kategorycznych"""
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
                self.label_encoders[col] = le
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
    
    def prepare_features_and_target(self, df: pd.DataFrame, target_column: str = 'stroke') -> Tuple[pd.DataFrame, pd.Series]:
        """Przygotowanie cech i zmiennej docelowej"""
        logger.info(f"Przygotowanie cech i zmiennej docelowej (target: {target_column})")
        
        # Usunięcie kolumny ID
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            logger.info("Usunięto kolumnę ID")
        
        # Podział na cechy i zmienną docelową
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        logger.info(f"Przygotowano {X.shape[1]} cech i {len(y)} obserwacji")
        
        return X, y
    
    def split_and_scale_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Podział danych i normalizacja"""
        logger.info("Rozpoczęcie podziału danych i normalizacji")
        
        # Podział na zbiory treningowe i testowe
        test_size = self.config['preprocessing']['test_size']
        random_state = self.config['preprocessing']['random_state']
        stratify = self.config['preprocessing']['stratify']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
        )
        
        # Normalizacja cech
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Podzielono dane: treningowe {X_train.shape[0]}, testowe {X_test.shape[0]}")
        logger.info("Zakończono normalizację cech")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    
    def create_clean_dataset(self, data_path: str, output_path: str = "data/processed/clean_dataset.csv") -> str:
        """Tworzenie jednego oczyszczonego pliku CSV"""
        logger.info("=== TWORZENIE OCZYSZCZONEGO ZBIORU DANYCH ===")
        
        # 1. Ładowanie danych
        df = self.load_data(data_path)
        logger.info(f"Załadowano {len(df)} wierszy")
        
        # 2. Obsługa problemów z jakością
        df = self.handle_data_quality_issues(df)
        
        # 3. Obsługa braków danych
        df = self.handle_missing_values(df)
        logger.info(f"Po obsłudze braków: {len(df)} wierszy")
        
        # 4. Kodowanie zmiennych kategorycznych
        df = self.encode_categorical_variables(df)
        
        # 5. Zapisanie oczyszczonego zbioru
        df.to_csv(output_path, index=False)
        logger.info(f"Zapisano oczyszczony zbiór: {output_path}")
        logger.info(f"Kolumny: {list(df.columns)}")
        logger.info(f"Rozmiar: {df.shape}")
        
        return output_path

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Walidacja jakości danych po preprocessingu"""
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

    def process_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Główna metoda przetwarzania danych"""
        logger.info("=== ROZPOCZĘCIE PRZETWARZANIA DANYCH ===")
        
        # 1. Ładowanie danych
        df = self.load_data(data_path)
        
        # 2. Obsługa problemów z jakością
        df = self.handle_data_quality_issues(df)
        
        # 3. Obsługa braków danych
        df = self.handle_missing_values(df)
        
        # 4. Kodowanie zmiennych kategorycznych
        df = self.encode_categorical_variables(df)
        
        # 5. Walidacja jakości danych po preprocessingu
        validation_results = self.validate_data_quality(df)
        self.preprocessing_stats['validation'] = validation_results
        
        # 6. Przygotowanie cech i zmiennej docelowej
        X, y = self.prepare_features_and_target(df)
        
        # 7. Podział i normalizacja
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y)
        
        # 8. Zapisanie przetworzonych danych - TYLKO STATYSTYKI
        if self.config['output']['save_intermediate']:
            # Zapisanie tylko statystyk preprocessingu
            import json
            output_path = Path(self.config['output']['processed_data_path'])
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / 'preprocessing_stats.json', 'w') as f:
                json.dump(self.preprocessing_stats, f, indent=2, default=str)
            
            logger.info(f"Zapisano statystyki preprocessingu w: {output_path}")
        
        logger.info("=== ZAKOŃCZENIE PRZETWARZANIA DANYCH ===")
        
        # Zwróć feature_names jako listę lub pustą listę jeśli None
        feature_names_list = self.feature_names if self.feature_names is not None else []
        return X_train, X_test, y_train, y_test, feature_names_list

if __name__ == "__main__":
    # Przykład użycia
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.process_data("data/raw/healthcare-dataset-stroke-data.csv")
    print(f"Przetworzono dane: {X_train.shape[0]} próbek treningowych, {X_test.shape[0]} testowych")
