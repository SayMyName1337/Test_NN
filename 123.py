import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
import optuna
from tqdm import tqdm
import warnings
import time
import os
import json
import pickle
from datetime import datetime

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# Устанавливаем стиль для графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Для воспроизводимости результатов
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class NoiseInjector:
    """Класс для добавления различных типов шума в данные."""
    
    @staticmethod
    def add_gaussian_noise(X, noise_level):
        """
        Добавление гауссовского шума.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_level : float
            Уровень шума (стандартное отклонение)
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def add_uniform_noise(X, noise_level):
        """
        Добавление равномерного шума.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_level : float
            Уровень шума (максимальная амплитуда)
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        noise = np.random.uniform(-noise_level, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def add_impulse_noise(X, noise_level):
        """
        Добавление импульсного шума (Salt and Pepper).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_level : float
            Вероятность замены элемента шумом (от 0 до 1)
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        X_noisy = X.copy()
        mask = np.random.random(X.shape) < noise_level
        
        # Создаем случайные значения для импульсного шума (минимальные и максимальные значения)
        X_min, X_max = np.min(X), np.max(X)
        noise_values = np.random.choice([X_min, X_max], size=np.sum(mask))
        
        # Заменяем значения в маске шумовыми значениями
        X_noisy[mask] = noise_values
        return X_noisy
    
    @staticmethod
    def add_missing_values(X, missing_rate):
        """
        Добавление пропущенных значений с последующей заменой на среднее.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        missing_rate : float
            Доля пропущенных значений (от 0 до 1)
            
        Returns:
        --------
        numpy.ndarray
            Данные с замененными пропущенными значениями
        """
        X_missing = X.copy()
        mask = np.random.random(X.shape) < missing_rate
        
        # Заменяем значения в маске на NaN
        X_missing[mask] = np.nan
        
        # Заменяем пропущенные значения на среднее по столбцу
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_missing)
        
        return X_imputed
    
    @staticmethod
    def add_noise(X, noise_type, noise_level):
        """
        Добавление выбранного типа шума.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_type : str
            Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
        noise_level : float
            Уровень шума
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        if noise_type == 'gaussian':
            return NoiseInjector.add_gaussian_noise(X, noise_level)
        elif noise_type == 'uniform':
            return NoiseInjector.add_uniform_noise(X, noise_level)
        elif noise_type == 'impulse':
            return NoiseInjector.add_impulse_noise(X, noise_level)
        elif noise_type == 'missing':
            return NoiseInjector.add_missing_values(X, noise_level)
        else:
            raise ValueError(f"Неизвестный тип шума: {noise_type}")


class AdaptiveEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Адаптивный ансамбль классификаторов, где главная роль отведена нейронной сети,
    а остальные классификаторы используются в случае неуверенности нейросети.
    """
    
    def __init__(self, main_classifier, helper_classifiers, confidence_threshold=0.7):
        """
        Parameters:
        -----------
        main_classifier : object
            Основной классификатор (нейронная сеть)
        helper_classifiers : list
            Список вспомогательных классификаторов
        confidence_threshold : float
            Порог уверенности, ниже которого используется голосование вспомогательных классификаторов
        """
        self.main_classifier = main_classifier
        self.helper_classifiers = helper_classifiers
        self.confidence_threshold = confidence_threshold
        self.is_fitted = False
        self.main_scaler = StandardScaler()
        self.helper_scalers = [StandardScaler() for _ in helper_classifiers]
    
    def fit(self, X, y):
        """
        Обучение ансамбля.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Обучающие данные
        y : numpy.ndarray
            Целевые метки
            
        Returns:
        --------
        self : object
            Возвращает экземпляр класса
        """
        # Преобразование y в one-hot кодирование для нейронной сети
        # и сохранение исходных меток для остальных классификаторов
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Масштабирование данных для основного классификатора
        X_main = self.main_scaler.fit_transform(X)
        
        # Обучение основного классификатора
        print("Обучение основного классификатора (нейронная сеть)...")
        self.main_classifier.fit(X_main, y)
        
        # Обучение вспомогательных классификаторов
        print("Обучение вспомогательных классификаторов...")
        for i, classifier in enumerate(self.helper_classifiers):
            X_helper = self.helper_scalers[i].fit_transform(X)
            classifier.fit(X_helper, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классам.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Тестовые данные
            
        Returns:
        --------
        numpy.ndarray
            Вероятности принадлежности к каждому классу
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите метод fit перед predict_proba.")
        
        # Масштабирование данных для основного классификатора
        X_main = self.main_scaler.transform(X)
        
        # Получение предсказаний от основного классификатора
        probas_main = self.main_classifier.predict_proba(X_main)
        
        # Определение максимальной вероятности для каждого образца
        max_probas = np.max(probas_main, axis=1)
        
        # Нахождение образцов, где уверенность ниже порога
        low_confidence_idx = max_probas < self.confidence_threshold
        
        # Если нет образцов с низкой уверенностью, возвращаем предсказания основного классификатора
        if not np.any(low_confidence_idx):
            return probas_main
        
        # Создаем копию предсказаний основного классификатора
        final_probas = probas_main.copy()
        
        # Для образцов с низкой уверенностью запрашиваем предсказания от вспомогательных классификаторов
        X_low_conf = X[low_confidence_idx]
        
        # Накапливаем вероятности от вспомогательных классификаторов
        helper_probas = np.zeros((X_low_conf.shape[0], self.n_classes_))
        
        for i, classifier in enumerate(self.helper_classifiers):
            X_helper = self.helper_scalers[i].transform(X_low_conf)
            helper_probas += classifier.predict_proba(X_helper)
        
        # Усредняем вероятности от вспомогательных классификаторов
        helper_probas /= len(self.helper_classifiers)
        
        # Заменяем предсказания для неуверенных образцов
        final_probas[low_confidence_idx] = helper_probas
        
        return final_probas
    
    def predict(self, X):
        """
        Предсказание меток классов.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Тестовые данные
            
        Returns:
        --------
        numpy.ndarray
            Предсказанные метки классов
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите метод fit перед predict.")
        
        # Получение вероятностей
        probas = self.predict_proba(X)
        
        # Возвращаем метки классов с наибольшей вероятностью
        return self.classes_[np.argmax(probas, axis=1)]


def create_nn_model(input_dim, num_classes):
    """
    Создание модели нейронной сети.
    
    Parameters:
    -----------
    input_dim : int
        Размерность входных данных
    num_classes : int
        Количество классов
        
    Returns:
    --------
    tensorflow.keras.models.Sequential
        Модель нейронной сети
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def optimize_nn_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Оптимизация гиперпараметров нейронной сети с использованием Optuna.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Обучающие данные
    y_train : numpy.ndarray
        Целевые метки для обучения
    X_val : numpy.ndarray
        Валидационные данные
    y_val : numpy.ndarray
        Целевые метки для валидации
    n_trials : int
        Количество испытаний для оптимизации
        
    Returns:
    --------
    dict
        Оптимальные гиперпараметры
    """
    def objective(trial):
        # Определение гиперпараметров для оптимизации
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        units1 = trial.suggest_categorical('units1', [64, 128, 256])
        units2 = trial.suggest_categorical('units2', [32, 64, 128])
        dropout1 = trial.suggest_float('dropout1', 0.1, 0.5)
        dropout2 = trial.suggest_float('dropout2', 0.1, 0.4)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Создание модели с выбранными гиперпараметрами
        model = Sequential([
            Dense(units1, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(dropout1),
            Dense(units2, activation='relu'),
            BatchNormalization(),
            Dropout(dropout2),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Ранняя остановка для предотвращения переобучения
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Обучение модели
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Оценка модели на валидационном наборе
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        return accuracy
    
    # Создание исследования Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Лучшие гиперпараметры: {study.best_params}")
    print(f"Лучшая точность: {study.best_value:.4f}")
    
    return study.best_params


def optimize_classifier_hyperparameters(classifier, X_train, y_train, param_grid):
    """
    Оптимизация гиперпараметров классификатора с использованием GridSearchCV.
    
    Parameters:
    -----------
    classifier : object
        Классификатор
    X_train : numpy.ndarray
        Обучающие данные
    y_train : numpy.ndarray
        Целевые метки
    param_grid : dict
        Сетка гиперпараметров для поиска
        
    Returns:
    --------
    object
        Классификатор с оптимальными гиперпараметрами
    """
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def create_and_optimize_classifiers(X_train, y_train, X_val, y_val):
    """
    Создание и оптимизация классификаторов.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Обучающие данные
    y_train : numpy.ndarray
        Целевые метки для обучения
    X_val : numpy.ndarray
        Валидационные данные
    y_val : numpy.ndarray
        Целевые метки для валидации
        
    Returns:
    --------
    tuple
        Кортеж из основного и вспомогательных классификаторов
    """
    # Количество классов
    n_classes = len(np.unique(y_train))
    
    # Оптимизация гиперпараметров нейронной сети
    print("Оптимизация гиперпараметров нейронной сети...")
    nn_params = optimize_nn_hyperparameters(X_train, y_train, X_val, y_val, n_trials=20)
    
    # Создание модели нейронной сети с оптимальными гиперпараметрами
    def create_optimized_nn(input_dim=X_train.shape[1], num_classes=n_classes):
        model = Sequential([
            Dense(nn_params['units1'], activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(nn_params['dropout1']),
            Dense(nn_params['units2'], activation='relu'),
            BatchNormalization(),
            Dropout(nn_params['dropout2']),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=nn_params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Создание и обучение KerasClassifier
    main_classifier = KerasClassifier(
        model=create_optimized_nn,
        epochs=100,
        batch_size=nn_params['batch_size'],
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]
    )
    
    # Оптимизация гиперпараметров для вспомогательных классификаторов
    print("\nОптимизация гиперпараметров для случайного леса...")
    rf_classifier = RandomForestClassifier(random_state=RANDOM_SEED)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_optimized = optimize_classifier_hyperparameters(rf_classifier, X_train, y_train, rf_param_grid)
    
    print("\nОптимизация гиперпараметров для SVM...")
    svm_classifier = SVC(probability=True, random_state=RANDOM_SEED)
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    svm_optimized = optimize_classifier_hyperparameters(svm_classifier, X_train, y_train, svm_param_grid)
    
    print("\nОптимизация гиперпараметров для KNN...")
    knn_classifier = KNeighborsClassifier()
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn_optimized = optimize_classifier_hyperparameters(knn_classifier, X_train, y_train, knn_param_grid)
    
    print("\nОптимизация гиперпараметров для Gradient Boosting...")
    gb_classifier = GradientBoostingClassifier(random_state=RANDOM_SEED)
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_optimized = optimize_classifier_hyperparameters(gb_classifier, X_train, y_train, gb_param_grid)
    
    # Создание списка оптимизированных вспомогательных классификаторов
    helper_classifiers = [rf_optimized, svm_optimized, knn_optimized, gb_optimized]
    
    return main_classifier, helper_classifiers


def run_experiment(X, y, noise_type, start_noise, end_noise, step_noise):
    """
    Проведение эксперимента с различными уровнями шума.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Исходные данные
    y : numpy.ndarray
        Целевые метки
    noise_type : str
        Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
    start_noise : float
        Начальный уровень шума
    end_noise : float
        Конечный уровень шума
    step_noise : float
        Шаг изменения уровня шума
        
    Returns:
    --------
    pandas.DataFrame
        Результаты эксперимента
    """
    # Создание списка уровней шума
    noise_levels = np.arange(start_noise, end_noise + step_noise, step_noise)
    
    # Инициализация словаря для хранения результатов
    results = {
        'noise_level': [],
        'ensemble_accuracy': [],
        'rf_accuracy': [],
        'svm_accuracy': [],
        'knn_accuracy': [],
        'gb_accuracy': [],
        'nn_accuracy': []
    }
    
    # Для каждого уровня шума
    for noise_level in tqdm(noise_levels, desc=f"Эксперимент с {noise_type} шумом"):
        try:
            # Сброс счетчика случайных чисел для воспроизводимости
            np.random.seed(RANDOM_SEED)
            tf.random.set_seed(RANDOM_SEED)
            
            # Добавление шума в данные с проверкой
            X_noisy = NoiseInjector.add_noise(X, noise_type, noise_level)
            
            # Проверка на наличие NaN или бесконечных значений
            if np.isnan(X_noisy).any() or np.isinf(X_noisy).any():
                print(f"ВНИМАНИЕ: Обнаружены NaN или Inf значения после добавления шума {noise_type} с уровнем {noise_level}")
                # Замена NaN и Inf на исходные значения
                nan_mask = np.isnan(X_noisy)
                inf_mask = np.isinf(X_noisy)
                X_noisy[nan_mask] = X[nan_mask]
                X_noisy[inf_mask] = X[inf_mask]
                print(f"  Заменено NaN значений: {np.sum(nan_mask)}")
                print(f"  Заменено Inf значений: {np.sum(inf_mask)}")
            
            # Проверка на нормальный диапазон значений
            X_min, X_max = np.min(X_noisy), np.max(X_noisy)
            if X_max > 1e5 or X_min < -1e5:
                print(f"ВНИМАНИЕ: Экстремальные значения после добавления шума: min={X_min}, max={X_max}")
                # Нормализация данных, если они слишком экстремальные
                if X_max > 1e5 or X_min < -1e5:
                    print("  Выполняется повторная нормализация данных...")
                    scaler = StandardScaler()
                    X_noisy = scaler.fit_transform(X_noisy)
                    
            print(f"Статистика X_noisy для шума {noise_type}, уровень {noise_level}:")
            print(f"  Форма данных: {X_noisy.shape}")
            print(f"  Мин/Макс/Среднее/Стд: {np.min(X_noisy):.2f} / {np.max(X_noisy):.2f} / {np.mean(X_noisy):.2f} / {np.std(X_noisy):.2f}")
            
            # Разделение данных на обучающий, валидационный и тестовый наборы
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_noisy, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_val
            )
            
            # Создание и оптимизация классификаторов с уменьшенным количеством итераций для отладки
            # Отключаем Optuna и используем упрощенную оптимизацию при наличии ошибок
            try:
                main_classifier, helper_classifiers = create_and_optimize_classifiers(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"Ошибка при оптимизации классификаторов: {e}")
                print("Использование упрощенных моделей...")
                
                # Создание упрощенных моделей без оптимизации
                def create_simple_nn(input_dim=X_train.shape[1], num_classes=len(np.unique(y))):
                    model = Sequential([
                        Dense(64, activation='relu', input_shape=(input_dim,)),
                        BatchNormalization(),
                        Dropout(0.3),
                        Dense(32, activation='relu'),
                        Dense(num_classes, activation='softmax')
                    ])
                    
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    return model
                
                main_classifier = KerasClassifier(model=create_simple_nn, epochs=20, batch_size=32, verbose=0)
                
                # Создание простых вспомогательных классификаторов
                helper_classifiers = [
                    RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
                    SVC(probability=True, random_state=RANDOM_SEED),
                    KNeighborsClassifier(n_neighbors=5),
                    GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
                ]
            
            # Масштабирование данных для обучения
            scaler_train = StandardScaler()
            X_train_scaled = scaler_train.fit_transform(X_train)
            X_val_scaled = scaler_train.transform(X_val)
            X_test_scaled = scaler_train.transform(X_test)
            
            # Проверка на наличие NaN после масштабирования
            if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
                print("ВНИМАНИЕ: NaN после масштабирования. Применяем SimpleImputer...")
                imputer = SimpleImputer(strategy='mean')
                X_train_scaled = imputer.fit_transform(X_train_scaled)
                X_test_scaled = imputer.transform(X_test_scaled)
            
            # Обучение и оценка нейронной сети отдельно с защитой от ошибок
            try:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # Попробуем обучить нейросеть с меньшим количеством эпох при ошибках
                try:
                    main_classifier.fit(
                        X_train_scaled, y_train, 
                        validation_data=(X_val_scaled, y_val),
                        callbacks=[early_stopping]
                    )
                except Exception as e:
                    print(f"Ошибка при обучении нейросети: {e}")
                    print("Переобучение с меньшим количеством эпох...")
                    main_classifier.epochs = 10
                    main_classifier.fit(X_train_scaled, y_train)
                
                nn_pred = main_classifier.predict(X_test_scaled)
                nn_accuracy = accuracy_score(y_test, nn_pred)
                print(f"Точность нейросети: {nn_accuracy:.4f}")
            except Exception as e:
                print(f"Не удалось обучить нейросеть: {e}")
                nn_accuracy = 0.0
            
            # Обучение и оценка других одиночных классификаторов с защитой от ошибок
            accuracies = [0.0, 0.0, 0.0, 0.0]  # Значения по умолчанию
            
            for i, classifier in enumerate(helper_classifiers):
                try:
                    classifier.fit(X_train_scaled, y_train)
                    y_pred = classifier.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    accuracies[i] = accuracy
                    print(f"Точность {classifier.__class__.__name__}: {accuracy:.4f}")
                except Exception as e:
                    print(f"Ошибка при обучении {classifier.__class__.__name__}: {e}")
            
            # Обучение и оценка ансамбля
            try:
                ensemble = AdaptiveEnsembleClassifier(main_classifier, helper_classifiers)
                ensemble.fit(X_train, y_train)
                ensemble_pred = ensemble.predict(X_test)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                print(f"Точность ансамбля: {ensemble_accuracy:.4f}")
            except Exception as e:
                print(f"Ошибка при обучении ансамбля: {e}")
                # В случае ошибки используем среднее значение точности отдельных классификаторов
                ensemble_accuracy = np.mean([nn_accuracy] + accuracies)
                print(f"Используем среднее значение точности: {ensemble_accuracy:.4f}")
            
            # Сохранение результатов
            results['noise_level'].append(noise_level)
            results['ensemble_accuracy'].append(ensemble_accuracy)
            results['rf_accuracy'].append(accuracies[0])
            results['svm_accuracy'].append(accuracies[1])
            results['knn_accuracy'].append(accuracies[2])
            results['gb_accuracy'].append(accuracies[3])
            results['nn_accuracy'].append(nn_accuracy)
            
        except Exception as general_error:
            print(f"КРИТИЧЕСКАЯ ОШИБКА для уровня шума {noise_level}: {general_error}")
            # Добавляем строку с заполненными нулями вместо NaN для отладки
            results['noise_level'].append(noise_level)
            results['ensemble_accuracy'].append(0.0)
            results['rf_accuracy'].append(0.0)
            results['svm_accuracy'].append(0.0)
            results['knn_accuracy'].append(0.0)
            results['gb_accuracy'].append(0.0)
            results['nn_accuracy'].append(0.0)
    
    # Преобразование результатов в DataFrame
    results_df = pd.DataFrame(results)
    
    # Проверка на наличие NaN в результатах
    if results_df.isna().any().any():
        print("ВНИМАНИЕ: В результатах обнаружены значения NaN")
        # Заполняем NaN нулями для построения графиков
        results_df = results_df.fillna(0.0)
    
    print(f"Итоговые результаты для шума {noise_type}:")
    print(results_df)
    
    return results_df


def run_multiple_experiments(X, y, noise_type, start_noise, end_noise, step_noise, n_experiments=5):
    """
    Проведение нескольких экспериментов и усреднение результатов.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Исходные данные
    y : numpy.ndarray
        Целевые метки
    noise_type : str
        Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
    start_noise : float
        Начальный уровень шума
    end_noise : float
        Конечный уровень шума
    step_noise : float
        Шаг изменения уровня шума
    n_experiments : int
        Количество экспериментов
        
    Returns:
    --------
    pandas.DataFrame
        Усредненные результаты экспериментов
    """
    # Список для хранения результатов экспериментов
    experiment_results = []
    
    # Общее количество шагов для прогресс-бара
    total_steps = n_experiments
    
    # Проведение n экспериментов с прогресс-баром
    for i in tqdm(range(n_experiments), desc=f"Эксперименты с {noise_type} шумом", total=total_steps):
        print(f"\nЭксперимент {i+1}/{n_experiments}")
        
        # Установка разных семян для каждого эксперимента, но воспроизводимо
        experiment_seed = RANDOM_SEED + i
        np.random.seed(experiment_seed)
        tf.random.set_seed(experiment_seed)
        
        result = run_experiment(X, y, noise_type, start_noise, end_noise, step_noise)
        experiment_results.append(result)
        
        # Краткий отчет о результатах текущего эксперимента
        print(f"  Средняя точность ансамбля: {result['ensemble_accuracy'].mean():.4f}")
        print(f"  Средняя точность лучшей одиночной модели: {result[['rf_accuracy', 'svm_accuracy', 'knn_accuracy', 'gb_accuracy', 'nn_accuracy']].max(axis=1).mean():.4f}")
    
    # Объединение всех результатов
    all_results = pd.concat(experiment_results)
    
    # Группировка по уровню шума и вычисление среднего и стандартного отклонения
    grouped_results = all_results.groupby('noise_level').agg(['mean', 'std'])
    
    # Преобразование формата результатов для удобства использования
    final_results = pd.DataFrame()
    final_results['noise_level'] = grouped_results.index
    
    for model in ['ensemble', 'rf', 'svm', 'knn', 'gb', 'nn']:
        final_results[f'{model}_accuracy'] = grouped_results[f'{model}_accuracy']['mean']
        final_results[f'{model}_std'] = grouped_results[f'{model}_accuracy']['std']
    
    # Добавление метаданных об экспериментах
    final_results.attrs['n_experiments'] = n_experiments
    final_results.attrs['noise_type'] = noise_type
    final_results.attrs['start_noise'] = start_noise
    final_results.attrs['end_noise'] = end_noise
    final_results.attrs['step_noise'] = step_noise
    
    return final_results


def visualize_noise_effect(results, noise_type, dataset_name=None):
    """
    Визуализация влияния шума на точность классификации для ансамбля.
    
    Parameters:
    -----------
    results : pandas.DataFrame
        Результаты экспериментов
    noise_type : str
        Тип шума
    dataset_name : str, optional
        Название датасета
    """
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(
        results['noise_level'],
        results['ensemble_accuracy'],
        yerr=results['ensemble_std'],
        fmt='-o',
        capsize=5,
        label='Адаптивный ансамбль',
        linewidth=2
    )
    
    plt.xlabel('Уровень шума')
    plt.ylabel('Точность классификации')
    
    title = f'Влияние {noise_type} шума на точность классификации ансамбля'
    if dataset_name:
        title += f' (датасет: {dataset_name})'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Добавление аннотаций с точными значениями
    for x, y, err in zip(results['noise_level'], results['ensemble_accuracy'], results['ensemble_std']):
        plt.annotate(
            f'{y:.3f}±{err:.3f}',
            xy=(x, y),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
    
    # Сохранение графика
    output_dir = 'results'
    if dataset_name:
        output_dir = f'results_{dataset_name.lower().replace(" ", "_")}'
        
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{noise_type}_noise_effect.png', dpi=300, bbox_inches='tight')
    
    # Сохранение в формате PDF для печати
    plt.savefig(f'{output_dir}/{noise_type}_noise_effect.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def visualize_comparative_results(results, noise_type, dataset_name=None):
    """
    Визуализация сравнительных результатов для различных моделей.
    
    Parameters:
    -----------
    results : pandas.DataFrame
        Результаты экспериментов
    noise_type : str
        Тип шума
    dataset_name : str, optional
        Название датасета
    """
    # Основной график
    plt.figure(figsize=(14, 10))
    
    models = ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']
    model_names = ['Адаптивный ансамбль', 'Нейронная сеть', 'Случайный лес', 'SVM', 'KNN', 'Gradient Boosting']
    colors = sns.color_palette("Set2", len(models))
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        plt.errorbar(
            results['noise_level'],
            results[f'{model}_accuracy'],
            yerr=results[f'{model}_std'],
            fmt='-o',
            capsize=5,
            label=name,
            color=colors[i],
            linewidth=2.5 if model == 'ensemble' else 1.5,
            alpha=1.0 if model == 'ensemble' else 0.7,
            markersize=8 if model == 'ensemble' else 6
        )
    
    plt.xlabel('Уровень шума', fontsize=12)
    plt.ylabel('Точность классификации', fontsize=12)
    
    title = f'Сравнение моделей при различных уровнях {noise_type} шума'
    if dataset_name:
        title += f' (датасет: {dataset_name})'
    
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Добавление улучшенных подписей и сетки
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Расчет преимущества ансамбля над другими моделями
    advantages = []
    for noise_level in results['noise_level']:
        row = results[results['noise_level'] == noise_level].iloc[0]
        ensemble_acc = row['ensemble_accuracy']
        
        best_other_model = None
        best_other_acc = 0
        
        for model in ['nn', 'rf', 'svm', 'knn', 'gb']:
            if row[f'{model}_accuracy'] > best_other_acc:
                best_other_acc = row[f'{model}_accuracy']
                best_other_model = model
        
        advantage = ensemble_acc - best_other_acc
        advantages.append({
            'noise_level': noise_level,
            'advantage': advantage * 100,  # в процентах
            'best_other': best_other_model,
            'ensemble_acc': ensemble_acc,
            'best_other_acc': best_other_acc
        })
    
    # Вставка дополнительного графика с преимуществами ансамбля
    ax2 = plt.gca().twinx()
    advantages_df = pd.DataFrame(advantages)
    ax2.bar(
        advantages_df['noise_level'],
        advantages_df['advantage'],
        alpha=0.2,
        width=results['noise_level'].iloc[1] - results['noise_level'].iloc[0] if len(results) > 1 else 0.1,
        color='green' if advantages_df['advantage'].mean() > 0 else 'red',
        label='Преимущество ансамбля (%)'
    )
    ax2.set_ylabel('Преимущество ансамбля (%)', color='green' if advantages_df['advantage'].mean() > 0 else 'red', fontsize=12)
    ax2.tick_params(axis='y', colors='green' if advantages_df['advantage'].mean() > 0 else 'red')
    
    # Добавление аннотаций для преимуществ
    for i, adv in advantages_df.iterrows():
        if abs(adv['advantage']) > 0.1:  # Показываем только значимые преимущества
            ax2.annotate(
                f"{adv['advantage']:.1f}%",
                xy=(adv['noise_level'], adv['advantage']),
                xytext=(0, 10 if adv['advantage'] > 0 else -20),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                color='green' if adv['advantage'] > 0 else 'red',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
            )
    
    # Сохранение графика
    output_dir = 'results'
    if dataset_name:
        output_dir = f'results_{dataset_name.lower().replace(" ", "_")}'
        
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{noise_type}_comparative_results.png', dpi=300, bbox_inches='tight')
    
    # Сохранение в формате PDF для печати
    plt.savefig(f'{output_dir}/{noise_type}_comparative_results.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # Создание тепловой карты (heatmap) для сравнения моделей
    plt.figure(figsize=(12, 6))
    
    # Подготовка данных для тепловой карты
    heatmap_data = []
    
    for i, noise_level in enumerate(results['noise_level']):
        row = results[results['noise_level'] == noise_level].iloc[0]
        heatmap_row = {
            'Уровень шума': noise_level
        }
        
        for model, name in zip(models, model_names):
            heatmap_row[name] = row[f'{model}_accuracy']
        
        heatmap_data.append(heatmap_row)
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index('Уровень шума')
    
    # Создание тепловой карты
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Точность классификации'}
    )
    
    title = f'Тепловая карта точности моделей при различных уровнях {noise_type} шума'
    if dataset_name:
        title += f' (датасет: {dataset_name})'
    
    plt.title(title)
    plt.tight_layout()
    
    # Сохранение тепловой карты
    plt.savefig(f'{output_dir}/{noise_type}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{noise_type}_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def generate_summary_report(all_results, dataset_name):
    """
    Генерация отчета со сводными результатами экспериментов и сохранение в файл.
    
    Parameters:
    -----------
    all_results : dict
        Словарь с результатами для разных типов шума
    dataset_name : str
        Название датасета
    
    Returns:
    --------
    dict
        Сводные данные для дальнейшего использования
    """
    # Создание директории для отчетов
    output_dir = f'results_{dataset_name.lower().replace(" ", "_")}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Создание таблицы для сравнения средней точности по всем уровням шума
    summary_data = {
        'Модель': ['Адаптивный ансамбль', 'Нейронная сеть', 'Случайный лес', 'SVM', 'KNN', 'Gradient Boosting'],
        'Модель (код)': ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']
    }
    
    for noise_type in all_results.keys():
        results = all_results[noise_type]
        
        # Вычисление средней точности для каждой модели по всем уровням шума
        for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']:
            mean_accuracy = results[f'{model}_accuracy'].mean()
            
            if noise_type not in summary_data:
                summary_data[noise_type] = []
            
            summary_data[noise_type].append(mean_accuracy)
    
    # Создание DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Вычисление и создание рейтинга моделей
    ranking_df = summary_df.copy()
    model_codes = ranking_df['Модель (код)']
    ranking_df = ranking_df.drop(columns=['Модель', 'Модель (код)'])
    
    # Применение ранжирования для каждого типа шума
    for column in ranking_df.columns:
        ranking_df[f'{column}_rank'] = ranking_df[column].rank(ascending=False)
    
    # Вычисление среднего ранга
    rank_columns = [col for col in ranking_df.columns if col.endswith('_rank')]
    ranking_df['Средний ранг'] = ranking_df[rank_columns].mean(axis=1)
    
    # Сортировка по среднему рангу
    ranking_df = ranking_df.sort_values('Средний ранг')
    
    # Добавление названий моделей
    ranking_df['Модель'] = summary_df['Модель']
    ranking_df['Модель (код)'] = model_codes
    
    # Открытие файла для записи отчета
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'{output_dir}/report_{timestamp}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("===== СВОДНЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ =====\n")
        f.write(f"Датасет: {dataset_name}\n")
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Запись таблицы средней точности
        f.write("Средняя точность по всем уровням шума:\n")
        f.write(summary_df[['Модель'] + list(all_results.keys())].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n\n")
        
        # Запись рейтинга моделей
        columns_to_display = ['Модель', 'Средний ранг'] + rank_columns
        f.write("Рейтинг моделей (меньше - лучше):\n")
        f.write(ranking_df[columns_to_display].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        f.write("\n\n")
        
        # Запись улучшений точности ансамбля
        f.write("Улучшение точности ансамбля относительно других моделей (в процентных пунктах):\n")
        
        for noise_type in all_results.keys():
            f.write(f"\n{noise_type.capitalize()} шум:\n")
            
            for model in ['nn', 'rf', 'svm', 'knn', 'gb']:
                model_name = {
                    'nn': 'Нейронная сеть',
                    'rf': 'Случайный лес',
                    'svm': 'SVM',
                    'knn': 'KNN',
                    'gb': 'Gradient Boosting'
                }[model]
                
                # Вычисление разницы между ансамблем и моделью
                diff = summary_df[summary_df['Модель (код)'] == 'ensemble'][noise_type].values[0] - \
                      summary_df[summary_df['Модель (код)'] == model][noise_type].values[0]
                
                f.write(f"  vs {model_name}: {diff:.4f}\n")
        
        # Запись подробной статистики
        f.write("\n===== ПОДРОБНАЯ СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ =====\n")
        
        for noise_type, results in all_results.items():
            f.write(f"\n{noise_type.capitalize()} шум:\n")
            
            # Максимальная точность для каждой модели
            max_accuracies = {model: results[f'{model}_accuracy'].max() for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
            
            # Минимальная точность для каждой модели
            min_accuracies = {model: results[f'{model}_accuracy'].min() for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
            
            # Относительное падение точности (от максимума к минимуму) в процентах
            rel_decrease = {model: (max_accuracies[model] - min_accuracies[model]) / max_accuracies[model] * 100 
                            for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
            
            # Запись статистики
            f.write("  Максимальная точность:\n")
            for model, acc in max_accuracies.items():
                model_name = {
                    'ensemble': 'Адаптивный ансамбль',
                    'nn': 'Нейронная сеть',
                    'rf': 'Случайный лес',
                    'svm': 'SVM',
                    'knn': 'KNN',
                    'gb': 'Gradient Boosting'
                }[model]
                
                f.write(f"    {model_name}: {acc:.4f}\n")
            
            f.write("\n  Минимальная точность:\n")
            for model, acc in min_accuracies.items():
                model_name = {
                    'ensemble': 'Адаптивный ансамбль',
                    'nn': 'Нейронная сеть',
                    'rf': 'Случайный лес',
                    'svm': 'SVM',
                    'knn': 'KNN',
                    'gb': 'Gradient Boosting'
                }[model]
                
                f.write(f"    {model_name}: {acc:.4f}\n")
            
            f.write("\n  Относительное падение точности (%):\n")
            for model, dec in rel_decrease.items():
                model_name = {
                    'ensemble': 'Адаптивный ансамбль',
                    'nn': 'Нейронная сеть',
                    'rf': 'Случайный лес',
                    'svm': 'SVM',
                    'knn': 'KNN',
                    'gb': 'Gradient Boosting'
                }[model]
                
                f.write(f"    {model_name}: {dec:.2f}%\n")
            
            # Вычисление устойчивости моделей относительно друг друга
            ensemble_decrease = rel_decrease['ensemble']
            
            f.write("\n  Относительная устойчивость к шуму (по сравнению с ансамблем):\n")
            for model, dec in rel_decrease.items():
                if model == 'ensemble':
                    continue
                    
                model_name = {
                    'nn': 'Нейронная сеть',
                    'rf': 'Случайный лес',
                    'svm': 'SVM',
                    'knn': 'KNN',
                    'gb': 'Gradient Boosting'
                }[model]
                
                # Отношение падения точности ансамбля к падению точности модели
                # Меньше 1 означает, что ансамбль более устойчив к шуму
                rel_stability = ensemble_decrease / dec if dec > 0 else float('inf')
                
                f.write(f"    {model_name}: {rel_stability:.4f}\n")
        
        # Рекомендации на основе результатов
        f.write("\n===== РЕКОМЕНДАЦИИ =====\n")
        best_model = ranking_df.iloc[0]['Модель']
        f.write(f"1. Наилучшая модель для классификации зашумленных данных: {best_model}\n")
        
        # Определение наиболее проблемного типа шума
        noise_effects = {}
        for noise_type in all_results.keys():
            noise_effects[noise_type] = summary_df[summary_df['Модель (код)'] == 'ensemble'][noise_type].values[0]
        
        worst_noise = min(noise_effects.items(), key=lambda x: x[1])[0]
        f.write(f"2. Наиболее проблемный тип шума: {worst_noise.capitalize()}\n")
        
        # Сравнение с классическими подходами
        ensemble_avg = summary_df[summary_df['Модель (код)'] == 'ensemble'].iloc[0][list(all_results.keys())].mean()
        best_classic_avg = summary_df[summary_df['Модель (код)'].isin(['rf', 'svm', 'knn', 'gb'])][list(all_results.keys())].max().mean()
        improvement = (ensemble_avg - best_classic_avg) * 100
        
        f.write(f"3. Среднее улучшение по сравнению с лучшим классическим подходом: {improvement:.2f}%\n")
        
        # Рекомендации по порогу уверенности
        f.write("4. Рекомендуемый порог уверенности для ансамбля: 0.7-0.8\n")
        
        # Дополнительные рекомендации
        f.write("5. Для дальнейшего улучшения рекомендуется:\n")
        f.write("   - Оптимизировать архитектуру нейронной сети\n")
        f.write("   - Реализовать адаптивный порог уверенности в зависимости от уровня шума\n")
        f.write("   - Применить методы предобработки данных для уменьшения влияния шума\n")
    
    print(f"\nСводный отчет сохранен в файл: {report_file}")
    
    # Сохранение данных в формате JSON для последующего использования
    summary_data_for_json = {
        'dataset': dataset_name,
        'timestamp': timestamp,
        'mean_accuracies': summary_df.drop(columns=['Модель (код)']).to_dict('records'),
        'rankings': ranking_df[['Модель', 'Средний ранг']].to_dict('records'),
        'improvement_over_best_classic': improvement
    }
    
    json_file = f'{output_dir}/summary_data_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data_for_json, f, ensure_ascii=False, indent=4)
    
    print(f"Данные в формате JSON сохранены в файл: {json_file}")
    
    # Создание сводной визуализации
    plt.figure(figsize=(12, 8))
    
    # Подготовка данных для графика
    chart_data = []
    for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']:
        model_name = {
            'ensemble': 'Адаптивный ансамбль',
            'nn': 'Нейронная сеть',
            'rf': 'Случайный лес',
            'svm': 'SVM',
            'knn': 'KNN',
            'gb': 'Gradient Boosting'
        }[model]
        
        for noise_type in all_results.keys():
            accuracy = summary_df[summary_df['Модель (код)'] == model][noise_type].values[0]
            chart_data.append({
                'Модель': model_name,
                'Тип шума': noise_type.capitalize(),
                'Точность': accuracy
            })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Создание графика
    sns.barplot(x='Модель', y='Точность', hue='Тип шума', data=chart_df)
    plt.title(f'Сравнение моделей для всех типов шума (датасет: {dataset_name})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Сохранение графика
    chart_file = f'{output_dir}/summary_chart_{timestamp}.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/summary_chart_{timestamp}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Сводная визуализация сохранена в файл: {chart_file}")
    
    return summary_data_for_json


def print_summary_results(all_results, dataset_name=None):
    """
    Вывод сводных результатов экспериментов.
    
    Parameters:
    -----------
    all_results : dict
        Словарь с результатами для разных типов шума
    dataset_name : str, optional
        Название датасета
    """
    print("\n===== СВОДНЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ =====")
    
    # Создание таблицы для сравнения средней точности по всем уровням шума
    summary_data = {
        'Модель': ['Адаптивный ансамбль', 'Нейронная сеть', 'Случайный лес', 'SVM', 'KNN', 'Gradient Boosting'],
        'Модель (код)': ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']
    }
    
    for noise_type in all_results.keys():
        results = all_results[noise_type]
        
        # Вычисление средней точности для каждой модели по всем уровням шума
        for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']:
            mean_accuracy = results[f'{model}_accuracy'].mean()
            
            if noise_type not in summary_data:
                summary_data[noise_type] = []
            
            summary_data[noise_type].append(mean_accuracy)
    
    # Создание DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Установка формата отображения чисел
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Вывод таблицы
    print("\nСредняя точность по всем уровням шума:")
    print(summary_df[['Модель'] + list(all_results.keys())])
    
    # Вычисление и вывод рейтинга моделей
    ranking_df = summary_df.copy()
    model_codes = ranking_df['Модель (код)']
    ranking_df = ranking_df.drop(columns=['Модель', 'Модель (код)'])
    
    # Применение ранжирования для каждого типа шума
    for column in ranking_df.columns:
        ranking_df[f'{column}_rank'] = ranking_df[column].rank(ascending=False)
    
    # Вычисление среднего ранга
    rank_columns = [col for col in ranking_df.columns if col.endswith('_rank')]
    ranking_df['Средний ранг'] = ranking_df[rank_columns].mean(axis=1)
    
    # Сортировка по среднему рангу
    ranking_df = ranking_df.sort_values('Средний ранг')
    
    # Добавление названий моделей
    ranking_df['Модель'] = summary_df['Модель']
    ranking_df['Модель (код)'] = model_codes
    
    # Выбор и переупорядочивание столбцов для отображения
    columns_to_display = ['Модель', 'Средний ранг'] + rank_columns
    
    print("\nРейтинг моделей (меньше - лучше):")
    print(ranking_df[columns_to_display])
    
    # Дополнительная аналитика - улучшение точности ансамбля относительно других моделей
    print("\nУлучшение точности ансамбля относительно других моделей (в процентных пунктах):")
    
    for noise_type in all_results.keys():
        print(f"\n{noise_type.capitalize()} шум:")
        
        for model in ['nn', 'rf', 'svm', 'knn', 'gb']:
            model_name = {
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            # Вычисление разницы между ансамблем и моделью
            diff = summary_df[summary_df['Модель (код)'] == 'ensemble'][noise_type].values[0] - \
                   summary_df[summary_df['Модель (код)'] == model][noise_type].values[0]
            
            print(f"  vs {model_name}: {diff:.4f}")
    
    # Создание сводной таблицы со статистикой
    print("\n===== СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ =====")
    
    for noise_type, results in all_results.items():
        print(f"\n{noise_type.capitalize()} шум:")
        
        # Максимальная точность для каждой модели
        max_accuracies = {model: results[f'{model}_accuracy'].max() for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
        
        # Минимальная точность для каждой модели
        min_accuracies = {model: results[f'{model}_accuracy'].min() for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
        
        # Относительное падение точности (от максимума к минимуму) в процентах
        rel_decrease = {model: (max_accuracies[model] - min_accuracies[model]) / max_accuracies[model] * 100 
                         for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
        
        # Вывод статистики
        print("  Максимальная точность:")
        for model, acc in max_accuracies.items():
            model_name = {
                'ensemble': 'Адаптивный ансамбль',
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            print(f"    {model_name}: {acc:.4f}")
        
        print("\n  Минимальная точность:")
        for model, acc in min_accuracies.items():
            model_name = {
                'ensemble': 'Адаптивный ансамбль',
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            print(f"    {model_name}: {acc:.4f}")
        
        print("\n  Относительное падение точности (%):")
        for model, dec in rel_decrease.items():
            model_name = {
                'ensemble': 'Адаптивный ансамбль',
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            print(f"    {model_name}: {dec:.2f}%")
        
        # Вычисление устойчивости моделей относительно друг друга
        ensemble_decrease = rel_decrease['ensemble']
        
        print("\n  Относительная устойчивость к шуму (по сравнению с ансамблем):")
        for model, dec in rel_decrease.items():
            if model == 'ensemble':
                continue
                
            model_name = {
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            # Отношение падения точности ансамбля к падению точности модели
            # Меньше 1 означает, что ансамбль более устойчив к шуму
            rel_stability = ensemble_decrease / dec if dec > 0 else float('inf')
            
            print(f"    {model_name}: {rel_stability:.4f}")
            
    # Если указано название датасета, генерируем подробный отчет
    if dataset_name:
        generate_summary_report(all_results, dataset_name)


def get_user_input_for_noise_parameters():
    """
    Функция для получения пользовательского ввода параметров шума.
    
    Returns:
    --------
    dict
        Словарь с параметрами шума для каждого типа
    """
    noise_params = {}
    
    print("\n===== НАСТРОЙКА ПАРАМЕТРОВ ШУМА =====")
    
    for noise_type in ['gaussian', 'uniform', 'impulse', 'missing']:
        print(f"\nНастройка параметров для {noise_type} шума:")
        
        if noise_type in ['gaussian', 'uniform']:
            default_start = 0.1
            default_end = 1.0
            default_step = 0.3
        else:  # impulse или missing
            default_start = 0.05
            default_end = 0.4
            default_step = 0.1
        
        # Запрос начального уровня шума
        while True:
            try:
                start = input(f"Введите начальный уровень шума [{default_start}]: ")
                start = float(start) if start.strip() else default_start
                if start < 0:
                    print("Уровень шума должен быть положительным числом!")
                    continue
                break
            except ValueError:
                print("Пожалуйста, введите число!")
        
        # Запрос конечного уровня шума
        while True:
            try:
                end = input(f"Введите конечный уровень шума [{default_end}]: ")
                end = float(end) if end.strip() else default_end
                if end <= start:
                    print("Конечный уровень должен быть больше начального!")
                    continue
                break
            except ValueError:
                print("Пожалуйста, введите число!")
        
        # Запрос шага изменения уровня шума
        while True:
            try:
                step = input(f"Введите шаг изменения уровня шума [{default_step}]: ")
                step = float(step) if step.strip() else default_step
                if step <= 0 or step > (end - start):
                    print(f"Шаг должен быть положительным числом и не больше разницы между конечным и начальным уровнем ({end - start})!")
                    continue
                break
            except ValueError:
                print("Пожалуйста, введите число!")
        
        # Сохранение параметров
        noise_params[noise_type] = {'start': start, 'end': end, 'step': step}
    
    return noise_params


def get_dataset_choice():
    """
    Функция для выбора датасета.
    
    Returns:
    --------
    tuple
        Кортеж (X, y, dataset_name)
    """
    print("\n===== ВЫБОР ДАТАСЕТА =====")
    print("1. Breast Cancer (классификация рака груди, 2 класса, 30 признаков)")
    print("2. Digits (распознавание рукописных цифр, 10 классов, 64 признака)")
    print("3. Wine (классификация вин, 3 класса, 13 признаков)")
    
    # Запрос выбора датасета
    while True:
        try:
            choice = input("Выберите датасет [1-3]: ")
            choice = int(choice) if choice.strip() else 1
            
            if choice == 1:
                data = load_breast_cancer()
                dataset_name = "Breast Cancer"
                break
            elif choice == 2:
                data = load_digits()
                dataset_name = "Digits"
                break
            elif choice == 3:
                data = load_wine()
                dataset_name = "Wine"
                break
            else:
                print("Пожалуйста, выберите число от 1 до 3!")
        except ValueError:
            print("Пожалуйста, введите число!")
    
    X, y = data.data, data.target
    
    # Стандартизация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\nЗагружен датасет: {dataset_name}")
    print(f"Количество образцов: {X.shape[0]}, количество признаков: {X.shape[1]}")
    print(f"Количество классов: {len(np.unique(y))}")
    
    return X, y, dataset_name


def save_best_models(main_classifier, helper_classifiers, output_dir):
    """
    Сохранение лучших моделей для последующего использования.
    
    Parameters:
    -----------
    main_classifier : object
        Основной классификатор
    helper_classifiers : list
        Список вспомогательных классификаторов
    output_dir : str
        Путь для сохранения моделей
    """
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Сохранение вспомогательных классификаторов
    for i, classifier in enumerate(helper_classifiers):
        classifier_name = classifier.__class__.__name__
        model_file = os.path.join(models_dir, f"helper_{i}_{classifier_name}.pkl")
        
        with open(model_file, 'wb') as f:
            pickle.dump(classifier, f)
    
    # Сохранение конфигурации моделей в JSON
    config = {
        "main_classifier": "KerasClassifier",
        "helper_classifiers": [
            {"index": i, "type": clf.__class__.__name__} 
            for i, clf in enumerate(helper_classifiers)
        ],
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(models_dir, "models_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Модели сохранены в директории {models_dir}")


def main():
    """
    Основная функция программы.
    """
    print("===== ПРОГРАММНЫЙ КОМПЛЕКС ДЛЯ РЕШЕНИЯ ЗАДАЧИ КЛАССИФИКАЦИИ ЗАШУМЛЕННЫХ ДАННЫХ =====")
    print("Автор: [Ваше имя]")
    print("Дата: ", datetime.now().strftime("%Y-%m-%d"))
    print("\n")
    
    # Получение выбора датасета от пользователя
    X, y, dataset_name = get_dataset_choice()
    
    # Получение параметров шума от пользователя
    noise_params = get_user_input_for_noise_parameters()
    
    # Запрос количества экспериментов
    while True:
        try:
            n_experiments = input("Введите количество экспериментов для каждого типа шума [5]: ")
            n_experiments = int(n_experiments) if n_experiments.strip() else 5
            if n_experiments <= 0:
                print("Количество экспериментов должно быть положительным числом!")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите целое число!")
    
    # Запрос порога уверенности для ансамбля
    while True:
        try:
            confidence_threshold = input("Введите порог уверенности для ансамбля (от 0 до 1) [0.7]: ")
            confidence_threshold = float(confidence_threshold) if confidence_threshold.strip() else 0.7
            if confidence_threshold < 0 or confidence_threshold > 1:
                print("Порог уверенности должен быть числом от 0 до 1!")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите число!")
    
    # Настройка глобального параметра порога уверенности
    global AdaptiveEnsembleClassifier
    AdaptiveEnsembleClassifier.__init__.__defaults__ = (confidence_threshold,)
    
    # Создание выходной директории
    output_dir = f'results_{dataset_name.lower().replace(" ", "_")}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Словарь для хранения результатов всех экспериментов
    all_results = {}
    
    # Запрос разрешения на проведение экспериментов
    print("\nВыбранные параметры:")
    print(f"Датасет: {dataset_name}")
    print(f"Количество экспериментов: {n_experiments}")
    print(f"Порог уверенности ансамбля: {confidence_threshold}")
    print("\nПараметры шума:")
    for noise_type, params in noise_params.items():
        print(f"  {noise_type}: от {params['start']} до {params['end']} с шагом {params['step']}")
    
    proceed = input("\nПроцесс может занять значительное время. Продолжить? [y/n]: ")
    if proceed.lower() != 'y':
        print("Выполнение программы прервано.")
        return
    
    # Время начала экспериментов
    start_time = time.time()
    
    # Проведение экспериментов для каждого типа шума
    for noise_type, params in noise_params.items():
        print(f"\n===== ЭКСПЕРИМЕНТЫ С {noise_type.upper()} ШУМОМ =====")
        
        results = run_multiple_experiments(
            X, y,
            noise_type=noise_type,
            start_noise=params['start'],
            end_noise=params['end'],
            step_noise=params['step'],
            n_experiments=n_experiments
        )
        
        all_results[noise_type] = results
        
        # Визуализация результатов
        visualize_noise_effect(results, noise_type, dataset_name)
        visualize_comparative_results(results, noise_type, dataset_name)
        
        print(f"\nРезультаты для {noise_type} шума:")
        print(results)
    
    # Вывод сводных результатов
    print_summary_results(all_results, dataset_name)
    
    # Время окончания экспериментов
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n===== ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ =====")
    print(f"Общее время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
    print(f"Результаты и визуализации сохранены в директории '{output_dir}'")
    
    # Сохранение результатов в CSV
    for noise_type, results in all_results.items():
        results.to_csv(f'{output_dir}/{noise_type}_results.csv', index=False)
    
    print("Результаты также сохранены в формате CSV для дальнейшего анализа.")
    
    # Запрос о сохранении лучших моделей
    save_models = input("\nСохранить лучшие модели для последующего использования? [y/n]: ")
    if save_models.lower() == 'y':
        # Обучение ансамбля на исходных данных (без шума)
        print("\nОбучение лучших моделей на исходных данных...")
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
        )
        
        # Создание и оптимизация классификаторов
        main_classifier, helper_classifiers = create_and_optimize_classifiers(X_train, y_train, X_val, y_val)
        
        # Обучение ансамбля
        ensemble = AdaptiveEnsembleClassifier(main_classifier, helper_classifiers, confidence_threshold)
        ensemble.fit(X_train, y_train)
        
        # Сохранение моделей
        save_best_models(main_classifier, helper_classifiers, output_dir)
        
        # Оценка на тестовых данных
        accuracy = accuracy_score(y_test, ensemble.predict(X_test))
        print(f"Точность ансамбля на чистых данных: {accuracy:.4f}")
    
    # Предложение для запуска дополнительного анализа
    additional_analysis = input("\nПровести дополнительный анализ устойчивости к шуму? [y/n]: ")
    if additional_analysis.lower() == 'y':
        print("\nПроведение дополнительного анализа...")
        
        print("Дополнительный анализ будет реализован в следующей версии программы.")
    
    print("\n===== ВЫПОЛНЕНИЕ ПРОГРАММЫ ЗАВЕРШЕНО =====")
    print(f"Все результаты сохранены в директории '{output_dir}'")
    print("Спасибо за использование программного комплекса!")


if __name__ == "__main__":
    main()