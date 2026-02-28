import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # هذه القيم مستخرجة مباشرة من ملف الموديل الخاص بك
        self.mapping_dict = {
            'gender': {'Male': 0, 'Female': 1},
            'customer_type': {'disloyal Customer': 0, 'Loyal Customer': 1},
            'type_of_travel': {'Personal Travel': 0, 'Business travel': 1},
            'class': {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
        }
        # القيمة المحفوظة للوسيط في الموديل الخاص بك
        self.arrival_median_ = 0.0 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. معالجة القيم المفقودة (Imputation)
        if 'arrival_delay_in_minutes' in X.columns:
            X['arrival_delay_in_minutes'] = X['arrival_delay_in_minutes'].fillna(self.arrival_median_)
        
        # 2. تحويل القيم النصية (Label Encoding)
        for col, mapping in self.mapping_dict.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        # 3. هندسة الميزات (Feature Engineering)
        # ملاحظة: الموديل يتوقع هذه الأسماء بالضبط (حسب فحص الملف)
        X['digital_Experience'] = (X['ease_of_online_booking'] + X['inflight_wifi_service']) / 2
        X['inflight_experience_score'] = X[['food_and_drink', 'cleanliness', 'seat_comfort']].mean(axis=1)
        X['total_Delay'] = X['departure_delay_in_minutes'] + X['arrival_delay_in_minutes']
        
        # 4. حذف الأعمدة التي تم استبعادها أثناء التدريب
        drop_cols = [
            'unnamed:_0', 'id', 'gate_location', 'ease_of_online_booking', 
            'inflight_wifi_service', 'food_and_drink', 'cleanliness', 
            'seat_comfort', 'departure_delay_in_minutes', 'arrival_delay_in_minutes', 'gender'
        ]
        
        X.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        return X