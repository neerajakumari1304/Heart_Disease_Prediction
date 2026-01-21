import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import sys
from log_code import setup_logging

logger = setup_logging('balance_data')

class BALANCING_DATA:
    def balance_data(X_train, y_train):
        try:
            # Flatten y_train if it's a DataFrame with one column
            y_train = y_train.values.ravel()

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            logger.info(f"Original training shape: {X_train.shape}, {y_train.shape}")
            logger.info(f"Balanced training shape: {X_res.shape}, {y_res.shape}")

            return X_res, y_res

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in line {error_line.tb_lineno}: {error_msg}")

