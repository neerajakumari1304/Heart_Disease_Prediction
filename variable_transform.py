import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class VT_OUT:
    def variable_transformation_outlier(X_train, X_test):
        try:
            # Make sure plot folder exists
            PLOT_PATH = "all_plots"
            os.makedirs(PLOT_PATH, exist_ok=True)

            # Columns to cap and log-transform
            cap_cols = ['chol', 'thalach', 'trestbps', 'oldpeak']
            log_cols = ['chol', 'oldpeak']

            # Outlier capping
            for col in cap_cols:
                Q1, Q3 = X_train[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                X_train[col] = X_train[col].clip(lower, upper)
                X_test[col] = X_test[col].clip(lower, upper)

            # Log transform
            for col in log_cols:
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])

            # Simple plots before/after
            for col in X_train.columns:
                plt.figure()
                sns.boxplot(x=X_train[col])
                plt.title(f'Boxplot-{col}')
                plt.savefig(f'{PLOT_PATH}/boxplot_{col}.png')
                plt.close()

            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error at line {error_line.tb_lineno}: {error_msg}')
