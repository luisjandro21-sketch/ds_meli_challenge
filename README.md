# MercadoLibre New vs Used Product Classification

This project implements a machine learning solution to classify MercadoLibre marketplace listings as either "new" or "used" products. The solution combines comprehensive exploratory data analysis, feature engineering, and optimized XGBoost modeling to achieve 89% accuracy.

## To Run 

It is recommended to create a virtual environment and install the libraries with which the original code was executed. Keep in mind that this was executed with Python version 3.11, there may be problems with other versions.

Run in terminal:

```bash
python -m venv {name_of_the_virtual_environment}
source {name_of_the_virtual_environment}/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt
```

### Training and Running the Model

**Using the modular pipeline (Recommended)**
```bash
python main.py --save-model ./output/model.pkl
```

**Option 2: Using the simple implementation (Not implemented)** 
```bash
python new_or_used.py 
```

Both approaches will generate a trained model. The modular pipeline provides better configuration management and extensibility.

### Using a Pre-trained Model

If you want to use a model fitted by the authors, run the following Python code:

```python
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

model = joblib.load("output/model.pkl")
```

## Project Structure

```
Ejercicio_nuevo/
├── notebooks/
│   └── new_or_used.ipynb                    # Comprehensive exploratory data analysis
├── src/
│   ├── config.py                    # Configuration and feature definitions
│   ├── pipeline.py                  # Main ML pipeline
│   ├── data/
│   │   ├── loader.py               # Data loading utilities
│   │   └── cleaner.py              # Data preprocessing
│   ├── features/
│   │   └── engineer.py             # Feature engineering
│   └── models/
│       ├── trainer.py              # Model training
│       └── evaluator.py            # Model evaluation
├── main.py                         # Main execution script
├── new_or_used.py                  # Alternative simple implementation
```

Enjoy!

---

**Authors**: Luis Alejandro Garcia
**Version**: 2.0 - Enhanced with comprehensive EDA and hyperparameter optimization  
**Last Updated**: May 2026