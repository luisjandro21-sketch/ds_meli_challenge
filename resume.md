## New or Used Classification Problem

This solution addresses challenge's of automatically classifying product listings as "new" or "used". The dataset contains 100,000 real marketplace listings from `MLA_100k.jsonlines`, with comprehensive feature engineering and model optimization.

### Comprehensive Data Analysis

The project includes extensive exploratory data analysis in `notebooks/EDA.ipynb`, covering:

- **Data Quality Assessment**: 36 engineered features across 100,000 records with minimal missing data (only warranty variable at 60.9% missing)
- **Feature Profiling**: Systematic analysis by data type (18 numeric, 8 categorical, 10 boolean features)
- **Target Relationship Analysis**: Detailed examination of feature correlations with new/used classification

### Feature Engineering

The feature engineering process transforms raw marketplace JSON data into model-ready features:

#### Derived Features:
- **Geographic signals**: `seller_state`, `seller_city` from nested address data
- **Payment preferences**: Consolidated payment method indicators (`nmp_mercadopago`, `nmp_cash`, etc.)
- **Content signals**: `has_video`, `num_pictures`, `has_attributes` for listing quality
- **Business signals**: `is_official_store`, `automatic_relist`, `shipping_free` for seller behavior
- **Text analytics**: Keyword extraction from `title` and `warranty` fields

#### Data Transformations:
- **Currency normalization**: USD prices converted to ARS using 2015 exchange rates
- **Log transformation**: Price normalization to handle outliers
- **Categorical consolidation**: Warranty reduced to 3 categories (has_warranty, no_warranty, unknown)
- **Title analysis**: Keyword-based classification signals from product titles

### Model Selection and Performance

The models evaluated include:

1. **Logistic Regression**: Baseline model achieving 82% accuracy
   - Pros: Fast, interpretable, handles large datasets
   - Cons: Limited complex interaction modeling

2. **XGBoost (Selected)**: Optimized gradient boosting achieving 89% accuracy  
   - Pros: Superior performance, handles complex interactions, fast training
   - Cons: Less interpretable than logistic regression

3. **Text Embeddings (Evaluated)**: SentenceTransformer embeddings
   - Result: Minimal improvement (+1% accuracy) at significant complexity cost
   - Decision: Not included in final model

#### Final Model Performance
```
               precision    recall  f1-score   support
     new         0.90      0.89      0.89      5406
    used         0.87      0.89      0.88      4594

accuracy                           0.89     10000
macro avg       0.88      0.89      0.89     10000
weighted avg    0.89      0.89      0.89     10000
```

#### Optimal Hyperparameters (via RandomizedSearchCV)
```python
{
    'max_depth': 15,
    'n_estimators': 90, 
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 1.0,
    'gamma': 0.7
}
```

### Business Impact and Risk Mitigation

The model addresses critical business risks:

- **Primary Risk**: Used → New misclassification (11% rate)
  - Impact: Customer dissatisfaction, returns, reputation damage
- **Secondary Risk**: New → Used misclassification (11% rate) 
  - Impact: Revenue loss but lower customer trust impact

### Feature Importance Analysis

Top predictive features identified:
1. **`listing_type_id_free`**: Strong indicator of used products (free listings)
2. **`initial_quantity`**: New products typically listed in higher quantities
3. **`title_2_new/used`**: Direct keyword signals from product titles
4. **`sold_quantity`**: Sales velocity differs significantly between categories
5. **Operational markers**: `automatic_relist`, `shipping_free`, `is_official_store`

### Technical Architecture

- **Pipeline Design**: Modular sklearn-compatible pipeline with preprocessing and model stages
- **Preprocessing**: StandardScaler for numeric, OneHotEncoder for categorical, passthrough for boolean/dummy features  
- **Cross-validation**: StratifiedKFold for robust performance estimation
- **Hyperparameter Optimization**: RandomizedSearchCV for efficient parameter tuning

### Evaluation Methodology

#### Metric Selection: ROC-AUC
The area under the ROC curve was chosen as the primary evaluation metric because:
- **Threshold-independent**: Evaluates model performance across all decision thresholds
- **Class-imbalance robust**: Not biased by slight class imbalance (54% new, 46% used)
- **Business relevance**: Aligns with need to balance precision and recall for both error types
- **Range interpretation**: 0-1 scale where 1 indicates perfect classification

#### Recall and Primary Risk Mitigation
Special attention is given to **recall for the "new" class** (89% achieved) as it directly addresses the primary business risk:
- **High recall for "new"** minimizes false negatives (used products classified as new)
- **Critical importance**: Used → New misclassification has severe consequences:
  - Customer receives lower quality than expected
  - Leads to returns, complaints, and reputation damage
  - Higher business impact than revenue loss from New → Used errors
- **Model optimization**: Hyperparameter tuning prioritized minimizing this high-impact error type
- **Threshold flexibility**: ROC-AUC enables post-deployment threshold adjustment to further optimize recall based on business tolerance

Final model achieves **ROC-AUC of 0.89** and **89% recall for "new" class**, indicating strong discriminative performance with effective primary risk mitigation.

### Production Readiness

The solution includes production-ready components:
- **Robust preprocessing pipeline**: Handles missing data and new categorical values
- **Fast inference**: Optimized for real-time marketplace classification  
- **Feature importance tracking**: Enables model interpretability and monitoring
- **Configurable thresholds**: Adjustable classification thresholds for business requirements

### Future Work

- **Deep Learning**: Transformer-based models for better text understanding
- **Model Ensemble**: Combining multiple algorithms for improved robustness
- **Feature Expansion**: Additional seller behavior patterns and seasonal signals
- **Real-time Learning**: Online learning capabilities for model adaptation
- **A/B Testing Framework**: Production deployment with gradual rollout capabilities

### Dependencies

Key libraries used:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost  
- **Visualization**: matplotlib, seaborn
- **Text Processing**: sentence-transformers (for embedding experiments)
- **Model Persistence**: joblib


### AI uses considerations 

The code and conclusions written in this test were largely supported by Claude's AI. The way it was used—and which I believe yields good results—is to be very precise and objective about what is being requested, providing clear context and limitations so that it does not write unnecessary code. In this way, AI-specific tools such as skills, agents, and sub-agents were used to make the analyses performed by the AI more efficient and faster. However, although these are cutting-edge tools, they are not immune to making mistakes or to recognizing errors inherent to the person using the AI. Therefore, it is important to always verify the outputs and modifications made by the AI, verify that it did what was asked of it, verify that the code actually runs and does what it is intended to do, and, in the case of conclusions, to verify that what is written is consistent with what is shown in the code’s outputs.