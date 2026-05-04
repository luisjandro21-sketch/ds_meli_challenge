import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MLA_100k.jsonlines")

NUMERICAL = [
    "price", "initial_quantity", "sold_quantity",
    "available_quantity", "num_sells", "num_pictures", "has_attributes",
]

CATEGORICAL = [
    "listing_type_id", "buying_mode", "currency_id",
    "seller_state", "warranty_2", "title_2",
]

BOOLEANS = [
    "accepts_mercadopago", "automatic_relist", "shipping_local_pick_up",
    "shipping_free", "has_parent_item_id", "common_cat", "common_quality",
    "is_official_store", "has_video", "is_active",
]

DUMMIES = [
    "nmp_agree_with_buyer", "nmp_cash", "nmp_cash_on_delivery",
    "nmp_credit_card", "nmp_mercadopago", "nmp_transfer",
    "has_variations", "tags_dragged", "tags_free_relist",
    "tags_good_quality_thumbnail", "tags_poor_quality_thumbnail",
]

BEST_PARAMS = {
    "max_depth": 15,
    "n_estimators": 90,
    "learning_rate": 0.1,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 1.0,
    "gamma": 0.7,
    "eval_metric": "auc",
    "random_state": 42,
}

# Reference metrics from the tuned notebook run (10k test set)
REFERENCE_METRICS = {
    "new": {"precision": 0.90, "recall": 0.89, "f1-score": 0.89},
    "used": {"precision": 0.87, "recall": 0.89, "f1-score": 0.88},
    "accuracy": 0.89,
}

# Alert if any metric deviates more than this amount from the reference
ALERT_THRESHOLD = 0.03

TEST_SIZE = 10_000
