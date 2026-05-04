import statistics

import numpy as np
import pandas as pd


_GOLD_VARIANTS = {"gold_special": "gold", "gold_pro": "gold", "gold_premium": "gold"}

_NMP_MAP = {
    "Transferencia bancaria": "transfer",
    "Acordar con el comprador": "agree_with_buyer",
    "Efectivo": "cash",
    "Tarjeta de crédito": "credit_card",
    "MasterCard": "credit_card",
    "Mastercard Maestro": "credit_card",
    "Visa Electron": "credit_card",
    "Contra reembolso": "cash_on_delivery",
    "Visa": "credit_card",
    "Diners": "credit_card",
    "American Express": "credit_card",
    "Giro postal": "transfer",
    "MercadoPago": "mercadopago",
    "Cheque certificado": "transfer",
}

_TAG_MAP = {"dragged_bids_and_visits": "dragged", "dragged_visits": "dragged"}

_NO_WARRANTY_KEYWORDS = [
    "sin gar", "sin warr", "no warr", "sin cober",
    "no aplica gar", "no tiene gar", "no incluye gar",
]

_USED_TITLE_KEYWORDS = [
    "usad", "used", "segunda mano", "de segund", "semi nuev",
    "seminuev", "poco uso", "buen estado", "excelente estado", "exc estado",
]

_NEW_TITLE_KEYWORDS = ["nuevo", "brand new", "sin uso", "new", "a estrenar"]

# Columns dropped as non-informative (mirrors notebook null_vars)
_NULL_VARS = [
    "seller_address", "sub_status", "seller_contact", "shipping",
    "non_mercado_pago_payment_methods", "nmp_payment_methods",
    "seller_id", "variations", "location", "site_id", "attributes",
    "tags", "listing_source", "parent_item_id", "coverage_areas",
    "descriptions", "international_delivery_mode", "picture_quality",
    "pictures", "official_store_id", "differential_pricing",
    "original_price", "thumbnail", "start_time", "stop_time",
    "date_created", "last_updated", "secure_thumbnail", "status",
    "video_id", "permalink", "geolocation", "subtitle",
    "catalog_product_id", "base_price", "deal_ids", "seller_city",
    "warranty", "title",
]


class DataCleaner:
    """Replicates the full feature-engineering pass from EDA.ipynb."""

    def __init__(self):
        self._top_categories: list | None = None
        self._top_qualities: list | None = None

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        tmp = df.copy()
        tmp = self._basic_transforms(tmp)
        # Learn top-6 categories from training data
        cat_counts = tmp["category_id"].value_counts()
        self._top_categories = cat_counts.iloc[:6].index.tolist()
        # Learn top-3 picture quality buckets
        tmp["_pic_quality"] = tmp["pictures"].apply(
            lambda x: statistics.mode([i.get("size") for i in x if i.get("size")])
            if len(x) > 0 else None
        )
        quality_counts = tmp["_pic_quality"].value_counts()
        self._top_qualities = quality_counts.iloc[:3].index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._top_categories is None or self._top_qualities is None:
            raise RuntimeError("Call fit() before transform().")
        data = df.copy()
        data = self._basic_transforms(data)
        data = self._category_features(data)
        data = self._picture_features(data)
        data = self._text_features(data)
        data = self._drop_columns(data)
        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _basic_transforms(self, data: pd.DataFrame) -> pd.DataFrame:
        # 1. seller_address → seller_state (drop seller_city per notebook cell 66)
        data["seller_state"] = data["seller_address"].apply(
            lambda x: x.get("state", {}).get("name") if isinstance(x, dict) else None
        )

        # 2. sub_status list → string
        data["sub_status"] = data["sub_status"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
        )

        # 3. shipping features
        data["shipping_local_pick_up"] = data["shipping"].apply(
            lambda x: x.get("local_pick_up", False) if isinstance(x, dict) else False
        )
        data["shipping_free"] = data["shipping"].apply(
            lambda x: x.get("free_shipping", False) if isinstance(x, dict) else False
        )

        # 4. non_mercado_pago_payment_methods → nmp_* dummies
        data["nmp_payment_methods"] = data["non_mercado_pago_payment_methods"].apply(
            lambda x: [_NMP_MAP.get(d["description"], d["description"]) for d in x]
            if isinstance(x, list) else []
        )
        nmp_dummies = data["nmp_payment_methods"].explode().str.get_dummies()
        nmp_dummies = nmp_dummies.groupby(level=0).max().fillna(0)
        for col in nmp_dummies.columns:
            data[f"nmp_{col}"] = nmp_dummies[col]

        # 5. num_sells per seller_id
        seller_counts = (
            data.groupby("seller_id")["id"]
            .count()
            .reset_index()
            .rename(columns={"id": "num_sells"})
        )
        data = data.merge(seller_counts, on="seller_id", how="inner")

        # 6. has_variations
        data["has_variations"] = data["variations"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # 7. listing_type_id consolidation
        data["listing_type_id"] = data["listing_type_id"].replace(_GOLD_VARIANTS)

        # 8. has_attributes
        data["has_attributes"] = data["attributes"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # 9. tags → dummy columns
        data["tags"] = data["tags"].apply(
            lambda x: [_TAG_MAP.get(t, t) for t in x] if isinstance(x, list) else []
        )
        tag_dummies = data["tags"].explode().str.get_dummies()
        tag_dummies = tag_dummies.groupby(level=0).max().fillna(0)
        for col in tag_dummies.columns:
            data[f"tags_{col}"] = tag_dummies[col]

        # 12. is_official_store
        data["is_official_store"] = data["official_store_id"].notna()

        # 13. has_parent_item_id
        data["has_parent_item_id"] = data["parent_item_id"].notna()

        # 14. is_active
        data["is_active"] = data["status"] == "active"

        # 15. has_video
        data["has_video"] = data["video_id"].notna()

        # 16. price conversion(based on 2015 info) + log10
        data["price"] = np.where(
            data["currency_id"] == "ARS", data["price"], data["price"] * 9 
        )
        data["price"] = np.log10(data["price"].clip(lower=1e-9))

        # num_pictures
        data["num_pictures"] = data["pictures"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        return data

    def _category_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data["common_cat"] = data["category_id"].isin(self._top_categories)
        return data

    def _picture_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data["picture_quality"] = data["pictures"].apply(
            lambda x: statistics.mode([i.get("size") for i in x if i.get("size")])
            if len(x) > 0 else None
        )
        data["common_quality"] = data["picture_quality"].isin(self._top_qualities)
        return data

    def _text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # warranty_2
        data["warranty_2"] = np.where(
            data["warranty"].isna(),
            "unknown",
            np.where(
                data["warranty"].str.lower().str.contains(
                    "|".join(_NO_WARRANTY_KEYWORDS), na=False
                ),
                "no_warranty",
                "has_warranty",
            ),
        )

        # title_2
        data["title_2"] = np.where(
            data["title"].str.lower().str.contains(
                "|".join(_USED_TITLE_KEYWORDS), na=False
            ),
            "used",
            np.where(
                data["title"].str.lower().str.contains(
                    "|".join(_NEW_TITLE_KEYWORDS), na=False
                ),
                "new",
                "other",
            ),
        )
        return data

    def _drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in _NULL_VARS if c in data.columns]
        data = data.drop(columns=cols_to_drop)
        return data
