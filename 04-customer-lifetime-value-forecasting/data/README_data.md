# Data Guide

## Data Source

The supplied notebook generates **synthetic transaction data**. No real customer records or personally identifiable information are included in this repository.

The deployed app does not need a training dataset. It loads the committed model and preprocessing artifacts from `models/`.

## Included Sample

`sample_input.csv` contains 24 privacy-safe customer rows that match the deployed model schema. Use it to test batch scoring locally or in Streamlit.

## Required Batch Columns

The most reliable batch input contains these columns:

```text
customer_id
n_orders
total_revenue
avg_order_value
std_order_value
total_quantity
avg_quantity
category_diversity
payment_diversity
channel_diversity
avg_discount
recency_days
tenure_days
orders_per_month
revenue_per_month
loyalty_score
discount_sensitivity
engagement_score
cohort_age_months
avg_revenue_per_active_month
country
preferred_channel
dominant_category
acquisition_quarter
customer_segment_name
```

The inference pipeline can calculate some rate fields when their source fields are present. Missing numerical fields are filled with training means and reported to the user. Unknown categories are replaced by documented fallback categories and also reported.

## Data Safety

Do not commit raw customer exports, email addresses, names, phone numbers, street addresses, payment data, or other sensitive fields.

For a real implementation:

1. Store raw data in an approved governed platform.
2. Create de-identified customer-level features.
3. Export only an approved sample or schema to GitHub.
4. Keep credentials in environment variables or Streamlit secrets, never in code.
5. Add data-quality, consent, retention, and access-control checks.

## Regenerating Synthetic Training Data

The training script generates privacy-safe data internally:

```bash
python -m src.model_training --customers 6000 --epochs 60
```

No external dataset download is required.
