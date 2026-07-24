"""
===============================================================================
Olist E-Commerce Analytics: RFM Customer Segmentation
===============================================================================
Business Question: What defines a high-value customer?
                   How do we identify, retain, and acquire them?

Method: RFM Analysis + K-Means Clustering + Customer Profiling

Output:
  - RFM score distribution plots
  - Customer segment pie/bar charts
  - Segment profiling (category, payment, geography, review behavior)
  - Actionable customer personas for marketing & retention

Usage:
  python scripts/rfm_analysis.py
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DATA_PATH = 'datasets/'
OUTPUT_PATH = 'outputs/'
REFERENCE_DATE = None  # Will be set to max(order_date) + 1 day

# Style settings
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
})
sns.set_style("whitegrid")
sns.set_palette("Set2")

# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
print("=" * 70)
print("PHASE 1: Data Loading & Cleaning")
print("=" * 70)

# Load core tables
orders = pd.read_csv(DATA_PATH + 'olist_orders_dataset.csv')
customers = pd.read_csv(DATA_PATH + 'olist_customers_dataset.csv')
order_items = pd.read_csv(DATA_PATH + 'olist_order_items_dataset.csv')
products = pd.read_csv(DATA_PATH + 'olist_products_dataset.csv')
payments = pd.read_csv(DATA_PATH + 'olist_order_payments_dataset.csv')
reviews = pd.read_csv(DATA_PATH + 'olist_order_reviews_dataset.csv')
category_translation = pd.read_csv(DATA_PATH + 'product_category_name_translation.csv')

# Parse datetime columns
date_cols = ['order_purchase_timestamp', 'order_approved_at',
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

print(f"  Orders loaded: {len(orders):,}")
print(f"  Customers loaded: {len(customers):,}")
print(f"  Order items loaded: {len(order_items):,}")

# ---------------------------------------------------------------------------
# 1a. Filter to delivered orders only (core business analysis)
# ---------------------------------------------------------------------------
delivered = orders[orders['order_status'] == 'delivered'].copy()
print(f"  Delivered orders: {len(delivered):,} "
      f"({len(delivered)/len(orders)*100:.1f}% of total)")

# ---------------------------------------------------------------------------
# 1b. Merge order-level data
# ---------------------------------------------------------------------------
df = delivered.merge(customers, on='customer_id', how='left')
df = df.merge(order_items, on='order_id', how='left')
df = df.merge(products[['product_id', 'product_category_name']],
              on='product_id', how='left')
df = df.merge(category_translation,
              on='product_category_name', how='left')

# Aggregate payments to order level
pay_agg = payments.groupby('order_id').agg(
    total_payment=('payment_value', 'sum'),
    payment_type_main=('payment_type', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'),
    payment_installments_max=('payment_installments', 'max')
).reset_index()
df = df.merge(pay_agg, on='order_id', how='left')

# Aggregate reviews to order level
rev_agg = reviews.groupby('order_id').agg(
    review_score=('review_score', 'mean')
).reset_index()
df = df.merge(rev_agg, on='order_id', how='left')

# Calculate total item value
df['total_item_value'] = df['price'] + df['freight_value']

print(f"  Merged dataset: {len(df):,} rows\n")


# ===========================================================================
# 2. RFM FEATURE ENGINEERING
# ===========================================================================
print("=" * 70)
print("PHASE 2: RFM Feature Engineering")
print("=" * 70)

if REFERENCE_DATE is None:
    REFERENCE_DATE = df['order_purchase_timestamp'].max() + timedelta(days=1)
print(f"  Reference date: {REFERENCE_DATE.date()}")

# Aggregate by customer_unique_id (real person, not order-level ID)
rfm = df.groupby('customer_unique_id').agg(
    # Recency: days since last purchase
    recency=('order_purchase_timestamp', lambda x: (REFERENCE_DATE - x.max()).days),
    # Frequency: number of distinct orders
    frequency=('order_id', 'nunique'),
    # Monetary: total spending
    monetary=('total_item_value', 'sum'),
    # Additional profile features
    avg_order_value=('total_item_value', 'mean'),
    first_purchase=('order_purchase_timestamp', 'min'),
    last_purchase=('order_purchase_timestamp', 'max'),
    customer_city=('customer_city', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'),
    customer_state=('customer_state', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'),
    avg_review_score=('review_score', 'mean'),
    top_category=('product_category_name_english', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'),
    preferred_payment=('payment_type_main', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'),
    total_products=('product_id', 'nunique'),
    avg_delivery_days=('order_delivered_customer_date',
                       lambda x: (x - df.loc[x.index, 'order_purchase_timestamp']).dt.days.mean()),
).reset_index()

rfm['customer_lifetime_days'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days
rfm['monthly_avg_spend'] = rfm['monetary'] / np.maximum(rfm['customer_lifetime_days'] / 30, 1)

# Handle one-time purchasers
rfm['is_repeat'] = rfm['frequency'] > 1

print(f"  Unique customers: {len(rfm):,}")
print(f"  Repeat customers: {rfm['is_repeat'].sum():,} ({rfm['is_repeat'].mean()*100:.1f}%)")
print(f"  Median recency: {rfm['recency'].median():.0f} days")
print(f"  Median frequency: {rfm['frequency'].median():.0f} orders")
print(f"  Median monetary: R$ {rfm['monetary'].median():.2f}\n")


# ===========================================================================
# 3. RFM SCORING (1-5 scale, 5 = best)
# ===========================================================================
print("=" * 70)
print("PHASE 3: RFM Scoring & Segmentation")
print("=" * 70)

# RFM Scoring: handle skewed distributions by dynamically adjusting labels
def safe_qcut(series, q, ascending=True):
    """qcut that handles duplicate bin edges by dynamically creating labels."""
    try:
        result, bins = pd.qcut(series, q=q, labels=False, retbins=True, duplicates='drop')
        n_bins = len(bins) - 1
        if ascending:
            return pd.Series(result + 1, index=series.index)
        else:
            return pd.Series(n_bins - result, index=series.index)
    except Exception:
        return pd.Series(1, index=series.index)

# Recency: lower is better → reverse scoring (higher score = more recent = better)
rfm['r_score'] = safe_qcut(rfm['recency'], 5, ascending=False)
# Frequency: higher is better
rfm['f_score'] = safe_qcut(rfm['frequency'], 5, ascending=True)
# Monetary: higher is better
rfm['m_score'] = safe_qcut(rfm['monetary'], 5, ascending=True)

rfm['rfm_total'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
rfm['rfm_avg'] = (rfm['r_score'] + rfm['f_score'] + rfm['m_score']) / 3

print("  RFM Score Distribution:")
print(f"    R (Recency): {rfm['r_score'].value_counts().sort_index().to_dict()}")
print(f"    F (Frequency): {rfm['f_score'].value_counts().sort_index().to_dict()}")
print(f"    M (Monetary): {rfm['m_score'].value_counts().sort_index().to_dict()}\n")

# ---------------------------------------------------------------------------
# 3a. Assign Customer Segments
# ---------------------------------------------------------------------------
def assign_segment(row):
    """Classify customer based on RFM scores using business rules.
    
    Adapted for e-commerce with high one-time-purchaser rate (~97%).
    Uses K-Means clusters as primary segmentation and RFM scores
    for descriptive labeling.
    """
    r, f, m = row['r_score'], row['f_score'], row['m_score']
    cluster = row['cluster']

    # Define cluster profiles based on K-Means output
    # These will be overridden by the cluster mapping below
    cluster_map = {
        0: 'High-Value One-Time Buyers',  # High monetary but single purchase
        1: 'New/Low-Value Buyers',         # Recent, low frequency, low monetary
        2: 'Repeat Buyers',                # Multiple purchases
        3: 'Recent One-Time Buyers',       # Very recent, single purchase
        4: 'Dormant One-Time Buyers',      # Old single purchase
    }
    
    if cluster in cluster_map:
        return cluster_map[cluster]
    
    # Fallback to RFM rules (kept for completeness)
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif f >= 4 and r >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f >= 2 and m >= 2:
        return 'Potential Loyalists'
    elif r >= 4 and f <= 1:
        return 'New Customers'
    elif r >= 3 and f <= 2 and m >= 2:
        return 'Promising'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'Need Attention'
    elif r <= 2 and f >= 2 and m >= 2:
        return 'About to Sleep'
    elif r <= 1 and f >= 2 and m >= 2:
        return 'At Risk'
    elif r <= 1 and (f >= 4 or m >= 4):
        return 'Cannot Lose Them'
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Hibernating'
    elif r <= 1 and f <= 1:
        return 'Lost'
    else:
        return 'Others'

# ===========================================================================
# 4. K-MEANS CLUSTERING
# ===========================================================================
print("=" * 70)
print("PHASE 4: K-Means Clustering")
print("=" * 70)

# Standardize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

# Elbow method to find optimal k
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

# Final clustering with k=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# Label clusters by their RFM characteristics
cluster_profiles = rfm.groupby('cluster').agg(
    size=('customer_unique_id', 'count'),
    avg_recency=('recency', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_monetary=('monetary', 'mean'),
).reset_index()
cluster_profiles['size_pct'] = round(cluster_profiles['size']/len(rfm)*100, 1)
print("  K-Means Cluster Profiles (k=5):")
print(cluster_profiles.to_string(index=False))
print()

# ===========================================================================
# 5. SEGMENT ASSIGNMENT (based on K-Means clusters)
# ===========================================================================
print("=" * 70)
print("PHASE 5: Customer Segment Assignment")
print("=" * 70)

rfm['segment'] = rfm.apply(assign_segment, axis=1)

# Segment order for consistent display
segment_order = [
    'Repeat Buyers', 'High-Value One-Time Buyers',
    'Recent One-Time Buyers', 'New/Low-Value Buyers',
    'Dormant One-Time Buyers'
]

print("  Segment distribution:")
seg_summary = rfm.groupby('segment').agg(
    count=('customer_unique_id', 'count'),
    pct=('customer_unique_id', lambda x: round(len(x)/len(rfm)*100, 2)),
    avg_monetary=('monetary', 'mean'),
    total_revenue=('monetary', 'sum'),
    avg_recency=('recency', 'mean'),
    avg_frequency=('frequency', 'mean'),
    repeat_rate=('is_repeat', 'mean'),
).reset_index()
seg_summary['pct_of_revenue'] = round(
    seg_summary['total_revenue'] / seg_summary['total_revenue'].sum() * 100, 2
)
seg_summary['segment'] = pd.Categorical(seg_summary['segment'],
                                         categories=segment_order, ordered=True)
seg_summary = seg_summary.sort_values('segment')
print(seg_summary.to_string(index=False))
print()


# ===========================================================================
# 5. CUSTOMER PROFILING — Deep Dive into Each Segment
# ===========================================================================
print("=" * 70)
print("PHASE 6: Customer Segment Profiling")
print("=" * 70)

profile_cols = [
    'segment', 'customer_unique_id', 'monetary', 'frequency', 'recency',
    'avg_order_value', 'avg_review_score', 'monthly_avg_spend',
    'customer_state', 'top_category', 'preferred_payment',
    'customer_lifetime_days', 'is_repeat'
]

# ===== 5a. Top categories by segment =====
cat_segment = rfm.groupby(['segment', 'top_category']).size().reset_index(name='count')
cat_segment['rank'] = cat_segment.groupby('segment')['count'].rank(ascending=False)
cat_top3 = cat_segment[cat_segment['rank'] <= 3].sort_values(['segment', 'rank'])

print("  Top 3 Categories per Segment:")
for seg in segment_order:
    seg_data = cat_top3[cat_top3['segment'] == seg]
    if len(seg_data) > 0:
        cats = ', '.join(seg_data['top_category'].values)
        print(f"    {seg}: {cats}")

# ===== 5b. State/Region concentration =====
state_segment = rfm.groupby(['segment', 'customer_state']).size().reset_index(name='count')
state_segment['pct'] = state_segment.groupby('segment')['count'].transform(
    lambda x: round(x / x.sum() * 100, 1)
)
state_top = state_segment.loc[state_segment.groupby('segment')['count'].idxmax()]

print("\n  Top State per Segment:")
for _, row in state_top.iterrows():
    print(f"    {row['segment']}: {row['customer_state']} ({row['pct']}%)")

# ===== 5c. Payment preference by segment =====
pay_segment = rfm.groupby(['segment', 'preferred_payment']).size().reset_index(name='count')
pay_segment['pct'] = pay_segment.groupby('segment')['count'].transform(
    lambda x: round(x / x.sum() * 100, 1)
)
pay_top = pay_segment.loc[pay_segment.groupby('segment')['count'].idxmax()]

print("\n  Preferred Payment per Segment:")
for _, row in pay_top.iterrows():
    print(f"    {row['segment']}: {row['preferred_payment']} ({row['pct']}%)")

print()


# ===========================================================================
# 6. VISUALIZATIONS
# ===========================================================================
print("=" * 70)
print("PHASE 7: Generating Visualizations")
print("=" * 70)

# --------------------------------------------------------------------------
# Fig 1: RFM Score Distribution (3-panel)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, title, color in zip(
    axes,
    ['r_score', 'f_score', 'm_score'],
    ['Recency Score (5=Most Recent)', 'Frequency Score (5=Most Frequent)', 'Monetary Score (5=Highest Spend)'],
    ['#3498db', '#2ecc71', '#e74c3c']
):
    counts = rfm[col].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Customer Count')
    ax.set_xticks(range(1, 6))

fig.suptitle('RFM Score Distribution — Olist Customers', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + '01_rfm_score_distribution.png')
plt.close()
print("  [OK] 01_rfm_score_distribution.png")

# --------------------------------------------------------------------------
# Fig 2: Customer Segment Distribution (Pie + Bar)
# --------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Pie — customer count %
seg_counts = seg_summary.set_index('segment')['count']
colors_pie = sns.color_palette("Set3", len(seg_counts))
wedges, texts, autotexts = ax1.pie(
    seg_counts, labels=None, autopct='%1.1f%%',
    colors=colors_pie, startangle=90, pctdistance=0.85
)
ax1.set_title('Customer Distribution by Segment', fontweight='bold', pad=20)

# Bar — revenue share %
seg_sorted = seg_summary.sort_values('pct_of_revenue', ascending=True)
bars = ax2.barh(seg_sorted['segment'], seg_sorted['pct_of_revenue'],
                color=colors_pie[:len(seg_sorted)], edgecolor='white')
ax2.set_title('Revenue Contribution by Segment', fontweight='bold')
ax2.set_xlabel('Revenue Share (%)')
for bar, val in zip(bars, seg_sorted['pct_of_revenue']):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10)

# Shared legend
fig.legend(wedges, seg_counts.index, title='Segments',
           loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + '02_customer_segments.png')
plt.close()
print("  [OK] 02_customer_segments.png")

# --------------------------------------------------------------------------
# Fig 3: Segment Comparison — Key Metrics Radar
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3a. Monetary distribution by segment
seg_order_plot = seg_summary.sort_values('avg_monetary', ascending=True)['segment'].tolist()
rfm_plot = rfm[rfm['segment'].isin(seg_order_plot)].copy()
rfm_plot['segment'] = pd.Categorical(rfm_plot['segment'], categories=seg_order_plot, ordered=True)

sns.boxplot(data=rfm_plot, x='monetary', y='segment', palette='Set3',
            ax=axes[0, 0], showfliers=False)
axes[0, 0].set_title('Monetary Value Distribution by Segment', fontweight='bold')
axes[0, 0].set_xlabel('Total Spend (R$)')
axes[0, 0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:,.0f}'))

# 3b. Recency vs Frequency scatter
scatter = axes[0, 1].scatter(
    rfm['recency'], rfm['frequency'],
    c=rfm['m_score'], cmap='RdYlGn', alpha=0.4, s=rfm['monetary']/100, edgecolors='none'
)
axes[0, 1].set_title('Recency vs Frequency (color=Monetary Score)', fontweight='bold')
axes[0, 1].set_xlabel('Recency (days since last purchase)')
axes[0, 1].set_ylabel('Frequency (# of orders)')
plt.colorbar(scatter, ax=axes[0, 1], label='Monetary Score')

# 3c. Review score by segment
review_seg = rfm.groupby('segment')['avg_review_score'].mean().reindex(segment_order).dropna()
review_seg.plot(kind='bar', ax=axes[1, 0], color='#f39c12', edgecolor='white')
axes[1, 0].set_title('Average Review Score by Segment', fontweight='bold')
axes[1, 0].set_ylabel('Review Score (1-5)')
axes[1, 0].set_ylim(2.5, 5)
axes[1, 0].axhline(y=rfm['avg_review_score'].mean(), color='red', linestyle='--',
                   label=f'Overall avg: {rfm["avg_review_score"].mean():.2f}')
axes[1, 0].legend()

# 3d. Repeat rate by segment
repeat_rate = rfm.groupby('segment')['is_repeat'].mean().reindex(segment_order).dropna() * 100
repeat_rate.plot(kind='bar', ax=axes[1, 1], color='#9b59b6', edgecolor='white')
axes[1, 1].set_title('Repeat Purchase Rate by Segment', fontweight='bold')
axes[1, 1].set_ylabel('Repeat Rate (%)')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + '03_segment_comparison.png')
plt.close()
print("  [OK] 03_segment_comparison.png")

# --------------------------------------------------------------------------
# Fig 4: Elbow Method for K-Means
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, inertias, 'bo-', markersize=8, linewidth=2)
ax.axvline(x=5, color='red', linestyle='--', label='Optimal k=5')
ax.set_title('Elbow Method — Optimal Number of Clusters', fontweight='bold')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PATH + '04_elbow_method.png')
plt.close()
print("  [OK] 04_elbow_method.png")

# --------------------------------------------------------------------------
# Fig 5: Category Preference Heatmap (Segment × Category)
# --------------------------------------------------------------------------
cat_matrix = rfm.pivot_table(
    index='segment', columns='top_category',
    values='customer_unique_id', aggfunc='count', fill_value=0
)
# Normalize by row
cat_matrix_pct = cat_matrix.div(cat_matrix.sum(axis=1), axis=0) * 100
# Keep only top 10 categories
top_cats = cat_matrix.sum().nlargest(10).index
cat_matrix_pct = cat_matrix_pct[top_cats]
# Keep only main segments
main_segs = [s for s in segment_order if s in cat_matrix_pct.index]
cat_matrix_pct = cat_matrix_pct.loc[main_segs]

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(cat_matrix_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': '% of Segment'})
ax.set_title('Category Preference by Customer Segment (% of Segment)', fontweight='bold')
ax.set_xlabel('Product Category')
ax.set_ylabel('Segment')
plt.tight_layout()
plt.savefig(OUTPUT_PATH + '05_category_heatmap.png')
plt.close()
print("  [OK] 05_category_heatmap.png")


# ===========================================================================
# 7. BUSINESS INSIGHTS — Automated Summary
# ===========================================================================
print("\n" + "=" * 70)
print("PHASE 8: Key Business Insights")
print("=" * 70)

# Use cluster-based segments for insights
repeat_buyers = rfm[rfm['segment'] == 'Repeat Buyers']
high_value_ot = rfm[rfm['segment'] == 'High-Value One-Time Buyers']
recent_ot = rfm[rfm['segment'] == 'Recent One-Time Buyers']
dormant = rfm[rfm['segment'] == 'Dormant One-Time Buyers']
low_value = rfm[rfm['segment'] == 'New/Low-Value Buyers']

total_revenue = rfm['monetary'].sum()
repeat_rev = repeat_buyers['monetary'].sum() if len(repeat_buyers) > 0 else 0
hv_rev = high_value_ot['monetary'].sum() if len(high_value_ot) > 0 else 0
dormant_rev = dormant['monetary'].sum() if len(dormant) > 0 else 0

# Safe top category extraction
def safe_top_cat(segment_df):
    if len(segment_df) == 0:
        return 'N/A'
    mode_vals = segment_df['top_category'].mode()
    return mode_vals.iloc[0] if len(mode_vals) > 0 else 'N/A'

def safe_top_payment(segment_df):
    if len(segment_df) == 0:
        return 'N/A'
    mode_vals = segment_df['preferred_payment'].mode()
    return mode_vals.iloc[0] if len(mode_vals) > 0 else 'N/A'

def safe_avg_review(segment_df):
    if len(segment_df) == 0:
        return 0
    return segment_df['avg_review_score'].mean()

print(f"""
  INSIGHT 1 — Value Concentration:
    High-Value One-Time Buyers ({len(high_value_ot):,} customers, {len(high_value_ot)/len(rfm)*100:.1f}% of base)
    contribute R${hv_rev:,.0f} ({hv_rev/total_revenue*100:.1f}% of total revenue).
    Average spend: R${high_value_ot['monetary'].mean():,.0f} per customer.
    → ACTION: Target these customers for repeat purchase campaigns.
           They've shown willingness to spend — incentivize a second order.

  INSIGHT 2 — Repeat Buyer Value:
    Repeat Buyers ({len(repeat_buyers):,} customers, {len(repeat_buyers)/len(rfm)*100:.1f}%)
    average {repeat_buyers['frequency'].mean():.1f} orders and R${repeat_buyers['monetary'].mean():,.0f} CLV.
    Top category: {safe_top_cat(repeat_buyers)}.
    → ACTION: Create exclusive loyalty perks for this segment.
           Their repeat behavior is rare (only {rfm['is_repeat'].mean()*100:.1f}% of all customers).

  INSIGHT 3 — Reactivation Opportunity:
    Dormant One-Time Buyers ({len(dormant):,} customers)
    have average recency of {dormant['recency'].mean():.0f} days.
    Total dormant revenue: R${dormant_rev:,.0f}.
    → ACTION: Send "We Miss You" email with category-based recommendations.
           These customers bought once and never returned — win them back.

  INSIGHT 4 — Recent Buyer Profile:
    Recent One-Time Buyers ({len(recent_ot):,}) are the freshest segment.
    Top category: {safe_top_cat(recent_ot)}.
    Preferred payment: {safe_top_payment(recent_ot)}.
    → ACTION: Use this segment's category preferences for cross-sell
           campaigns while they're still engaged (within 90 days).
""")

# ===========================================================================
# 8. EXPORT DATA FOR POWER BI
# ===========================================================================
print("=" * 70)
print("PHASE 9: Exporting Data for Power BI")
print("=" * 70)

rfm_export = rfm[['customer_unique_id', 'segment', 'recency', 'frequency',
                   'monetary', 'r_score', 'f_score', 'm_score', 'rfm_avg',
                   'avg_order_value', 'avg_review_score', 'customer_state',
                   'top_category', 'preferred_payment', 'customer_lifetime_days',
                   'monthly_avg_spend', 'is_repeat', 'cluster']]
rfm_export.to_csv(OUTPUT_PATH + 'rfm_customer_segments.csv', index=False)
print(f"  [OK] rfm_customer_segments.csv ({len(rfm_export):,} rows)")

seg_summary.to_csv(OUTPUT_PATH + 'rfm_segment_summary.csv', index=False)
print(f"  [OK] rfm_segment_summary.csv ({len(seg_summary)} rows)")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print(f"Outputs saved to: {OUTPUT_PATH}")
print("=" * 70)
