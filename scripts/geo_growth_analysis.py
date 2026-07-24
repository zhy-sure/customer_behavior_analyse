"""
===============================================================================
Olist E-Commerce Analytics: Geographic & Category Growth Analysis
===============================================================================
Business Question: Which cities and product categories have the highest
                   growth potential? Where should Olist invest in promotions?

Method: City-level revenue trend analysis + City×Category matrix +
        Growth Opportunity Scoring

Output:
  - City growth ranking (top 20 fastest-growing cities)
  - Revenue trend charts for top/bottom cities
  - City × Category heatmap (growth opportunity matrix)
  - Geographic map of revenue distribution (Brazil states)
  - State-level revenue dashboard charts

Usage:
  python scripts/geo_growth_analysis.py
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# CONFIGURATION
# ===========================================================================
DATA_PATH = 'datasets/'
OUTPUT_PATH = 'outputs/'

plt.rcParams.update({
    'figure.figsize': (14, 7),
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
print("PHASE 1: Data Loading & Preparation")
print("=" * 70)

orders = pd.read_csv(DATA_PATH + 'olist_orders_dataset.csv')
customers = pd.read_csv(DATA_PATH + 'olist_customers_dataset.csv')
order_items = pd.read_csv(DATA_PATH + 'olist_order_items_dataset.csv')
products = pd.read_csv(DATA_PATH + 'olist_products_dataset.csv')
category_translation = pd.read_csv(DATA_PATH + 'product_category_name_translation.csv')
geolocation = pd.read_csv(DATA_PATH + 'olist_geolocation_dataset.csv')

# Parse dates
date_cols = ['order_purchase_timestamp', 'order_approved_at',
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col])

# Filter delivered orders
delivered = orders[orders['order_status'] == 'delivered'].copy()
delivered['order_month'] = delivered['order_purchase_timestamp'].dt.to_period('M')

# Merge with customer, items, products, categories
df = delivered.merge(customers, on='customer_id', how='left')
df = df.merge(order_items, on='order_id', how='left')
df = df.merge(products[['product_id', 'product_category_name']],
              on='product_id', how='left')
df = df.merge(category_translation, on='product_category_name', how='left')
df['total_item_value'] = df['price'] + df['freight_value']

# Clean: drop rows with missing category translations
df = df.dropna(subset=['product_category_name_english'])

print(f"  Analysis dataset: {len(df):,} rows")
print(f"  Date range: {delivered['order_purchase_timestamp'].min().date()} to "
      f"{delivered['order_purchase_timestamp'].max().date()}")
print(f"  Unique cities: {df['customer_city'].nunique():,}")
print(f"  Unique categories: {df['product_category_name_english'].nunique():,}\n")


# ===========================================================================
# 2. CITY-LEVEL MONTHLY REVENUE TRENDS
# ===========================================================================
print("=" * 70)
print("PHASE 2: City-Level Revenue Analysis")
print("=" * 70)

# Monthly revenue by city
city_monthly = df.groupby(['customer_city', 'customer_state', 'order_month']).agg(
    order_count=('order_id', 'nunique'),
    unique_customers=('customer_unique_id', 'nunique'),
    total_revenue=('total_item_value', 'sum'),
    avg_order_value=('total_item_value', 'mean'),
    avg_freight=('freight_value', 'mean'),
).reset_index()

city_monthly['order_month'] = city_monthly['order_month'].astype(str)

# Calculate MoM growth for each city
city_monthly = city_monthly.sort_values(['customer_city', 'customer_state', 'order_month'])
city_monthly['prev_revenue'] = city_monthly.groupby(
    ['customer_city', 'customer_state'])['total_revenue'].shift(1)
city_monthly['mom_growth_pct'] = (
    (city_monthly['total_revenue'] - city_monthly['prev_revenue'])
    / city_monthly['prev_revenue'].replace(0, np.nan) * 100
)

# City summary
city_summary = city_monthly.groupby(['customer_city', 'customer_state']).agg(
    total_orders=('order_count', 'sum'),
    total_customers=('unique_customers', 'sum'),
    total_revenue=('total_revenue', 'sum'),
    avg_order_value=('avg_order_value', 'mean'),
    avg_monthly_orders=('order_count', 'mean'),
    active_months=('order_month', 'nunique'),
    first_month=('order_month', 'min'),
    last_month=('order_month', 'max'),
    avg_mom_growth=('mom_growth_pct', 'mean'),
).reset_index()

# Early vs Recent revenue for growth rate calculation
month_list = sorted(city_monthly['order_month'].unique())
early_months = month_list[:3]
recent_months = month_list[-3:]

early_rev = city_monthly[city_monthly['order_month'].isin(early_months)].groupby(
    ['customer_city', 'customer_state'])['total_revenue'].mean().reset_index(name='early_avg_revenue')
recent_rev = city_monthly[city_monthly['order_month'].isin(recent_months)].groupby(
    ['customer_city', 'customer_state'])['total_revenue'].mean().reset_index(name='recent_avg_revenue')

city_summary = city_summary.merge(early_rev, on=['customer_city', 'customer_state'], how='left')
city_summary = city_summary.merge(recent_rev, on=['customer_city', 'customer_state'], how='left')
city_summary['growth_rate_pct'] = (
    (city_summary['recent_avg_revenue'] - city_summary['early_avg_revenue'])
    / city_summary['early_avg_revenue'].replace(0, np.nan) * 100
)

# Filter: at least 20 total orders for statistical significance
city_significant = city_summary[city_summary['total_orders'] >= 20].copy()
city_significant['revenue_share_pct'] = (
    city_significant['total_revenue'] / city_significant['total_revenue'].sum() * 100
)
city_significant['revenue_rank'] = city_significant['total_revenue'].rank(ascending=False)

# Growth opportunity score
def safe_qcut_score(series, q):
    """qcut that returns integer scores 1 to n_bins."""
    try:
        result, bins = pd.qcut(series, q=q, labels=False, retbins=True, duplicates='drop')
        return pd.Series(result + 1, index=series.index)  # 1-based scoring
    except Exception:
        return pd.Series(1, index=series.index)

city_significant['market_size_score'] = safe_qcut_score(
    city_significant['total_revenue'], 5)
city_significant['growth_score'] = safe_qcut_score(
    city_significant['growth_rate_pct'].fillna(0).rank(method='first'), 5)
city_significant['opportunity_score'] = (
    city_significant['market_size_score'] * 0.5
    + city_significant['growth_score'] * 0.5
)

print(f"  Cities with >= 20 orders: {len(city_significant):,}")
print(f"  Top 5 by revenue: {city_significant.nlargest(5, 'total_revenue')[['customer_city', 'customer_state', 'total_revenue']].to_string(index=False)}")
print(f"  Top 5 by growth: {city_significant.nlargest(5, 'growth_rate_pct')[['customer_city', 'customer_state', 'growth_rate_pct']].to_string(index=False)}\n")


# ===========================================================================
# 3. CITY × CATEGORY GROWTH ANALYSIS
# ===========================================================================
print("=" * 70)
print("PHASE 3: City × Category Growth Opportunity Matrix")
print("=" * 70)

# Revenue by city + category
city_cat = df.groupby(['customer_city', 'customer_state', 'product_category_name_english']).agg(
    order_count=('order_id', 'nunique'),
    total_revenue=('total_item_value', 'sum'),
    unique_customers=('customer_unique_id', 'nunique'),
    avg_item_value=('total_item_value', 'mean'),
).reset_index()

# Category share within each city
city_cat['city_total_revenue'] = city_cat.groupby(
    ['customer_city', 'customer_state'])['total_revenue'].transform('sum')
city_cat['category_share_pct'] = (
    city_cat['total_revenue'] / city_cat['city_total_revenue'] * 100
)
city_cat['category_rank'] = city_cat.groupby(
    ['customer_city', 'customer_state'])['total_revenue'].rank(ascending=False)

# Merge with city summary for growth data
city_cat = city_cat.merge(
    city_significant[['customer_city', 'customer_state', 'growth_rate_pct',
                       'opportunity_score', 'total_revenue', 'total_orders']],
    on=['customer_city', 'customer_state'], how='inner',
    suffixes=('', '_city')
)

# Growth Opportunity Score for city-category combinations
# = category_share × city_growth × city_size
max_growth = city_cat['growth_rate_pct'].max()
max_share = city_cat['category_share_pct'].max()
max_rev = city_cat['total_revenue_city'].max()

city_cat['growth_opp_score'] = (
    (city_cat['category_share_pct'] / max_share) * 0.4
    + (city_cat['growth_rate_pct'] / max_growth) * 0.3
    + (city_cat['total_revenue_city'] / max_rev) * 0.3
) * 100

# Top opportunities (category rank 1-5 in each city)
top_opps = city_cat[city_cat['category_rank'] <= 5].nlargest(50, 'growth_opp_score')
print(f"  Top 10 Growth Opportunities (City × Category):")
print(top_opps[['customer_city', 'customer_state', 'product_category_name_english',
                 'category_share_pct', 'growth_rate_pct', 'growth_opp_score']].head(10).to_string(index=False))
print()


# ===========================================================================
# 4. STATE-LEVEL AGGREGATION
# ===========================================================================
print("=" * 70)
print("PHASE 4: State-Level Summary")
print("=" * 70)

state_summary = df.groupby('customer_state').agg(
    city_count=('customer_city', 'nunique'),
    total_orders=('order_id', 'nunique'),
    total_customers=('customer_unique_id', 'nunique'),
    total_revenue=('total_item_value', 'sum'),
    avg_order_value=('total_item_value', 'mean'),
).reset_index()

# Join with city growth data
state_growth = city_significant.groupby('customer_state')['growth_rate_pct'].mean().reset_index()
state_growth.columns = ['customer_state', 'avg_growth_rate_pct']
state_summary = state_summary.merge(state_growth, on='customer_state', how='left')
state_summary['revenue_share_pct'] = (
    state_summary['total_revenue'] / state_summary['total_revenue'].sum() * 100
)
state_summary = state_summary.sort_values('total_revenue', ascending=False)

print(state_summary.head(10)[['customer_state', 'city_count', 'total_revenue',
                               'revenue_share_pct', 'avg_growth_rate_pct']].to_string(index=False))
print()


# ===========================================================================
# 5. VISUALIZATIONS
# ===========================================================================
print("=" * 70)
print("PHASE 5: Generating Visualizations")
print("=" * 70)

# --------------------------------------------------------------------------
# Fig 1: Top 20 Cities — Revenue vs Growth Rate (Quadrant Chart)
# --------------------------------------------------------------------------
top_cities = city_significant.nlargest(30, 'total_revenue')

fig, ax = plt.subplots(figsize=(16, 10))
scatter = ax.scatter(
    top_cities['total_revenue'], top_cities['growth_rate_pct'],
    s=top_cities['total_orders'] / 5,
    c=top_cities['opportunity_score'], cmap='RdYlGn',
    alpha=0.7, edgecolors='black', linewidth=0.5
)

# Add city labels for top performers
for _, row in top_cities.nlargest(15, 'opportunity_score').iterrows():
    ax.annotate(f"{row['customer_city']}",
                (row['total_revenue'], row['growth_rate_pct']),
                fontsize=8, alpha=0.9,
                xytext=(5, 5), textcoords='offset points')

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=top_cities['total_revenue'].median(), color='gray', linestyle='--', alpha=0.5)
ax.set_title('City Revenue vs Growth Rate (bubble size = order volume)', fontweight='bold')
ax.set_xlabel('Total Revenue (R$)')
ax.set_ylabel('Growth Rate (%)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1000:.0f}K'))
plt.colorbar(scatter, ax=ax, label='Growth Opportunity Score')

# Quadrant labels
mid_rev = top_cities['total_revenue'].median()
ax.text(top_cities['total_revenue'].max()*0.7, top_cities['growth_rate_pct'].max()*0.9,
        'STARS\n(High Revenue, High Growth)', fontsize=11, fontweight='bold',
        ha='center', color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(top_cities['total_revenue'].min()*1.3, top_cities['growth_rate_pct'].max()*0.9,
        'RISING\n(Low Revenue, High Growth)', fontsize=11, fontweight='bold',
        ha='center', color='darkblue',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_PATH + '06_city_growth_quadrant.png')
plt.close()
print("  [OK] 06_city_growth_quadrant.png")

# --------------------------------------------------------------------------
# Fig 2: Monthly Revenue Trend — Top 5 Cities
# --------------------------------------------------------------------------
top5_cities = city_significant.nlargest(5, 'total_revenue')[['customer_city', 'customer_state']]
city_trend = city_monthly.merge(top5_cities, on=['customer_city', 'customer_state'])
city_trend_pivot = city_trend.pivot_table(
    index='order_month', columns='customer_city',
    values='total_revenue', aggfunc='sum'
)

fig, ax = plt.subplots(figsize=(16, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for city, color in zip(city_trend_pivot.columns, colors):
    ax.plot(range(len(city_trend_pivot)), city_trend_pivot[city].values,
            marker='o', linewidth=2, label=city, color=color, markersize=5)

ax.set_title('Monthly Revenue Trend — Top 5 Cities', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue (R$)')
ax.set_xticks(range(0, len(city_trend_pivot), 2))
ax.set_xticklabels([str(m) for m in city_trend_pivot.index[::2]], rotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1000:.0f}K'))
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_PATH + '07_top5_city_trends.png')
plt.close()
print("  [OK] 07_top5_city_trends.png")

# --------------------------------------------------------------------------
# Fig 3: Bottom 5 Cities — Revenue Trend (Growth Potential)
# --------------------------------------------------------------------------
# Cities with recent 3-month average < early but with growth
rising_cities = city_significant[
    (city_significant['total_orders'] >= 20) &
    (city_significant['growth_rate_pct'] > 50)
].nlargest(5, 'growth_rate_pct')[['customer_city', 'customer_state']]

if len(rising_cities) > 0:
    rise_trend = city_monthly.merge(rising_cities, on=['customer_city', 'customer_state'])
    rise_pivot = rise_trend.pivot_table(
        index='order_month', columns='customer_city',
        values='total_revenue', aggfunc='sum'
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    for city in rise_pivot.columns:
        ax.plot(range(len(rise_pivot)), rise_pivot[city].values,
                marker='s', linewidth=2, label=city, markersize=5)

    ax.set_title('Monthly Revenue Trend — Top 5 Fastest-Growing Cities', fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue (R$)')
    ax.set_xticks(range(0, len(rise_pivot), 2))
    ax.set_xticklabels([str(m) for m in rise_pivot.index[::2]], rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:,.0f}'))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH + '08_rising_cities_trends.png')
    plt.close()
    print("  [OK] 08_rising_cities_trends.png")
else:
    print("  [SKIP] 08_rising_cities_trends.png (insufficient data)")

# --------------------------------------------------------------------------
# Fig 4: State Revenue Distribution (Bar Chart)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))
states_sorted = state_summary.head(15)
bars = ax.bar(range(len(states_sorted)), states_sorted['total_revenue'],
              color=plt.cm.RdYlGn(states_sorted['revenue_share_pct'] / states_sorted['revenue_share_pct'].max()))

ax.set_title('Revenue by State — Top 15 States', fontweight='bold')
ax.set_xlabel('State')
ax.set_ylabel('Total Revenue (R$)')
ax.set_xticks(range(len(states_sorted)))
ax.set_xticklabels(states_sorted['customer_state'])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x/1000:.0f}K'))

# Add revenue share labels
for i, (_, row) in enumerate(states_sorted.iterrows()):
    ax.text(i, row['total_revenue'] + max(states_sorted['total_revenue'])*0.02,
            f"{row['revenue_share_pct']:.1f}%", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + '09_state_revenue.png')
plt.close()
print("  [OK] 09_state_revenue.png")

# --------------------------------------------------------------------------
# Fig 5: Category × State Revenue Heatmap
# ---------------------------------------------------------------------------
state_cat = df.groupby(['customer_state', 'product_category_name_english']).agg(
    total_revenue=('total_item_value', 'sum')
).reset_index()

# Top 10 categories and top 10 states
top10_cats = state_cat.groupby('product_category_name_english')['total_revenue'].sum().nlargest(10).index
top10_states = state_summary.nlargest(10, 'total_revenue')['customer_state'].tolist()

heatmap_data = state_cat[
    state_cat['product_category_name_english'].isin(top10_cats) &
    state_cat['customer_state'].isin(top10_states)
].pivot_table(
    index='customer_state', columns='product_category_name_english',
    values='total_revenue', aggfunc='sum', fill_value=0
)

# Normalize: percentage of state revenue
heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(18, 8))
sns.heatmap(heatmap_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': '% of State Revenue'})
ax.set_title('Category Revenue Share by State (Top 10 States × Top 10 Categories)', fontweight='bold')
ax.set_xlabel('Product Category')
ax.set_ylabel('State')
plt.tight_layout()
plt.savefig(OUTPUT_PATH + '10_state_category_heatmap.png')
plt.close()
print("  [OK] 10_state_category_heatmap.png")

# --------------------------------------------------------------------------
# Fig 6: Growth Opportunity Matrix — Top City×Category Combinations
# --------------------------------------------------------------------------
top_opps_plot = top_opps.head(20).copy()
top_opps_plot['label'] = top_opps_plot['customer_city'] + ' - ' + top_opps_plot['product_category_name_english']

fig, ax = plt.subplots(figsize=(14, 10))
colors_opp = plt.cm.RdYlGn(top_opps_plot['growth_opp_score'] / top_opps_plot['growth_opp_score'].max())
bars = ax.barh(range(len(top_opps_plot)), top_opps_plot['growth_opp_score'], color=colors_opp, edgecolor='white')
ax.set_yticks(range(len(top_opps_plot)))
ax.set_yticklabels(top_opps_plot['label'], fontsize=9)
ax.set_title('Top 20 Growth Opportunities — City × Category', fontweight='bold')
ax.set_xlabel('Growth Opportunity Score (0-100)')
ax.invert_yaxis()

for i, (_, row) in enumerate(top_opps_plot.iterrows()):
    ax.text(row['growth_opp_score'] + 1, i,
            f"Share: {row['category_share_pct']:.1f}% | Growth: {row['growth_rate_pct']:.0f}%",
            va='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + '11_growth_opportunity_matrix.png')
plt.close()
print("  [OK] 11_growth_opportunity_matrix.png")


# ===========================================================================
# 6. BUSINESS INSIGHTS
# ===========================================================================
print("\n" + "=" * 70)
print("PHASE 6: Key Business Insights")
print("=" * 70)

# Top growth cities
top_growth_cities = city_significant.nlargest(5, 'growth_rate_pct')
top_revenue_cities = city_significant.nlargest(5, 'total_revenue')
top_states = state_summary.nlargest(3, 'total_revenue')
top_categories = df.groupby('product_category_name_english')['total_item_value'].sum().nlargest(5)

stars = city_significant[
    (city_significant['total_revenue'] > city_significant['total_revenue'].median()) &
    (city_significant['growth_rate_pct'] > city_significant['growth_rate_pct'].median())
]

print(f"""
  INSIGHT 1 — Geographic Concentration:
    Top 3 states ({', '.join(top_states['customer_state'].head(3).tolist())})
    account for {top_states['revenue_share_pct'].head(3).sum():.1f}% of total revenue.
    → ACTION: Maintain dominance in core states, second-tier cities as expansion targets.

  INSIGHT 2 — Growth Hotspots:
    {len(stars)} cities are "Stars" (above-median revenue AND growth).
    Fastest-growing city: {top_growth_cities.iloc[0]['customer_city']}
    (growth: {top_growth_cities.iloc[0]['growth_rate_pct']:.0f}%).
    → ACTION: Pilot promotional campaigns in these cities first.

  INSIGHT 3 — Category Opportunities:
    Top growing categories: {', '.join(top_categories.head(3).index.tolist())}
    Top opportunity: {top_opps.iloc[0]['product_category_name_english']}
    in {top_opps.iloc[0]['customer_city']}
    (opportunity score: {top_opps.iloc[0]['growth_opp_score']:.1f}).
    → ACTION: Run category-specific flash sales in high-scoring city-category combos.

  INSIGHT 4 — Underserved Markets:
    Cities with high growth but low absolute revenue are prime for
    aggressive marketing investment — low competition, high upside.
""")

# ===========================================================================
# 7. EXPORT FOR POWER BI
# ===========================================================================
print("=" * 70)
print("PHASE 7: Exporting for Power BI")
print("=" * 70)

city_significant.to_csv(OUTPUT_PATH + 'city_growth_summary.csv', index=False)
print(f"  [OK] city_growth_summary.csv ({len(city_significant)} rows)")

city_monthly.to_csv(OUTPUT_PATH + 'city_monthly_revenue.csv', index=False)
print(f"  [OK] city_monthly_revenue.csv ({len(city_monthly)} rows)")

top_opps.to_csv(OUTPUT_PATH + 'growth_opportunities_top50.csv', index=False)
print(f"  [OK] growth_opportunities_top50.csv ({len(top_opps)} rows)")

state_summary.to_csv(OUTPUT_PATH + 'state_summary.csv', index=False)
print(f"  [OK] state_summary.csv ({len(state_summary)} rows)")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print(f"Outputs saved to: {OUTPUT_PATH}")
print("=" * 70)
