-- ===========================================================================
-- Olist E-Commerce Analytics: Geographic & Category Growth Analysis
-- ===========================================================================
-- Purpose: Identify high-growth cities and categories to guide
--          regional marketing investment and promotional targeting.
--
-- Key Questions:
--   1. Which cities have the fastest-growing revenue?
--   2. Which categories are under-penetrated in high-potential cities?
--   3. Where should Olist prioritize promotional campaigns?
-- ===========================================================================


-- ---------------------------------------------------------------------------
-- STEP 1: City-level Revenue & Growth Metrics
-- ---------------------------------------------------------------------------
-- Monthly revenue by city, with month-over-month growth rates.

CREATE OR REPLACE VIEW vw_city_monthly_revenue AS
WITH monthly_city AS (
    SELECT 
        c.customer_city,
        c.customer_state,
        DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01') AS order_month,
        COUNT(DISTINCT o.order_id) AS order_count,
        COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
        ROUND(SUM(oi.price + oi.freight_value), 2) AS total_revenue,
        ROUND(AVG(oi.price + oi.freight_value), 2) AS avg_order_value
    FROM olist_orders o
    INNER JOIN olist_customers c ON o.customer_id = c.customer_id
    INNER JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_city, c.customer_state, order_month
)
SELECT 
    customer_city,
    customer_state,
    order_month,
    order_count,
    unique_customers,
    total_revenue,
    avg_order_value,
    -- Month-over-month growth
    ROUND(
        (total_revenue - LAG(total_revenue) OVER (
            PARTITION BY customer_city, customer_state 
            ORDER BY order_month
        )) / NULLIF(LAG(total_revenue) OVER (
            PARTITION BY customer_city, customer_state 
            ORDER BY order_month
        ), 0) * 100, 1
    ) AS revenue_mom_growth_pct,
    -- 3-month moving average revenue
    ROUND(AVG(total_revenue) OVER (
        PARTITION BY customer_city, customer_state 
        ORDER BY order_month 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS revenue_3m_ma
FROM monthly_city;


-- ---------------------------------------------------------------------------
-- STEP 2: City Summary — Aggregated metrics across full time period
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_city_summary AS
WITH city_totals AS (
    SELECT 
        customer_city,
        customer_state,
        SUM(order_count) AS total_orders,
        SUM(unique_customers) AS total_customers,
        ROUND(SUM(total_revenue), 2) AS total_revenue,
        ROUND(AVG(avg_order_value), 2) AS avg_order_value,
        MIN(order_month) AS first_active_month,
        MAX(order_month) AS last_active_month,
        -- Recent 3-month average revenue (for growth scoring)
        ROUND(AVG(CASE 
            WHEN order_month >= DATE_SUB((SELECT MAX(order_month) FROM vw_city_monthly_revenue), INTERVAL 3 MONTH)
            THEN total_revenue 
        END), 2) AS recent_3m_avg_revenue,
        -- Early 3-month average revenue (for baseline)
        ROUND(AVG(CASE 
            WHEN order_month <= DATE_ADD((SELECT MIN(order_month) FROM vw_city_monthly_revenue), INTERVAL 3 MONTH)
            THEN total_revenue 
        END), 2) AS early_3m_avg_revenue
    FROM vw_city_monthly_revenue
    GROUP BY customer_city, customer_state
)
SELECT 
    *,
    -- Growth rate: (recent - early) / early
    ROUND(
        (recent_3m_avg_revenue - early_3m_avg_revenue) 
        / NULLIF(early_3m_avg_revenue, 0) * 100, 1
    ) AS growth_rate_pct,
    -- Active months
    TIMESTAMPDIFF(MONTH, first_active_month, last_active_month) + 1 AS active_months,
    -- Revenue share
    ROUND(total_revenue * 100.0 / SUM(total_revenue) OVER(), 2) AS revenue_share_pct,
    -- Revenue rank
    ROW_NUMBER() OVER (ORDER BY total_revenue DESC) AS revenue_rank
FROM city_totals;


-- ---------------------------------------------------------------------------
-- STEP 3: City × Category Matrix — Revenue per city-category combination
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_city_category_revenue AS
SELECT 
    c.customer_city,
    c.customer_state,
    t.product_category_name_english AS category,
    COUNT(DISTINCT o.order_id) AS order_count,
    COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS category_revenue,
    ROUND(AVG(oi.price + oi.freight_value), 2) AS avg_item_value,
    ROUND(AVG(rev.review_score), 2) AS avg_review_score
FROM olist_orders o
INNER JOIN olist_customers c ON o.customer_id = c.customer_id
INNER JOIN olist_order_items oi ON o.order_id = oi.order_id
INNER JOIN olist_products p ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation t 
    ON p.product_category_name = t.product_category_name
LEFT JOIN olist_order_reviews rev ON o.order_id = rev.order_id
WHERE o.order_status = 'delivered'
    AND t.product_category_name_english IS NOT NULL
GROUP BY c.customer_city, c.customer_state, t.product_category_name_english;


-- ---------------------------------------------------------------------------
-- STEP 4: Growth Opportunity Score — Identify high-potential city-categories
-- ---------------------------------------------------------------------------
-- Score = (Category Market Share in City) × (City Growth Rate) × (City Size)
-- This identifies categories that are popular AND in growing cities.

CREATE OR REPLACE VIEW vw_growth_opportunities AS
WITH category_rank_in_city AS (
    SELECT 
        ccr.*,
        -- Category rank within city (1 = top category by revenue)
        ROW_NUMBER() OVER (
            PARTITION BY customer_city, customer_state 
            ORDER BY category_revenue DESC
        ) AS category_rank,
        -- Category share of city's total revenue
        ROUND(
            category_revenue * 100.0 / SUM(category_revenue) OVER (
                PARTITION BY customer_city, customer_state
            ), 2
        ) AS category_share_pct,
        -- How many cities carry this category
        COUNT(*) OVER (PARTITION BY category) AS cities_with_category
    FROM vw_city_category_revenue ccr
    JOIN vw_city_summary cs 
        ON ccr.customer_city = cs.customer_city 
        AND ccr.customer_state = cs.customer_state
    WHERE cs.total_revenue > 1000  -- Filter out tiny markets
)
SELECT 
    rc.*,
    cs.total_revenue AS city_total_revenue,
    cs.total_orders AS city_total_orders,
    cs.growth_rate_pct AS city_growth_rate_pct,
    cs.revenue_share_pct AS city_revenue_share_pct,
    cs.revenue_rank AS city_revenue_rank,
    -- Growth Opportunity Score (0-100 scale)
    -- High score = large category share in a fast-growing, sizeable city
    ROUND(
        (rc.category_share_pct / 100.0) * 0.4   -- category importance
        + (cs.growth_rate_pct / NULLIF((SELECT MAX(growth_rate_pct) FROM vw_city_summary), 0)) * 0.3  -- growth
        + (cs.revenue_share_pct / NULLIF((SELECT MAX(revenue_share_pct) FROM vw_city_summary), 0)) * 0.3  -- market size
    , 4) * 100 AS growth_opportunity_score
FROM category_rank_in_city rc
JOIN vw_city_summary cs 
    ON rc.customer_city = cs.customer_city 
    AND rc.customer_state = cs.customer_state
WHERE rc.category_rank <= 10  -- Top 10 categories per city
ORDER BY growth_opportunity_score DESC;


-- ---------------------------------------------------------------------------
-- STEP 5: State-Level Summary (for geographic roll-up)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_state_summary AS
SELECT 
    customer_state,
    COUNT(DISTINCT customer_city) AS city_count,
    SUM(total_orders) AS total_orders,
    SUM(total_customers) AS total_customers,
    ROUND(SUM(total_revenue), 2) AS total_revenue,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(growth_rate_pct), 1) AS avg_growth_rate_pct,
    ROUND(SUM(total_revenue) * 100.0 / SUM(SUM(total_revenue)) OVER(), 2) AS revenue_share_pct
FROM vw_city_summary
GROUP BY customer_state
ORDER BY total_revenue DESC;


-- ---------------------------------------------------------------------------
-- FINAL EXPORTS (run these to extract data for Python)
-- ---------------------------------------------------------------------------

-- Export 1: Top 30 growth cities
SELECT * FROM vw_city_summary 
WHERE total_orders >= 20
ORDER BY growth_rate_pct DESC 
LIMIT 30;

-- Export 2: Top 50 growth opportunities (city × category)
SELECT * FROM vw_growth_opportunities 
WHERE city_total_orders >= 10 AND category_rank <= 5
ORDER BY growth_opportunity_score DESC 
LIMIT 50;

-- Export 3: State-level summary
SELECT * FROM vw_state_summary;

-- Export 4: Top categories by total revenue (benchmark)
SELECT 
    category,
    SUM(category_revenue) AS total_revenue,
    COUNT(DISTINCT customer_city) AS city_count,
    ROUND(AVG(avg_review_score), 2) AS avg_review
FROM vw_city_category_revenue
GROUP BY category
ORDER BY total_revenue DESC
LIMIT 20;
