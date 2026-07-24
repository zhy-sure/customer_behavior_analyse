-- ===========================================================================
-- Olist E-Commerce Analytics: RFM Customer Segmentation Data Extraction
-- ===========================================================================
-- Purpose: Extract and engineer RFM features for customer segmentation.
-- RFM = Recency (days since last purchase), Frequency (total orders),
--        Monetary (total lifetime spend)
-- 
-- Output: One row per unique customer with RFM scores, segment labels,
--         and enriched demographic/purchase behavior attributes.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- STEP 1: Create a denormalized view of delivered orders with all dimensions
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_order_full AS
SELECT 
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) AS delivery_days,
    DATEDIFF(o.order_estimated_delivery_date, o.order_delivered_customer_date) AS delivery_delay,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    c.customer_zip_code_prefix,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    (oi.price + oi.freight_value) AS total_item_value,
    p.product_category_name,
    t.product_category_name_english,
    op.payment_type,
    op.payment_installments,
    op.payment_value,
    rev.review_score
FROM olist_orders o
LEFT JOIN olist_customers c 
    ON o.customer_id = c.customer_id
LEFT JOIN olist_order_items oi 
    ON o.order_id = oi.order_id
LEFT JOIN olist_products p 
    ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation t 
    ON p.product_category_name = t.product_category_name
LEFT JOIN olist_order_payments op 
    ON o.order_id = op.order_id
LEFT JOIN olist_order_reviews rev 
    ON o.order_id = rev.order_id
WHERE o.order_status = 'delivered';


-- ---------------------------------------------------------------------------
-- STEP 2: Aggregate at customer level — core RFM metrics
-- ---------------------------------------------------------------------------
-- Note: customer_unique_id is the real person; customer_id is per-order.
-- We aggregate at customer_unique_id level for true customer analysis.

CREATE OR REPLACE VIEW vw_customer_rfm AS
WITH customer_orders AS (
    -- One row per unique customer, per order
    SELECT 
        customer_unique_id,
        customer_city,
        customer_state,
        order_id,
        MAX(order_purchase_timestamp) AS order_purchase_timestamp,
        SUM(total_item_value) AS order_total_value,
        AVG(review_score) AS order_avg_review,
        COUNT(DISTINCT product_id) AS order_distinct_products,
        MAX(payment_type) AS payment_type,
        MAX(payment_installments) AS payment_installments,
        AVG(delivery_days) AS order_delivery_days
    FROM vw_order_full
    GROUP BY customer_unique_id, customer_city, customer_state, order_id
),
-- Reference date: 30 days after the last order in the dataset (avoids negative recency)
ref_date AS (
    SELECT DATE_ADD(MAX(order_purchase_timestamp), INTERVAL 1 DAY) AS reference_date
    FROM customer_orders
)
SELECT 
    co.customer_unique_id,
    co.customer_city,
    co.customer_state,
    -- RFM Metrics
    DATEDIFF((SELECT reference_date FROM ref_date), MAX(co.order_purchase_timestamp)) AS recency_days,
    COUNT(DISTINCT co.order_id) AS frequency,
    ROUND(SUM(co.order_total_value), 2) AS monetary,
    -- Derived metrics
    ROUND(AVG(co.order_total_value), 2) AS avg_order_value,
    ROUND(AVG(co.order_avg_review), 2) AS avg_review_score,
    ROUND(AVG(co.order_distinct_products), 1) AS avg_products_per_order,
    ROUND(AVG(co.order_delivery_days), 1) AS avg_delivery_days,
    MIN(co.order_purchase_timestamp) AS first_purchase_date,
    MAX(co.order_purchase_timestamp) AS last_purchase_date
FROM customer_orders co
GROUP BY co.customer_unique_id, co.customer_city, co.customer_state;


-- ---------------------------------------------------------------------------
-- STEP 3: Enrich with category preferences and payment behavior
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_customer_profile AS
WITH category_prefs AS (
    -- Top 3 categories per customer by spend
    SELECT 
        customer_unique_id,
        product_category_name_english,
        SUM(total_item_value) AS category_spend,
        COUNT(DISTINCT order_id) AS category_order_count,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id 
            ORDER BY SUM(total_item_value) DESC
        ) AS category_rank
    FROM vw_order_full
    WHERE product_category_name_english IS NOT NULL
    GROUP BY customer_unique_id, product_category_name_english
),
payment_prefs AS (
    -- Preferred payment method per customer
    SELECT 
        customer_unique_id,
        payment_type,
        COUNT(*) AS payment_count,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id 
            ORDER BY COUNT(*) DESC
        ) AS payment_rank
    FROM vw_order_full
    WHERE payment_type IS NOT NULL
    GROUP BY customer_unique_id, payment_type
)
SELECT 
    r.*,
    -- Top category
    MAX(CASE WHEN cp.category_rank = 1 THEN cp.product_category_name_english END) AS top_category,
    MAX(CASE WHEN cp.category_rank = 2 THEN cp.product_category_name_english END) AS second_category,
    MAX(CASE WHEN cp.category_rank = 3 THEN cp.product_category_name_english END) AS third_category,
    -- Preferred payment
    MAX(CASE WHEN pp.payment_rank = 1 THEN pp.payment_type END) AS preferred_payment,
    -- Customer lifetime (months)
    ROUND(DATEDIFF(r.last_purchase_date, r.first_purchase_date) / 30.0, 1) AS customer_lifetime_months,
    -- Monthly average spend
    CASE 
        WHEN DATEDIFF(r.last_purchase_date, r.first_purchase_date) > 0 
        THEN ROUND(r.monetary / (DATEDIFF(r.last_purchase_date, r.first_purchase_date) / 30.0), 2)
        ELSE r.monetary 
    END AS avg_monthly_spend
FROM vw_customer_rfm r
LEFT JOIN category_prefs cp 
    ON r.customer_unique_id = cp.customer_unique_id AND cp.category_rank <= 3
LEFT JOIN payment_prefs pp 
    ON r.customer_unique_id = pp.customer_unique_id AND pp.payment_rank = 1
GROUP BY 
    r.customer_unique_id, r.customer_city, r.customer_state,
    r.recency_days, r.frequency, r.monetary, r.avg_order_value,
    r.avg_review_score, r.avg_products_per_order, r.avg_delivery_days,
    r.first_purchase_date, r.last_purchase_date;


-- ---------------------------------------------------------------------------
-- STEP 4: RFM Scoring (1-5 scale, 5 = best)
-- ---------------------------------------------------------------------------
-- Using NTILE to split customers into quintiles for each RFM dimension

CREATE OR REPLACE VIEW vw_customer_rfm_scores AS
SELECT 
    *,
    -- Recency: lower is better → reverse the score
    6 - NTILE(5) OVER (ORDER BY recency_days ASC) AS r_score,
    -- Frequency: higher is better
    NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
    -- Monetary: higher is better
    NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
FROM vw_customer_profile;


-- ---------------------------------------------------------------------------
-- STEP 5: Segment Assignment
-- ---------------------------------------------------------------------------
-- Based on classic RFM segmentation rules.
-- Segment definitions adapted for e-commerce with dominant purchase patterns.

CREATE OR REPLACE VIEW vw_customer_segments AS
SELECT 
    *,
    (r_score + f_score + m_score) AS rfm_total_score,
    ROUND((r_score + f_score + m_score) / 3.0, 1) AS rfm_avg_score,
    CASE
        -- Champions: Recent, frequent, high spenders
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 
            THEN 'Champions'
        -- Loyal Customers: Frequent buyers, not necessarily highest spend
        WHEN f_score >= 4 AND r_score >= 3 
            THEN 'Loyal Customers'
        -- Potential Loyalists: Recent with decent frequency
        WHEN r_score >= 4 AND f_score >= 2 AND m_score >= 2 
            THEN 'Potential Loyalists'
        -- New Customers: Very recent but low frequency
        WHEN r_score >= 4 AND f_score <= 1 
            THEN 'New Customers'
        -- Promising: Recent but not yet frequent
        WHEN r_score >= 3 AND f_score <= 2 AND m_score >= 2 
            THEN 'Promising'
        -- Need Attention: Above average on some metrics but slipping
        WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 
            THEN 'Need Attention'
        -- About to Sleep: Below average recency, but decent history
        WHEN r_score <= 2 AND f_score >= 2 AND m_score >= 2 
            THEN 'About to Sleep'
        -- At Risk: Previously good but haven't returned
        WHEN r_score <= 1 AND f_score >= 2 AND m_score >= 2 
            THEN 'At Risk'
        -- Cannot Lose Them: High value but inactive
        WHEN r_score <= 1 AND (f_score >= 4 OR m_score >= 4) 
            THEN 'Cannot Lose Them'
        -- Hibernating: Low across all metrics
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 
            THEN 'Hibernating'
        -- Lost: Very low across the board
        WHEN r_score <= 1 AND f_score <= 1 
            THEN 'Lost'
        ELSE 'Others'
    END AS customer_segment
FROM vw_customer_rfm_scores;


-- ---------------------------------------------------------------------------
-- STEP 6: Segment Summary Statistics (for reporting)
-- ---------------------------------------------------------------------------
SELECT 
    customer_segment,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct_of_total,
    ROUND(SUM(monetary), 2) AS total_revenue,
    ROUND(SUM(monetary) * 100.0 / SUM(SUM(monetary)) OVER(), 2) AS pct_of_revenue,
    ROUND(AVG(monetary), 2) AS avg_clv,
    ROUND(AVG(recency_days), 0) AS avg_recency_days,
    ROUND(AVG(frequency), 1) AS avg_frequency,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(avg_review_score), 2) AS avg_review_score,
    ROUND(AVG(rfm_avg_score), 2) AS avg_rfm_score
FROM vw_customer_segments
GROUP BY customer_segment
ORDER BY AVG(rfm_avg_score) DESC;


-- ---------------------------------------------------------------------------
-- FINAL EXPORT: Full customer profile with segment (ready for Python)
-- ---------------------------------------------------------------------------
SELECT * FROM vw_customer_segments;
