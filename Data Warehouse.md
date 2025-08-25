# 🏢 Data Warehouse - ML System Design Note

> **An enterprise store for structured and semi-structured data from many sources, designed for analytics, reporting, and business intelligence.**

## 🤔 What is a Data Warehouse?

Imagine you run an e-commerce company. Your customer data lives in CRM, order data comes from your website, inventory data sits in your warehouse system, and marketing data flows from various ad platforms. **A data warehouse is like a central library** where all this scattered information gets organized, cleaned, and stored in a way that makes it easy to answer business questions.

Instead of jumping between 10 different systems to understand "Which products sold best in Turkey last quarter?", you query one place and get your answer in seconds.

![Data Warehouse Architecture](assets/data-warehouse.png)

## 🎯 When to use

✅ **Many disparate sources** - You have data scattered across different systems  
✅ **Business questions** - Need dashboards, reports, and ad-hoc analysis  
✅ **Historical trends** - Want to track KPIs and performance over time  
✅ **Reliable reporting** - Need consistent, governed access to data  

## 🏦 Data Warehouse vs Alternatives

| Solution | Best for | Example |
|----------|----------|---------|
| **Data Warehouse** | Structured reporting & BI | Monthly sales reports, executive dashboards |
| **Database** | Application data | User login, product catalog |
| **Data Lake** | Raw data storage & ML | Machine learning, data science experiments |

## 🔄 How it Works (Simple Example)

**Real scenario:** An online store wants to analyze sales performance

```
1. 📊 EXTRACT: Pull customer data from CRM, orders from website, products from inventory
2. 🔄 TRANSFORM: Clean data, standardize formats, create calculated fields  
3. 📥 LOAD: Store in warehouse with organized structure
4. 📈 ANALYZE: Create dashboard showing "Revenue by Product Category by Month"
```

**Architecture Flow:**
```
📊 Sources → 📥 Ingestion → 🔄 Staging → ⚡ Transform → 🏢 Data Warehouse → 📈 Visualization
```

## 📊 Core Concepts Explained

### Facts vs Dimensions (The Building Blocks)

Think of your data like a spreadsheet:

**📈 Facts = Numbers you want to analyze**
- Sales amount: $1,250
- Quantity sold: 5 items  
- Profit margin: 23%

**📋 Dimensions = Context around those numbers**
- Customer: "John Smith from Istanbul"
- Product: "iPhone 14 Pro, Electronics category"
- Time: "March 15, 2024"

### Star Schema (How data gets organized)

```
        📋 Customer           📋 Product
            |                     |
📋 Time ----📈 Sales Fact----📋 Store
            |
        📋 Promotion
```

All your numerical data (facts) sits in the center, connected to descriptive information (dimensions).

## ⚙️ Core Design Decisions

### 🔀 ETL vs ELT
| Approach | What it means | When to use |
|----------|---------------|-------------|
| **ETL** | Clean data first, then load | When you have strict rules, limited storage |
| **ELT** | Load raw data first, clean later | Modern approach, more flexible for changes |

### 📐 Data Modeling Approaches
- **⭐ Star Schema** - Simple, fast queries (recommended for beginners)
- **❄️ Snowflake Schema** - More complex but saves storage space
- **🔄 Slowly Changing Dimensions** - Track how things change over time (e.g., customer addresses)

### ⏱️ Data Freshness
- **🔄 Batch Processing** - Update every hour/day (simpler, cheaper)
- **🌊 Stream Processing** - Update in real-time (complex, expensive)

### 🚀 Performance Tips
- **📂 Partition data** by date - Makes time-based queries super fast
- **🎯 Cluster similar data** together - Groups related records
- **💾 Use columnar storage** - Perfect for analytical queries

## 🛡️ Data Quality & Governance

### Why This Matters
Without proper governance, your warehouse becomes a "garbage dump" where nobody trusts the numbers.

### 🔍 Essential Practices
- **✅ Data contracts** - Define what each data source should provide
- **✅ Quality tests** - Catch bad data before it spreads
- **✅ Clear ownership** - Know who to ask when data looks wrong
- **🔒 Security controls** - Protect sensitive information (PII)

## 💰 Cost & Performance Optimization

### 🔍 Query Best Practices
```sql
-- ✅ GOOD: Uses partition filter, specific columns
SELECT customer_id, SUM(revenue) 
FROM sales_facts 
WHERE sale_date >= '2024-01-01'
GROUP BY customer_id;

-- ❌ BAD: No date filter, selects everything
SELECT * FROM sales_facts;
```

### 🎛️ Resource Management
- **💾 Cache common queries** - Don't recalculate the same thing
- **🧹 Regular maintenance** - Clean up old data and optimize tables
- **📊 Monitor costs** - Set alerts when spending gets high

## 🤖 ML-Specific Considerations

### 📊 Training Data Best Practices
When using warehouse data for machine learning:

- **⏰ Point-in-time correctness** - Don't use future data to predict the past
- **📸 Snapshot training data** - Keep a copy for reproducible experiments  
- **🎯 Feature documentation** - Track what each column means

**Example:** If predicting customer churn, only use data that was available before the customer actually churned.

## 🛠️ Technology Stack Options

### ☁️ Cloud Data Warehouses (Managed Services)
- **BigQuery** (Google) - Pay per query, serverless
- **Snowflake** - Works on any cloud, separate compute/storage
- **Redshift** (AWS) - Good integration with AWS ecosystem

### 🏠 Build Your Own (More Control)
- **Storage**: Amazon S3, Google Cloud Storage
- **Query Engine**: Apache Trino, Presto
- **Table Format**: Delta Lake, Apache Iceberg

### 🔧 Supporting Tools
| Need | Tool Options |
|------|-------------|
| **Data Pipelines** | Airflow, Prefect, Dagster |
| **Data Transformation** | dbt, Apache Spark |
| **Data Ingestion** | Fivetran, Airbyte, Apache Kafka |
| **Business Intelligence** | Tableau, Looker, Metabase |

## ✅ Implementation Roadmap

**Phase 1: Foundation**
- [ ] 📋 Choose 2-3 most important data sources
- [ ] 🏗️ Set up basic ingestion pipeline  
- [ ] 📊 Create simple dimensional model

**Phase 2: Production Ready**
- [ ] 🔍 Add data quality tests
- [ ] 🔒 Implement security controls
- [ ] 💰 Set up cost monitoring

**Phase 3: Scale & Optimize**
- [ ] 🚀 Optimize query performance
- [ ] 🤖 Enable ML use cases
- [ ] 📈 Expand to more data sources

## 🏷️ Naming Conventions

Keep it simple and consistent:
```
raw_salesforce_contacts     # Raw data from Salesforce
stg_salesforce_contacts     # Cleaned/standardized version  
dim_customers              # Final dimension table
fct_orders                 # Final fact table
```

## 💻 Real Query Example

```sql
-- 🎯 Business question: "What's our monthly revenue by country?"
SELECT 
    c.country,
    DATE_TRUNC('month', o.order_date) as month,
    SUM(o.total_amount) as monthly_revenue
FROM fct_orders o
JOIN dim_customers c ON o.customer_key = c.customer_key
WHERE o.order_date >= '2024-01-01'
GROUP BY c.country, DATE_TRUNC('month', o.order_date)
ORDER BY month DESC, monthly_revenue DESC;
```

## 🎯 Key Takeaways

> **A data warehouse is your "single source of truth" for business intelligence.** It takes messy, scattered data and organizes it for easy analysis. Think of it as the foundation that powers all your dashboards, reports, and data-driven decisions.
> 
> **Start simple:** Pick a few important data sources, create basic reports, then expand. Don't try to build everything at once.

## 🚀 Next Steps

1. **Learn by doing**: Try BigQuery or Snowflake free tier with sample data
2. **Read more**: Check out "The Data Warehouse Toolkit" by Kimball & Ross  
3. **Practice SQL**: Get comfortable with analytical queries
4. **Explore tools**: Play with dbt for transformations, Metabase for visualization

---

*More ML System Design notes coming soon...*
