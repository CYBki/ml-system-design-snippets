# 🏢 Data Warehouse - ML System Design Note

> **An enterprise store for structured and semi-structured data from many sources, designed for analytics, reporting, and business intelligence.**

![Data Warehouse Architecture](assets/data-warehouse.png)

---

## 🎯 When to use

✅ **Many disparate sources** - need a single source of truth  
✅ **Ad-hoc SQL queries** - dashboards and governed data access  
✅ **Historical analysis** - trend analysis, KPI tracking, decision support  
✅ **Strong SLAs** - freshness, cost control, and security requirements  

---

## 🔄 Architecture Flow

```
📊 Sources → 📥 Ingestion → 🔄 Staging → ⚡ Transform → 🏢 Data Warehouse → 📈 Visualization
```

Data flows from various sources through ingestion pipelines (batch or streaming), gets staged, transformed via ETL/ELT processes, stored in a structured warehouse with facts and dimensions, then consumed by BI tools.

---

## ⚙️ Core Design Decisions

### 🔀 ETL vs ELT
| Approach | When to Use | Benefits |
|----------|-------------|----------|
| **ELT** | Modern, scalable systems | Load first, transform in-warehouse. More flexible |
| **ETL** | Strict governance needs | Transform before loading. Better control |

### 📐 Data Modeling
- **⭐ Star Schema** - Denormalized, faster for BI queries
- **❄️ Snowflake Schema** - Normalized, saves storage but more complex joins
- **🔄 Slowly Changing Dimensions** - Track history with SCD Type-1 (overwrite) or Type-2 (versioning)

### ⏱️ Data Freshness
- **🔄 Batch Processing** - Hourly/daily updates, simpler to manage
- **🌊 Stream Processing** - Near real-time, handles late arrivals and duplicates
- **⚡ Watermarks** - Plan for out-of-order data

### 🚀 Performance Optimization
- **📂 Partitioning** - By `event_date` or `load_date` for time-based queries
- **🎯 Clustering** - By high-cardinality columns like `customer_id`, `country`
- **💾 Storage Format** - Columnar formats (Parquet, ORC) with compression
- **📊 Materialized Views** - Pre-compute heavy aggregations

---

## 🛡️ Data Quality & Governance

### 🔍 Testing & Contracts
- ✅ Define data contracts between teams
- ✅ Implement data quality tests (dbt tests, Great Expectations)
- ✅ Track data lineage for impact analysis

### 🔒 Security & Compliance
- 🔐 **PII Handling** - Masking or tokenization
- 👥 **Access Control** - Role-based permissions
- 🛡️ **Encryption** - At rest and in transit

### 📚 Metadata Management
- 📋 Document data owners and SLAs
- 📊 Track freshness and cost metrics
- 📖 Maintain business glossary

---

## 💰 Performance & Cost Optimization

### 🔍 Query Optimization
- ✅ Use denormalized read models for BI workloads
- ✅ Pre-aggregate frequently queried data
- ✅ Always include partition filters in queries
- ❌ Avoid `SELECT *` in production

### 🎛️ Resource Management
- 💾 Implement result caching where possible
- ⏰ Use time travel features sparingly
- 🧹 Schedule automatic maintenance (vacuum, optimize)
- 📊 Set up cost monitoring and alerts

---

## 🤖 ML-Specific Considerations

### 📊 Training Data Integrity
- ⏰ **Point-in-time correctness** - Avoid data leakage
- 📸 **Snapshots** - Raw and staging tables for reproducibility
- 📝 **Documentation** - Feature lineage and definitions

### 🔄 Training-Serving Alignment
- 🎯 Keep training and serving schemas aligned
- 📊 Monitor for training-serving skew
- 🏪 Consider feeding a feature store from the warehouse

---

## 🛠️ Technology Stack Options

### ☁️ Cloud Data Warehouses
- **BigQuery** (Google Cloud)
- **Snowflake** (Multi-cloud)
- **Redshift** (AWS)

### 🏠 Lakehouse Architecture
- **Storage**: S3, GCS, ADLS
- **Table Formats**: Delta Lake, Iceberg, Hudi
- **Query Engines**: Trino, Presto, Athena

### 🔧 Supporting Tools
| Category | Options |
|----------|---------|
| **Orchestration** | Airflow, Dagster |
| **Transformation** | dbt, Spark |
| **Ingestion** | Fivetran, Airbyte, Kafka |
| **BI Tools** | Looker, Tableau, Metabase, Superset |

---

## ✅ Implementation Checklist

**Before going to production:**

- [ ] 📋 Define data sources, SLAs, and contracts
- [ ] 🏗️ Build staging to dimensional model pipeline
- [ ] 📊 Choose partitioning and clustering strategy
- [ ] 🔍 Implement data quality tests and lineage tracking
- [ ] 🔒 Set up PII policies and access controls
- [ ] 💰 Configure cost guardrails and monitoring
- [ ] 🔄 Create backfill and recovery procedures

---

## 🏷️ Naming Conventions

Use consistent layer prefixes:
- `raw_*` - Source data
- `stg_*` - Staging/cleaning
- `dim_*` - Dimension tables
- `fct_*` - Fact tables

---

## 💻 Example Query Pattern

```sql
-- 🎯 Partition-pruned revenue analysis
SELECT 
    d.customer_id, 
    SUM(f.net_revenue) AS total_revenue
FROM fct_orders f
JOIN dim_customers d ON d.customer_key = f.customer_key
WHERE f.event_date BETWEEN DATE '2025-08-01' AND DATE '2025-08-31'
  AND d.country = 'Turkey'
GROUP BY d.customer_id
ORDER BY total_revenue DESC;
```

---

## 🎯 Key Takeaways

> **Data warehouses excel at structured analytics and reporting with strong governance.** They provide a reliable foundation for business intelligence and can support ML workflows when designed with proper time-based partitioning and data quality controls. 
> 
> **Choose your architecture based on scale, budget, and team expertise.**

---

## 📚 More Resources

**⭐ Star this repo if you found it helpful!**

*More ML System Design notes coming soon...*
