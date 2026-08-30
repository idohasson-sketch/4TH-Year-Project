# 📊 Grafana Telemetry Dashboard — SQL Analytics & Documentation

## Overview
This document compiles and details all production SQL queries powering the A-EYE Tracker analytical dashboards on Grafana. The queries evaluate real-time and historical telemetry data streamed from the edge hardware (OpenMV N6) into the Supabase PostgreSQL database ("Observations" table).

---

## 1. Top KPI Metric Panels

### 1.1 Total Reports (Time-Filtered)
Calculates the total number of hardware observation events logged within the selected dashboard timeframe.

SELECT COUNT(*) AS "Total Reports"
FROM "Observations"
WHERE ("eventtime" AT TIME ZONE 'Asia/Jerusalem') BETWEEN $__timeFrom() AND $__timeTo();

### 1.2 Total Bird Individuals Sighted
Calculates the aggregate count of all bird individuals across all recorded events within the timeframe.

SELECT SUM(amount) AS "Total Birds"
FROM "Observations"
WHERE ("eventtime" AT TIME ZONE 'Asia/Jerusalem') BETWEEN $__timeFrom() AND $__timeTo();

### 1.3 Misclassified Observations Count
Aggregates the number of misclassified reports (is_correct = FALSE) within the timeframe for edge model performance monitoring.

SELECT SUM(amount) AS "Misclassified Reports"
FROM "Observations"
WHERE is_correct = FALSE
AND ("eventtime" AT TIME ZONE 'Asia/Jerusalem') BETWEEN $__timeFrom() AND $__timeTo();

---

## 2. Species Breakdown & Multi-Class Distribution

### 2.1 Full 5-Class Species Distribution (Single Row KPI / Bar Gauge)
Breaks down the absolute count across all 5 active deployment categories.

SELECT 
  'Total Count' AS metric,
  SUM(CASE WHEN "birds species" = 'House Sparrow' THEN amount ELSE 0 END) AS "House Sparrow",
  SUM(CASE WHEN "birds species" = 'Feral Pigeon' THEN amount ELSE 0 END) AS "Feral Pigeon",
  SUM(CASE WHEN "birds species" = 'Rose ringed Parakeet' THEN amount ELSE 0 END) AS "Rose ringed Parakeet",
  SUM(CASE WHEN "birds species" = 'Hooded Crow' THEN amount ELSE 0 END) AS "Hooded Crow",
  SUM(CASE WHEN "birds species" = 'Other' THEN amount ELSE 0 END) AS "Other"
FROM "Observations"
WHERE ("eventtime" AT TIME ZONE 'Asia/Jerusalem') BETWEEN $__timeFrom() AND $__timeTo();

### 2.2 Correctly Classified Individuals by Species
Lists total verified sightings per species sorted in descending volume.

SELECT
  "birds species" AS bird_species,
  SUM(amount) AS total_healthy_birds
FROM "Observations"
WHERE is_correct = TRUE
GROUP BY "birds species"
ORDER BY total_healthy_birds DESC;

### 2.3 Misclassified Individuals by Species
Identifies which species experience higher misclassification rates to guide future fine-tuning.

SELECT
  "birds species" AS bird_species,
  SUM(amount) AS total_injured_birds
FROM "Observations"
WHERE is_correct = FALSE
GROUP BY "birds species"
ORDER BY total_injured_birds DESC;

---

## 3. Temporal & Diurnal Trends

### 3.1 24-Hour Diurnal Activity Distribution (Heatmap / Bar Chart)
Extracts the hour of day from the timestamp to visualize peak activity hours in the monitored habitat.
'''
SELECT
  EXTRACT(HOUR FROM eventtime)::int AS "Hour of Day",
  COUNT(*) AS "Observations"
FROM "Observations"
GROUP BY 1
ORDER BY 1;

### 3.2 Weekly Ingestion & Individual Count Trend
Aggregates reports and total birds week-over-week to monitor seasonal activity trends.

SELECT
  date_trunc('week', eventtime)::timestamptz AS "time",
  COUNT(*) AS "Total Reports",
  SUM(amount) AS "Total Birds"
FROM "Observations"
GROUP BY 1
ORDER BY 1;

### 3.3 Weekly Model Accuracy Rate (%)
Tracks model classification accuracy versus misclassification percentage over time.

SELECT
  date_trunc('week', eventtime)::timestamptz AS "time",
  ROUND(100.0 * SUM(CASE WHEN is_correct = TRUE THEN 1 ELSE 0 END) / COUNT(*), 1) AS "Classified %",
  ROUND(100.0 * SUM(CASE WHEN is_correct = FALSE THEN 1 ELSE 0 END) / COUNT(*), 1) AS "Misclassified %"
FROM "Observations"
GROUP BY 1
ORDER BY 1;

---

## 4. Live Logs & Advanced Behavioral Analytics

### 4.1 Last 24 Hours Activity Log Table
Displays the latest incoming telemetry events in real time.

SELECT
  eventtime AS "Time",
  "birds species" AS "Species",
  amount AS "Birds",
  location AS "Location",
  CASE WHEN is_correct = TRUE THEN 'Classified' ELSE 'Misclassified' END AS "Status"
FROM "Observations"
WHERE eventtime >= NOW() - INTERVAL '24 hours'
ORDER BY eventtime DESC;

### 4.2 Overall Global Classification Ratio
Provides the aggregate status distribution for pie charts and status gauges.

SELECT 
  CASE 
    WHEN is_correct = TRUE THEN 'Classified'
    WHEN is_correct = FALSE THEN 'Misclassified'
    ELSE 'Unknown'
  END AS is_correct,
  COUNT(*) AS total_reports
FROM "Observations"
GROUP BY is_correct;

### 4.3 Peak Sightings & Optimization Recommendation Engine
Analyzes hourly volume per species and generates automated observation recommendations.

WITH hourly_counts AS (
  SELECT
    "birds species" AS species,
    EXTRACT(HOUR FROM eventtime)::int AS hour,
    COUNT(*) AS cnt
  FROM "Observations"
  GROUP BY 1, 2
),
species_best AS (
  SELECT DISTINCT ON (species)
    species, hour AS best_hour, cnt AS best_cnt
  FROM hourly_counts
  ORDER BY species, cnt DESC
),
species_worst AS (
  SELECT DISTINCT ON (species)
    species, hour AS worst_hour, cnt AS worst_cnt
  FROM hourly_counts
  ORDER BY species, cnt ASC
),
all_hourly AS (
  SELECT
    EXTRACT(HOUR FROM eventtime)::int AS hour,
    COUNT(*) AS cnt
  FROM "Observations"
  GROUP BY 1
),
all_best AS (
  SELECT hour AS best_hour, cnt AS best_cnt FROM all_hourly ORDER BY cnt DESC LIMIT 1
),
all_worst AS (
  SELECT hour AS worst_hour, cnt AS worst_cnt FROM all_hourly ORDER BY cnt ASC LIMIT 1
)
SELECT
  sb.species AS "Species",
  LPAD(sb.best_hour::text, 2, '0') || ':00' AS "Best Time",
  sb.best_cnt AS "Peak Sightings",
  LPAD(sw.worst_hour::text, 2, '0') || ':00' AS "Worst Time",
  sw.worst_cnt AS "Low Sightings",
  'Head out at ' || LPAD(sb.best_hour::text, 2, '0') || ':00 for up to ' || sb.best_cnt || ' sightings. Avoid ' || LPAD(sw.worst_hour::text, 2, '0') || ':00 — only ' || sw.worst_cnt || ' sightings recorded.' AS "Recommendation"
FROM species_best sb
JOIN species_worst sw ON sb.species = sw.species
UNION ALL
SELECT
  '★ All Species' AS "Species",
  LPAD(ab.best_hour::text, 2, '0') || ':00' AS "Best Time",
  ab.best_cnt AS "Peak Sightings",
  LPAD(aw.worst_hour::text, 2, '0') || ':00' AS "Worst Time",
  aw.worst_cnt AS "Low Sightings",
  'Overall peak hour is ' || LPAD(ab.best_hour::text, 2, '0') || ':00 (' || ab.best_cnt || ' sightings). Quietest at ' || LPAD(aw.worst_hour::text, 2, '0') || ':00 (' || aw.worst_cnt || ' sightings).' AS "Recommendation"
FROM all_best ab, all_worst aw
ORDER BY "Peak Sightings" DESC;
