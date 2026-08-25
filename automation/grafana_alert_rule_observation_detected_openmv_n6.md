# 📡 Real-Time Telemetry & Telegram Alerting Pipeline

## Overview & Architecture
The telemetry pipeline continuously monitors live wildlife detections stored in the PostgreSQL database (Supabase) and dispatches instantaneous alerts to a dedicated Telegram Channel using Grafana's alerting engine.

- Live Automation & Alert Rule: https://idohasson.grafana.net/alerting/ffnefzquem9kwd/edit
- Data Source: Supabase PostgreSQL ("Observations" table)
- Target Notification Point: Telegram Bot API Channel

---

## Alert Rule SQL Implementation

The alerting engine periodically evaluates the following SQL query against the database:

SELECT 
  COUNT(*) AS alert_triggered,
  STRING_AGG(
    FORMAT(
      'Hi, on %s at %s a bird of species "%s" was spotted at the station located in %s. The observation included %s item(s)',
      TO_CHAR(eventtime, 'YYYY-MM-DD'),
      TO_CHAR(eventtime, 'HH24:MI'),
      "birds species",
      location,
      amount
    ),
    E'\n\n'
  ) AS telegram_message
FROM "Observations"
WHERE eventtime > (NOW() AT TIME ZONE 'Asia/Jerusalem') - INTERVAL '1 minutes'
HAVING COUNT(*) > 0;

---

## Technical Component Breakdown

### 1. Real-Time Time-Window Synchronization (WHERE Clause)
- Timezone Normalization: Explicitly converts the database system clock to the local Israel timezone (Asia/Jerusalem), ensuring exact synchronization with the hardware RTC on the OpenMV camera.
- Sliding Detection Window: Filters exclusively for records inserted within the last 60 seconds, matching the 1-minute evaluation cycle of the alerting engine to eliminate duplicate triggers.

### 2. Dynamic Notification Formatting & Aggregation (STRING_AGG & FORMAT)
- Automated Message Generation: Extracts and formats raw database values into a clear, natural-language notification containing:
  * Observation Date (YYYY-MM-DD) and Local Time (HH24:MI).
  * Classified Species Name ("birds species").
  * Physical Camera Deployment Site (location).
  * Total Target Count in Frame (amount).
- Batch Aggregation: If multiple birds are detected within the same 1-minute evaluation cycle, STRING_AGG aggregates all detections into a single message separated by double line breaks (\n\n), preventing Telegram API rate-limiting issues.

### 3. Firing Threshold & State Evaluation (HAVING Clause)
- Restricts output generation strictly to cycles where at least one valid observation was recorded.
- If no motion or detections occurred, the query returns 0 rows, keeping the alert state in Normal (Green) without consuming external API credits.

---

## Grafana Dispatcher Configuration Parameters

| Parameter | Configuration | Purpose |
| :--- | :--- | :--- |
| Evaluation Interval | 1m | Scans database every 60 seconds |
| Pending Period | 0s | Dispatches immediate alert without wait duration |
| Alert Condition | B: alert_triggered > 0 | Triggers alert when count is strictly positive |
| Contact Point | Telegram Bot Integration | Routes telegram_message directly to the channel |
