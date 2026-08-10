# Level 3 Task 2 — Dashboard Build Guide

**Important:** I can prep the data and tell you exactly what to build, but I can't generate an actual `.pbix`/`.twbx` file — Power BI and Tableau are desktop apps, not something I can output directly. This is the one task you'll need to physically build yourself (10-15 min if you follow this spec).

**File to import:** `churn_dashboard_data.csv`

## Build this (Power BI or Tableau — either satisfies the task)

1. **Import** `churn_dashboard_data.csv`
2. **KPI cards (top row):**
   - Total Customers (count of rows)
   - Churn Rate % (`Churn = Yes` count ÷ total)
   - Avg Customer Service Calls
3. **Bar chart:** Churn count by `State` (top 10 states by churn count)
4. **Bar chart:** Churn rate by `International plan` (Yes vs No) — this will show international plan customers churn at a much higher rate, a real, presentable insight
5. **Scatter or box plot:** `Total day minutes` vs `Churn` — day-usage is your #1 churn driver per the model, so this chart backs that up visually
6. **Filters/slicers:** `State`, `International plan`, `Voice mail plan`, `Customer service calls`
7. **Publish** and grab the share link (Power BI) or export as `.twbx`/PDF (Tableau) for your submission

## Talking point for your video
"I identified that customers with an international plan and high daily usage but low customer-service engagement churn at a much higher rate — the dashboard lets the business filter by state and plan type to see where retention campaigns should focus first."
