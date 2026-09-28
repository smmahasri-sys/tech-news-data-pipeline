-- Latest ARR by company
WITH ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY company_id
           ORDER BY published_date_normalized DESC, article_id DESC
         ) AS rn
  FROM fct_arr_observation
)
SELECT company_id, company_name, published_date_normalized, arr_usd
FROM ranked
WHERE rn = 1;

-- Quarterly ARR observations
SELECT company_id, year, quarter, MAX(arr_usd) AS quarterly_arr_usd
FROM fct_arr_observation
GROUP BY company_id, year, quarter
ORDER BY company_id, year, quarter;

-- Find the source article for an ARR observation
SELECT a.article_id, a.title, a.url, o.arr_usd, o.published_date_normalized
FROM fct_arr_observation o
JOIN fct_article a USING (article_id);


-- The actual point of an ARR-observations-over-time table is tracking revenue trend, not just storing points. 
 --Write a query that, per company, computes the ARR delta 
 --between each observation and the prior one, and labels each delta as expansion or contraction.

 select company_id, 
       published_date_normalized, 
       arr_usd, 
       lag(arr_usd) over (partition by company_id order by published_date_normalized) as previous_arr_usd,
       arr_usd - lag(arr_usd) over (partition by company_id order by published_date_normalized) as arr_delta,
       case 
           when arr_usd - lag(arr_usd) over (partition by company_id order by published_date_normalized) > 0 then 'expansion'
           when arr_usd - lag(arr_usd) over (partition by company_id order by published_date_normalized) < 0 then 'contraction'
           else 'no change'
       end as delta_type
from fct_arr_observation
order by company_id, published_date_normalized;