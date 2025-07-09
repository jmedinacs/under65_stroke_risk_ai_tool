-- =========================================================
-- This query is the entire process of preliminary cleaning and eda of the unprocessed dataset.
-- Here's the list of the actions done within this query:
-- 1. Filter the dataset to exclude patients 65 and over.
-- 2. Check and remove the single instance of 'other' gender.
-- 3. Create a column for heart_diease and hypertension status (see cleaning log for justification)
-- 4. Create view in preparation for python friendly table export
-- =========================================================

-- =========================================================
-- Step 1:
-- Create a copy of the original unprocessed data
-- Filter for patients under the age of 65
-- =========================================================
CREATE TABLE under65_ai_stroke_data AS --copy the data from the original unprocessed dataset
SELECT *
FROM stroke_data
WHERE age < 65; -- filter under 65 patients

-- check filtered table data count
SELECT count(*)
FROM under65_ai_stroke_data
LIMIT 50;

-- =========================================================
-- Step 2:
-- Check and remove the single instance of 'other' gender.
-- =========================================================
SELECT *
FROM under65_ai_stroke_data
WHERE gender = 'other'
-- RESULT = none (already filtered out)


-- =========================================================
-- Step 3:
-- Create a column for heart_diease and hypertension status
-- Since these features are binary and numeric data type, one-hot coding may fail during modeling.
-- =========================================================
ALTER TABLE under65_ai_stroke_data
ADD COLUMN hypertension_status TEXT,
ADD COLUMN heart_disease_status TEXT;

UPDATE under65_ai_stroke_data
SET hypertension_status = CASE
	WHEN hypertension = 0 THEN 'no_hypertension'
	ELSE 'hypertension'
END;

UPDATE under65_ai_stroke_data
SET heart_disease_status = CASE
	WHEN heart_disease = 0 THEN 'no_heart_disease'
	ELSE 'heart_disease'
END;


-- =========================================================
-- Step 4:
-- Create view in preparation for python friendly table export
-- =========================================================
CREATE VIEW sql_export_under65_clean AS
SELECT
	id,
	age,
	gender,
	ever_married,
	work_type,
	residence_type,
	avg_glucose_level,
	bmi, 
	smoking_status,
	heart_disease_status AS heart_disease,
	hypertension_status AS hypertension,
	stroke
FROM 
	under65_ai_stroke_data;

SELECT * FROM sql_export_under65_clean
