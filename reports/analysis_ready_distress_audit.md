# Analysis Ready Distress Audit

## Workbook overview

- Sheets: Table Reference, Field Reference, Codes Reference, ANALYSIS_DIS_AC, ANALYSIS_DIS_CRCP, ANALYSIS_DIS_JPCC, EXPERIMENT_SECTION, SHRP_INFO, TST_L05B, PERFORMANCE_EVENT

## Table Reference

- Rows: 6

- Columns: 4

- First columns: TABLE_NAME, TABLE_ALIAS, TABLE_DESCRIPTION, CLASS_NAME

- Columns with >=50% missing: 0


## Field Reference

- Rows: 511

- Columns: 7

- First columns: TABLE_NAME, FIELD_NAME, FIELD_ALIAS, FIELD_DESCRIPTION, FIELD_UNIT, FIELD_CODETYPE, UNIT_SYSTEM

- Columns with >=50% missing: 3


## Codes Reference

- Rows: 466

- Columns: 5

- First columns: CODETYPE, CODE, DETAIL, ADDL_CODE, ADDL_CODETYPE

- Columns with >=50% missing: 2


## ANALYSIS_DIS_AC

- Rows: 13,097

- Columns: 137

- Alias: AC Pavement Distress

- Description: Distress information for AC pavement. Includes flags for distress data that were anomalous.

- First columns: STATE_CODE, STATE_CODE_EXP, SHRP_ID, CONSTRUCTION_NO, SURVEY_DATE, HPMS16_CRACKING_PERCENT_AC, MEPDG_CRACKING_PERCENT_AC, MEPDG_TRANS_CRACK_LENGTH_AC, MEPDG_LONG_CRACK_LENGTH_AC, ME_PERCENT_WHEEL_PATH_CRACK, SURVEY_WIDTH, GATOR_CRACK_A, GATOR_CRACK_A_L, GATOR_CRACK_A_M, GATOR_CRACK_A_H ... (+122 more)

- Survey date range: 1988-08-05 to 2024-05-22

- Unique sections (STATE_CODE+SHRP_ID): 1,807

- Columns with >=50% missing: 57


## ANALYSIS_DIS_CRCP

- Rows: 544

- Columns: 123

- Alias: CRCP Pavement Distress

- Description: Distress information for CRCP pavement. Includes flags for distress data that were anomalous.

- First columns: SHRP_ID, STATE_CODE, STATE_CODE_EXP, SURVEY_DATE, CONSTRUCTION_NO, HPMS16_CRACKING_PERCENT_CRCP, MEPDG_PUNCHOUTS_CRCP, SURVEY_WIDTH, DURAB_CRACK_A, DURAB_CRACK_A_L, DURAB_CRACK_A_M, DURAB_CRACK_A_H, DURAB_CRACK_NO, DURAB_CRACK_NO_L, DURAB_CRACK_NO_M ... (+108 more)

- Survey date range: 1989-07-07 to 2022-08-10

- Unique sections (STATE_CODE+SHRP_ID): 108

- Columns with >=50% missing: 51


## ANALYSIS_DIS_JPCC

- Rows: 5,750

- Columns: 164

- Alias: JPCC Pavement Distress

- Description: Distress information for JPCC pavement. Includes flags for distress data that were anomalous. 

- First columns: SHRP_ID, STATE_CODE, STATE_CODE_EXP, SURVEY_DATE, CONSTRUCTION_NO, HPMS16_CRACKING_PERCENT_JPCC, MEPDG_CRACKING_PERCENT_JPCC, ME_PERCENT_CRACKED_SLABS, SURVEY_WIDTH, JT_SEALED_EXP, JT_SEALED, TRANS_CRACK_L, TRANS_CRACK_L_L, TRANS_CRACK_L_M, TRANS_CRACK_L_H ... (+149 more)

- Survey date range: 1989-08-14 to 2024-04-29

- Unique sections (STATE_CODE+SHRP_ID): 794

- Columns with >=50% missing: 71


## EXPERIMENT_SECTION

- Rows: 7,934

- Columns: 25

- Alias: Experiment Section

- Description: Contains the time history of changes in construction numbers, experiment designations, and reasons for these changes.

- First columns: STATE_CODE, STATE_CODE_EXP, SHRP_ID, CONSTRUCTION_NO, CN_ASSIGN_DATE, CN_CHANGE_REASON, CN_CHANGE_REASON_EXP, RECORD_STATUS, GPS_SPS, GPS_SPS_EXP, EXPERIMENT_NO, EXPERIMENT_NO_EXP, STATUS_EXP, STATUS, ASSIGN_DATE ... (+10 more)

- Unique sections (STATE_CODE+SHRP_ID): 2,829

- Columns with >=50% missing: 2


## SHRP_INFO

- Rows: 4,588

- Columns: 39

- Alias: LTPP Traffic Site Information

- Description: Data describing the traffic data relationships and site conditions for a given SPS project or GPS Site.

- First columns: STATE_CODE, STATE_CODE_EXP, SHRP_ID, START_DATE, RECORD_STATUS, END_DATE, VOLUME_SITE, CLASS_SITE, WIM_SITE, ID3, ID6, LTPP_DIR, LTPP_DIR_EXP, LTPP_LANE, LANES_LTPP_DIR ... (+24 more)

- Unique sections (STATE_CODE+SHRP_ID): 2,523

- Columns with >=50% missing: 1


## TST_L05B

- Rows: 38,960

- Columns: 23

- Alias: Material Characterization and Thickness Data

- Description: Table containing layer descriptions for all constructions.

- First columns: STATE_CODE, STATE_CODE_EXP, SHRP_ID, CONSTRUCTION_NO, LAYER_NO, PROJECT_LAYER_CODE, DESCRIPTION, DESCRIPTION_EXP, LAYER_TYPE, LAYER_TYPE_EXP, REPR_THICKNESS, MATL_CODE_EXP, MATL_CODE, LAYER_COMMENT1, LAYER_COMMENT1_EXP ... (+8 more)

- Unique sections (STATE_CODE+SHRP_ID): 2,581

- Columns with >=50% missing: 8


## PERFORMANCE_EVENT

- Rows: 46

- Columns: 6

- First columns: LTPP Change Events with Potential of Affecting Pavement Condition Measurements, Unnamed: 1, Unnamed: 2, Unnamed: 3, Unnamed: 4, Unnamed: 5

- Columns with >=50% missing: 0


## AC target audit

### HPMS16_CRACKING_PERCENT_AC

- Alias: 2016 HPMS AC Cracking Percentage

- Unit: %

- Observed rows: 13,065 / 13,097 (99.8%)

- Zero share among observed rows: 45.6%

- Mean / median: 9.566 / 1.000

- P90 / P99 / max: 37.000 / 54.000 / 65.000

- Coverage: 1,806 sections across 37 years

- Description: Percent of section cracked using 2016 HPMS Field Guide definitions. Includes only longitudinal and fatigue cracking in assumed 1m wide wheel paths.


### MEPDG_CRACKING_PERCENT_AC

- Alias: MEPDG AC Cracking Percentage

- Unit: %

- Observed rows: 13,076 / 13,097 (99.8%)

- Zero share among observed rows: 64.5%

- Mean / median: 5.039 / 0.000

- P90 / P99 / max: 17.000 / 63.000 / 100.000

- Coverage: 1,806 sections across 37 years

- Description: The total area of alligator cracking summed across all levels of severity, divided by the total area of the test section, in accordance with MEPDG definitions. Note that LTPP alligator cracking interpretations are not restricted to the wheel path.


### MEPDG_TRANS_CRACK_LENGTH_AC

- Alias: MEPDG AC Transverse Cracking Length

- Unit: ft/mi

- Observed rows: 12,927 / 13,097 (98.7%)

- Zero share among observed rows: 40.4%

- Mean / median: 1073.787 / 156.000

- P90 / P99 / max: 3333.000 / 7831.540 / 18522.000

- Coverage: 1,804 sections across 37 years

- Description: Total length of sealed and unsealed transverse cracks at all levels of severity, divided by test section length in accordance with current pavement MEPDG distress definitions.


### PATCH_A

- Alias: Patches Area Total

- Unit: sq m

- Observed rows: 13,094 / 13,097 (100.0%)

- Zero share among observed rows: 88.4%

- Mean / median: 4.798 / 0.000

- P90 / P99 / max: 0.300 / 153.554 / 530.400

- Coverage: 1,807 sections across 37 years

- Description: Total area of patching.


### POTHOLES_A

- Alias: Potholes Area Total

- Unit: sq m

- Observed rows: 13,095 / 13,097 (100.0%)

- Zero share among observed rows: 97.6%

- Mean / median: 0.007 / 0.000

- P90 / P99 / max: 0.000 / 0.200 / 7.150

- Coverage: 1,807 sections across 37 years

- Description: Total pothole area.


## What the workbook gives you for prediction

- **Targets**: 5 main AC distress outcomes already present in one analysis-ready sheet.

- **Close lag predictors inside the same sheet**: many distress sub-components and severity splits (e.g. low/medium/high cracking, sealed vs unsealed, counts vs areas).

- **Quality controls**: many `_FLAG` and `_FLAG_EXP` fields identifying anomalous or qualified measurements.

- **Structure / treatment context from other sheets**: `EXPERIMENT_SECTION`, `TST_L05B`, `SHRP_INFO`.

- **Event timing context**: `PERFORMANCE_EVENT` is tiny and looks like a manual reference sheet, not a large modelling table.

## Modelling implications

- Do **not** treat all 137 AC columns as equal predictors: many are near-duplicates, severity decompositions, or quality flags.

- For longitudinal prediction, the strongest families are likely: lagged distress levels, distress composition, treatment history, pavement structure, and traffic-site characteristics.

- Heavy-tailed sparse targets (`PATCH_A`, `POTHOLES_A`, partly transverse cracking) need transformation or two-stage modelling because they are dominated by zeros and rare extremes.

- Flags should usually be used either as quality filters or explicit reliability indicators, not mixed blindly with physical predictors.
