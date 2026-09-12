# Data Dictionary

## Dataset
**Dataset name:** StressLevelDataset.csv

**Source:** https://raw.githubusercontent.com/mibur1/psy111/main/book/statistics/4_Moderated_Reg/data/StressLevelDataset.csv

**Purpose:** The dataset contains student-related academic, psychological, lifestyle, and social factors used to study the relationship between student well-being and academic performance.

## Variables
| Variable | Description |
|---|---|
| anxiety_level | Level of anxiety reported by the student |
| self_esteem | Student's self-esteem level |
| mental_health_history | Indicator of previous mental health history |
| depression | Depression level reported by the student |
| headache | Indicator of headache-related symptoms |
| blood_pressure | Blood pressure category/measurement |
| sleep_quality | Quality of the student's sleep |
| breathing_problem | Indicator of breathing-related problems |
| noise_level | Noise level experienced by the student |
| living_conditions | Quality of the student's living conditions |
| safety | Perceived level of safety |
| basic_needs | Whether basic needs are adequately met |
| academic_performance | Academic performance score |
| study_load | Student's study/workload level |
| teacher_student_relationship | Quality of the teacher-student relationship |
| future_career_concerns | Level of concern about future career |
| social_support | Level of social support |
| peer_pressure | Level of peer pressure |
| extracurricular_activities | Participation in extracurricular activities |
| bullying | Indicator/level of bullying experienced |
| stress_level | Student's reported stress level |

## Variables Used in the Final Model

The final classification model uses the following predictors:

- `stress_level`
- `mental_health_score`
- `sleep_quality`
- `social_support`
- `self_esteem`

`mental_health_score` is derived from `depression` and `anxiety_level`:

`mental_health_score = (depression + anxiety_level) / 2`

The target variable is:

`low_performance`

Academic performance values less than or equal to 2 are coded as Low Performance (1), while the remaining values are coded as High Performance (0).

## Data Preparation
The project removes `blood_pressure` and `headache`, checks for missing values and duplicates, fills numerical missing values using the median, removes duplicate rows, and resets the index. The original `academic_performance` variable is removed after creating the binary target.

## Source and License

The dataset was retrieved from the public source URL shown above for this academic project. The provided source materials do not specify a separate license or permission statement, so no specific license is claimed here.

Users of the dataset should verify the original source's terms and permissions before redistributing the data outside this academic submission.
