# Framingham — التنبؤ بمرض القلب التاجي خلال 10 سنوات

كل تجربة في ملف بايثون مستقل. كل ملف يطبع جداول نتائجه، يحفظها في `results/`، ويحفظ رسومه في `figures/`.

## التجهيز (مرة واحدة)

المشروع مجرَّب على Python 3.13. افتح PowerShell داخل هذا المجلد، أنشئ بيئة خاصة بالمشروع، ثم ثبّت المكتبات:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## التشغيل

افتح PowerShell داخل هذا المجلد ثم شغّل أي ملف ببايثون البيئة الخاصة بالمشروع:

```
.\.venv\Scripts\python.exe 01_visualization.py
```

الرسوم تظهر في نافذة، وأغلق النافذة ليكمل الملف. لتشغيل بدون نوافذ (حفظ الرسوم فقط):

```
$env:NO_SHOW = "1"
```

## الملفات بالترتيب

| الملف | الخطوة | الوقت التقريبي |
|---|---|---|
| `01_visualization.py` | 1. Visualize | 10 ث |
| `02_missing_data.py` | 2. Missing data: مقارنة 8 طرق | 40 ث |
| `03_distribution.py` | 3. Distribution: الانحراف والقيم الشاذة و6 تحويلات | 25 ث |
| `04_feature_scaling.py` | 4. Feature scaling: 4 طرق على 4 موديلات | 20 ث |
| `04b_onehot_encoding.py` | 4. One-Hot Encoding لعمود التعليم (من Lab 1) | 50 ث |
| `05a_regularization_ridge_lasso_elasticnet.py` | 5. Ridge و Lasso و Elastic Net | 30 ث |
| `05b_pca.py` | 5. PCA من 1 إلى 15 مكوّن | 15 ث |
| `05c_stepwise_aic_bic.py` | 5. اختيار الخصائص بالـ AIC و BIC | 25 ث |
| `05d_other_feature_selection.py` | 5. RFE و ANOVA و Mutual information و Random Forest | 60 ث |
| `05e_compare_reduction_methods.py` | 5. مقارنة كل الطرق (يحتاج 05a إلى 05d) | 10 ث |
| `06a_models_default.py` | 6. 17 موديل بالإعدادات الافتراضية | 2 د |
| `06b_hyperparameter_tuning.py` | 6. ضبط معاملات كل موديل | 2 د |
| `06c_threshold_tuning.py` | 6. ضبط عتبة القرار (يحتاج 06b) | 1 د |
| `06d_ensembles.py` | 6. Voting و Stacking، ومنها Stacking بطريقة Lab 2 (يحتاج 06b) | 4 د |
| `06e_tree_pruning.py` | 6. تقليم شجرة القرار: pre و post pruning (من Lecture 5) | 1 د |
| `06f_feature_engineering.py` | 6. تجربة هندسة الخصائص | 1.5 د |
| `06g_optimizers_sgd_newton.py` | 6. Gradient Descent و SGD و Newton (من Lecture 4 و Lab 2) | 20 ث |
| `06h_test_evaluation_all_models.py` | 6. نتائج كل الموديلات على الـ test (يحتاج 06b و 06c و 06d) | 1 د |
| `07_final_model.py` | **الكود النهائي** | 30 ث |

## تشغيل الكل وتوليد الصور

```
.\.venv\Scripts\python.exe run_all.py
```

يشغّل الملفات الـ 19 بالترتيب بدون نوافذ (حوالي 18 دقيقة) ويحفظ مخرجات كل ملف في `logs/`.

```
.\.venv\Scripts\python.exe make_screenshots.py
```

يحوّل المخرجات المحفوظة في `logs/` إلى صور بشكل نافذة terminal للتقرير والبرزنتيشن:

- `screenshots/full/`: صورة كاملة لمخرجات كل ملف.
- `screenshots/sections/`: صورة منفصلة لكل جدول أو خلاصة.

والرسوم البيانية نفسها في `figures/`.

ملفان مساعدان لا يُشغَّلان مباشرة:

- `common.py`: تحميل الداتا، التقسيم الثابت train/test، الـ cross-validation، وأدوات المعالجة.
- `models.py`: تعريف الـ 17 موديل ونطاقات البحث عن معاملاتها.

## قواعد سلامة النتائج

- لا تُستخدم أي موازنة للفئات (SMOTE أو غيرها) في أي مكان بالمشروع.
- التقسيم 80/20 ثابت (`random_state = 42`) ويتم في `common.py` قبل أي شيء.
- كل الملفات من 02 إلى 06g تستخدم cross-validation داخل الـ training فقط.
- الـ test لا يُستخدم إلا في `06h` و `07`.
- كل المعالجة (تعويض، scaling، PCA، اختيار خصائص) داخل Pipeline، فتتعلم من الـ training فقط.
