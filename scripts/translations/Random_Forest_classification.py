# -*- coding: utf-8 -*-
"""Arabic translations for Random_Forest_classification.ipynb — keyed by cell index."""

CELLS = {
    0: """# تصنيف الغابة العشوائية — Google Colab

**الهدف:** التنبؤ بـ **فئة ثنائية** (0/1) باستخدام **تصنيف الغابة العشوائية** — **مجموعة** من أشجار القرار **تصوّت** لتقليل الإفراط في التلائم وتحسين الدقة.

| المثال | المتغير(ات) المستقل(ة) (X) | المتغير المستهدف (y) | مجموعة البيانات |
|---------|----------------|------------|---------|
| **المثال 1** | Age, EstimatedSalary | Purchased (0/1) | `../Datasets/social_network_ads.csv` |
| **المثال 2** | CreditScore, Income, LoanAmount, YearsEmployed | Approved (0/1) | `../Datasets/loan_approval.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | التجميع، العيّنة الذاتية، التصويت بالأغلبية |
| المرحلة 1 | معالجة البيانات مسبقاً | تحميل → تنظيف → ترميز → تقسيم |
| المرحلة 2 | الخوارزمية | تدريب → تنبؤ → تصوير → تقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)""",

    1: """---
# دليل الخوارزمية — تصنيف الغابة العشوائية

## ما هي الغابة العشوائية؟

**الغابة العشوائية** = **عدة أشجار قرار** تُدرَّب على **مجموعات فرعية عشوائية مختلفة** من البيانات والميزات، ثم تُجمَع تنبؤاتها بالفئات عبر **التصويت بالأغلبية**.

إنها طريقة **تجميع Bagging** (Bootstrap Aggregating) اخترعها Leo Breiman (2001).

## كيف تعمل (خطوة بخطوة)

| الخطوة | الاسم | ماذا يحدث |
|------|------|--------------|
| 1 | **عيّنة Bootstrap** | سحب n صف **مع الاستبدال** من بيانات التدريب |
| 2 | **ميزات عشوائية** | عند كل تقسيم، يُنظر فقط إلى **مجموعة فرعية عشوائية** من الميزات |
| 3 | **نمو الشجرة** | بناء شجرة CART للتصنيف على تلك العيّنة الذاتية |
| 4 | **التكرار** | إنشاء `n_estimators` شجرة (مثلاً 100) |
| 5 | **التجميع** | الفئة النهائية = **التصويت بالأغلبية** عبر جميع الأشجار |

## قاعدة التنبؤ (التصنيف)

لعيّنة جديدة **x**:

1. كل شجرة **b** تتنبأ بفئة: `ŷ_b(x) ∈ {0, 1}`
2. التنبؤ النهائي = الفئة ذات **أكثر الأصوات**
3. `predict_proba()` = نسبة الأشجار التي صوّتت لكل فئة

**مثال:** 100 شجرة → 72 تصوّت «نعم»، 28 تصوّت «لا» → التنبؤ **نعم**، P(نعم) = 0.72

## المعاملات الفائقة الرئيسية

| المعامل | الدور | التأثير |
|-----------|------|--------|
| `n_estimators` | عدد الأشجار في الغابة | المزيد من الأشجار → استقرار أكبر (عائد متناقص) |
| `max_depth` | أقصى عمق لكل شجرة | يحدّ من الإفراط في التلائم لكل شجرة |
| `min_samples_split` | أقل عدد عيّنات لتقسيم عقدة | أعلى → أشجار أبسط |
| `min_samples_leaf` | أقل عدد عيّنات في ورقة | أعلى → قرارات أنعم |
| `max_features` | الميزات المُراعاة عند كل تقسيم | `'sqrt'` يضيف تنوعاً بين الأشجار |
| `bootstrap` | استخدام عيّنة Bootstrap | `True` (افتراضي) — جوهر التجميع |
| `random_state` | البذرة العشوائية | غابة قابلة للتكرار |

## الغابة العشوائية مقابل شجرة قرار واحدة (CART)

| | CART واحدة | الغابة العشوائية |
|---|-------------|---------------|
| الأشجار | شجرة **1** | **عدة** أشجار (مجموعة) |
| الإفراط في التلائم | خطر **مرتفع** إذا كانت عميقة | **أقل** — التصويت يقلّل التباين |
| الحدود | خطوات محاذية للمحاور | مناطق **أنعم** بالتصويت |
| قابلية التفسير | سهلة (`plot_tree`) | استخدم **أهمية الميزات** |
| التحجيم | غير مطلوب | **غير مطلوب** |
| السرعة | تدريب/تنبؤ سريع | أبطأ (تدريب B شجرة) |

## أهمية الميزات

تحسب الغابة العشوائية متوسط مقدار تقليل كل ميزة **لـ Gini impurity** عبر جميع التقسيمات في جميع الأشجار. قيمة أعلى = تأثير أكبر على التصنيف.

## ما يجب أن يتذكره الطالب

1. الغابة العشوائية = **Bagging + مجموعات فرعية عشوائية للميزات** عند كل تقسيم.
2. الفئة النهائية = **التصويت بالأغلبية** لجميع الأشجار (تصنيف).
3. **لا حاجة لتحجيم الميزات** — كما في أشجار القرار.
4. المزيد من الأشجار (`n_estimators`) عادةً يساعد حتى تستقر الأداء.
5. استخدم **أهمية الميزات** عندما تكون الغابة كبيرة جداً للتصوير.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبيت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات للمعالجة المسبقة وتصنيف الغابة العشوائية والتقييم.

**ما تفعله هذه الخلية:** تحميل numpy و pandas و matplotlib و RandomForestClassifier من sklearn والمقاييس.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء الرسوم البيانية
import seaborn as sns           # تصورات إحصائية (خرائط حرارية)

from sklearn.model_selection import train_test_split       # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.ensemble import RandomForestClassifier        # مصنّف الغابة العشوائية التجميعي
from sklearn.metrics import (                              # مقاييس التصنيف
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

plt.rcParams['figure.figsize'] = (10, 6)  # حجم الرسم الافتراضي: عرض=10، ارتفاع=6 بوصة
plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لتقسيمات قابلة للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح""",

    6: """---
# المثال 1: إعلانات الشبكات الاجتماعية — تصنيف الغابة العشوائية

التنبؤ بما إذا **اشترى** المستخدم من **Age** و **EstimatedSalary** باستخدام الغابة العشوائية.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Age` | ميزة (X₁) | عمر المستخدم بالسنوات |
| `EstimatedSalary` | ميزة (X₂) | الراتب السنوي التقديري بالدولار |
| `Purchased` | متغير مستهدف (y) | 0 = لا، 1 = نعم |

**الملف:** `../Datasets/social_network_ads.csv`""",

    7: """---
# المرحلة 1: معالجة البيانات مسبقاً

تحضير البيانات قبل التدريب — نفس القالب يُعاد استخدامه لخوارزميات أخرى.""",

    8: """## المثال 1 — الخلية 1: تحميل واستكشاف البيانات

تحميل ملف CSV وإجراء استكشاف أولي (head, info, describe, shape).

**ما تفعله هذه الخلية:** قراءة `../Datasets/social_network_ads.csv` وعرض إحصائيات أساسية.""",

    9: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/social_network_ads.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['Age', 'EstimatedSalary']  # ميزتان عدديتان للإدخال
TARGET_COL = 'Purchased'                    # هدف ثنائي: 0 = لا، 1 = نعم

print('أول 5 صفوف:')
display(dataset.head())

print('\\nمعلومات مجموعة البيانات:')
dataset.info()

print('\\nملخص إحصائي:')
display(dataset.describe())

print(f'\\nتوازن الفئات ({TARGET_COL}):')
print(dataset[TARGET_COL].value_counts())

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')""",

    10: """## المثال 1 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

التحقق من القيم المفقودة وإزالة التكرارات وتطبيق الاستبدال عند الحاجة.

**ما تفعله هذه الخلية:** تنظيف مجموعة البيانات قبل النمذجة.""",

    11: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')
print(dataset.isnull().sum())

rows_before = len(dataset)
dataset = dataset.drop_duplicates().reset_index(drop=True)
rows_after = len(dataset)
print(f'\\nالتكرارات المُزالة: {rows_before - rows_after}')

num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
if dataset.isnull().sum().sum() > 0:
    dataset[num_cols] = imputer.fit_transform(dataset[num_cols])
    print('تم استبدال القيم المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُستبدِل')

print(f'\\nعدد الصفوف بعد التنظيف: {rows_after}')""",

    12: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة عددية — يُتخطَّى الترميز.

**ما تفعله هذه الخلية:** يؤكد عدم الحاجة إلى ترميز فئوي.""",

    13: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات: {FEATURE_COLS}')
    print(f'عمود الهدف: {TARGET_COL} (0/1 مسبقاً)')""",

    14: """## المثال 1 — الخلية 4: تقسيم البيانات

تعريف X و y، ثم تقسيم 80/20 مع **stratify** للحفاظ على نسبة الفئات.

**ما تفعله هذه الخلية:** إنشاء مصفوفات الميزات/الهدف وتطبيق train_test_split الطبقي.""",

    15: """# الخطوة 4) تقسيم التدريب-الاختبار (طبقي للتصنيف)

X = dataset[FEATURE_COLS].values  # مصفوفة الميزات: Age و EstimatedSalary
y = dataset[TARGET_COL].values    # متجه الهدف: Purchased (0 أو 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'شكل X_train: {X_train.shape}')
print(f'شكل X_test:  {X_test.shape}')
print(f'عدد فئات y_train: {np.bincount(y_train)}')
print(f'عدد فئات y_test:  {np.bincount(y_test)}')""",

    16: """> **ملاحظة:** تصنيف الغابة العشوائية **لا يتطلب** تحجيم الميزات. الأشجار تقسم عند عتبات — المقياس لا يهم.""",

    17: """---
# المرحلة 2: تصنيف الغابة العشوائية — إعلانات الشبكات الاجتماعية

تدريب غابة من الأشجار والتقييم بالتصويت بالأغلبية.""",

    18: """## المثال 1 — الخلية 5: تدريب النموذج

تدريب `RandomForestClassifier` بـ `n_estimators=100` شجرة.

**ما تفعله هذه الخلية:** يلائم الغابة ويطبع عدد الأشجار.""",

    19: """# الخطوة 5) تدريب مصنّف الغابة العشوائية

classifier = RandomForestClassifier(
    n_estimators=100,      # عدد الأشجار في الغابة
    max_depth=8,           # تحديد العمق لكل شجرة
    min_samples_leaf=3,    # أقل عدد عيّنات مطلوب في ورقة
    max_features='sqrt',   # مجموعة فرعية عشوائية من الميزات عند كل تقسيم
    random_state=42,       # عيّنة Bootstrap وتقسيمات قابلة للتكرار
    n_jobs=-1              # استخدام جميع أنوية المعالج للتدريب الأسرع
)

classifier.fit(X_train, y_train)  # تدريب جميع الأشجار على عيّنات Bootstrap

print('تم تدريب الغابة العشوائية بنجاح.')
print(f'عدد الأشجار (estimators): {len(classifier.estimators_)}')  # يجب أن يساوي n_estimators
print(f'أهمية الميزات: {classifier.feature_importances_.round(4)}')  # أهمية Age مقابل Salary""",

    20: """## المثال 1 — الخلية 6: التنبؤ

التنبؤ بتسميات الفئات عبر **التصويت بالأغلبية** لجميع الأشجار.

**ما تفعله هذه الخلية:** يستخدم `predict()` و `predict_proba()` (نسب الأصوات).""",

    21: """# الخطوة 6) التنبؤ بالفئات والاحتمالات

y_pred_train = classifier.predict(X_train)            # التصويت بالأغلبية لمجموعة التدريب
y_pred_test = classifier.predict(X_test)              # التصويت بالأغلبية لمجموعة الاختبار
y_proba_test = classifier.predict_proba(X_test)[:, 1]  # نسبة الأشجار التي صوّتت للفئة 1

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    label = 'نعم' if y_pred_test[i] == 1 else 'لا'
    actual = 'نعم' if y_test[i] == 1 else 'لا'
    print(f'  فعلي={actual}، متوقع={label}، تصويت الأشجار P(نعم)={y_proba_test[i]:.2%}')""",

    22: """## المثال 1 — الخلية 7: التصوير

رسم **مناطق القرار** من الغابة العشوائية (أنعم من شجرة واحدة).

**ما تفعله هذه الخلية:** يعرض مناطق التصنيف بالتصويت في مستوى العمر–الراتب.""",

    23: """# الخطوة 7) التصوير — مناطق قرار الغابة العشوائية

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(grid).reshape(xx.shape)  # التصويت بالأغلبية عند كل نقطة شبكة

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlGn')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='لم يُشترَ', alpha=0.5, s=40)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', label='تم الشراء', alpha=0.5, s=40)
plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', edgecolors='k', s=80, label='نقاط الاختبار')
plt.xlabel('العمر')
plt.ylabel('الراتب التقديري (USD)')
plt.title('مناطق قرار الغابة العشوائية — إعلانات الشبكات الاجتماعية')
plt.legend()
plt.tight_layout()
plt.show()""",

    24: """## المثال 1 — الخلية 8: التقييم

التقييم بالدقة والدقة الإيجابية والاستدعاء وF1 ومصفوفة الالتباس.

**ما تفعله هذه الخلية:** يحسب مقاييس التصنيف على مجموعة الاختبار.""",

    25: """# الخطوة 8) التقييم — مقاييس التصنيف

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'المقياس': ['الدقة', 'الدقة الإيجابية', 'الاستدعاء', 'مقياس F1'],
    'القيمة': [acc, prec, rec, f1],
    'الوصف': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع «نعم»، نسبة ما كان فعلاً «نعم»',
        'من «نعم» الفعلي، نسبة ما تنبأ به النموذج بشكل صحيح',
        'توازن بين الدقة الإيجابية والاستدعاء'
    ]
})

display(results.round(4))

cm = confusion_matrix(y_test, y_pred_test)
print('\\nمصفوفة الالتباس (صفوف=فعلي، أعمدة=متوقع):')
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['لا (0)', 'نعم (1)']).plot(ax=ax, cmap='Blues')
plt.title('مصفوفة الالتباس — الغابة العشوائية')
plt.tight_layout()
plt.show()

print(f'\\nدقة اختبار المثال 1 = {acc:.4f}')""",

    26: """## لماذا تعمل الغابة العشوائية لإعلانات الشبكات الاجتماعية؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **أنماط غير خطية** | سلوك الشراء يختلف حسب العمر/الراتب — الغابة تلتقط مناطق معقدة |
| 2 | **تقليل الإفراط في التلائم** | التصويت عبر 100 شجرة أكثر استقراراً من CART عميقة واحدة |
| 3 | **لا حاجة للتحجيم** | الأشجار تقسم عند عتبات العمر والراتب الخام |
| 4 | **تنوع Bootstrap** | كل شجرة ترى عيّنة مختلفة قليلاً — مجموعة قوية |
| 5 | **احتمال من الأصوات** | P(نعم) = نسبة الأشجار التي صوّتت «نعم» — مفيد لترتيب العملاء المحتملين |

> **الخلاصة:** الغابة العشوائية **تصوّت** عبر أشجار كثيرة — أنعم وغالباً أدق من CART واحدة.""",

    27: """## فهم مقاييس التصنيف — المثال 1

| المقياس | سياق الغابة العشوائية |
|--------|----------------------|
| **الدقة** | إجمالي الأصوات الصحيحة بالأغلبية |
| **الدقة الإيجابية** | عندما تتنبأ الغابة «نعم»، كم مرة تكون صحيحة |
| **الاستدعاء** | من كل «نعم» الفعلي، كم تلتقطها الغابة |
| **F1** | توازن عندما تكون الفئات غير متوازنة |

> **المقارنة:** الغابة العشوائية مقابل شجرة قرار واحدة على نفس البيانات — RF عادةً **دقة اختبار أعلى**.""",

    28: """---
# المثال 2: الموافقة على القرض — تصنيف الغابة العشوائية

التنبؤ **Approved** مقابل **Rejected** من أربع ميزات مالية باستخدام الغابة العشوائية.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `CreditScore` | ميزة (X₁) | درجة الائتمان (300–850) |
| `Income` | ميزة (X₂) | الدخل السنوي بالدولار |
| `LoanAmount` | ميزة (X₃) | مبلغ القرض المطلوب بالدولار |
| `YearsEmployed` | ميزة (X₄) | سنوات في الوظيفة الحالية |
| `Approved` | متغير مستهدف (y) | 1 = موافق، 0 = مرفوض |

**الملف:** `../Datasets/loan_approval.csv`""",

    29: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV الموافقة على القرض وفحص البيانات.

**ما تفعله هذه الخلية:** قراءة `../Datasets/loan_approval.csv` وعرض إحصائيات أساسية.""",

    30: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/loan_approval.csv')

FEATURE_COLS = ['CreditScore', 'Income', 'LoanAmount', 'YearsEmployed']
TARGET_COL = 'Approved'

print('أول 5 صفوف:')
display(dataset.head())

print('\\nمعلومات مجموعة البيانات:')
dataset.info()

print('\\nملخص إحصائي:')
display(dataset.describe())

print(f'\\nتوازن الفئات ({TARGET_COL}):')
print(dataset[TARGET_COL].value_counts())

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')""",

    31: """## المثال 2 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

تتضمن مجموعة البيانات هذه قيماً مفقودة في الميزات لعرض `SimpleImputer`.

**ما تفعله هذه الخلية:** يتحقق من القيم الفارغة ويزيل التكرارات ويستبدل القيم المفقودة.""",

    32: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')
print(dataset.isnull().sum())

rows_before = len(dataset)
dataset = dataset.drop_duplicates().reset_index(drop=True)
print(f'\\nالتكرارات المُزالة: {rows_before - len(dataset)}')

if dataset[TARGET_COL].isnull().sum() > 0:
    dataset = dataset.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print('تمت إزالة الصفوف ذات الهدف المفقود')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
cols = FEATURE_COLS + [TARGET_COL]
if dataset[cols].isnull().sum().sum() > 0:
    dataset[cols] = imputer.fit_transform(dataset[cols])
    print('تم استبدال قيم الميزات المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُستبدِل')

print(f'\\nعدد الصفوف بعد التنظيف: {len(dataset)}')""",

    33: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة عددية — يُتخطَّى الترميز.

**ما تفعله هذه الخلية:** يؤكد عدم الحاجة إلى ترميز فئوي.""",

    34: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات: {FEATURE_COLS}')""",

    35: """## المثال 2 — الخلية 4: تقسيم البيانات

تعريف X (4 ميزات) و y (Approved)، ثم تقسيم 80/20 مع stratify.

**ما تفعله هذه الخلية:** إنشاء مصفوفات الميزات/الهدف وتطبيق train_test_split الطبقي.""",

    36: """# الخطوة 4) تقسيم التدريب-الاختبار

X = dataset[FEATURE_COLS].values
y = dataset[TARGET_COL].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'شكل X_train: {X_train.shape}')
print(f'شكل X_test:  {X_test.shape}')
print(f'عدد فئات y_train: {np.bincount(y_train)}')
print(f'عدد فئات y_test:  {np.bincount(y_test)}')""",

    37: """## المثال 2 — الخلية 5: تدريب النموذج

تدريب الغابة العشوائية على أربع ميزات مالية — كل شجرة تصوّت على الموافقة.

**ما تفعله هذه الخلية:** يلائم الغابة ويعرض أهمية الميزات.""",

    38: """# الخطوة 5) تدريب مصنّف الغابة العشوائية

classifier = RandomForestClassifier(
    n_estimators=100,      # 100 شجرة في المجموعة
    max_depth=8,           # أشجار أعمق — بيانات أكثر من المثال 1
    min_samples_split=4,   # 4 عيّنات على الأقل لتقسيم عقدة
    min_samples_leaf=2,    # كل ورقة يجب أن تحتوي على عيّنتين على الأقل
    max_features='sqrt',   # sqrt(4) ≈ 2 ميزة تُراعى عند كل تقسيم
    random_state=42,
    n_jobs=-1              # تدريب متوازٍ عبر أنوية المعالج
)

classifier.fit(X_train, y_train)

print('تم تدريب الغابة العشوائية بنجاح.')
print(f'عدد الأشجار: {len(classifier.estimators_)}')

print('\\nأهمية الميزات:')
for name, imp in zip(FEATURE_COLS, classifier.feature_importances_):
    print(f'  {name:15s} -> {imp:.4f}')""",

    39: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بالموافقة على القرض في مجموعة الاختبار بالتصويت بالأغلبية.

**ما تفعله هذه الخلية:** يولّد تنبؤات المجموعة.""",

    40: """# الخطوة 6) التنبؤ

y_pred_test = classifier.predict(X_test)
y_proba_test = classifier.predict_proba(X_test)[:, 1]

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    actual = 'موافق' if y_test[i] == 1 else 'مرفوض'
    pred = 'موافق' if y_pred_test[i] == 1 else 'مرفوض'
    print(f'  فعلي={actual}، متوقع={pred}، تصويت الأشجار P(موافق)={y_proba_test[i]:.2%}')""",

    41: """## المثال 2 — الخلية 7: التصوير

رسم **أهمية الميزات** و **مصفوفة الالتباس**.

**ما تفعله هذه الخلية:** يعرض أي الميزات تدفع قرارات الموافقة عبر جميع الأشجار.""",

    42: """# الخطوة 7) التصوير — أهمية الميزات + مصفوفة الالتباس

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(FEATURE_COLS, classifier.feature_importances_, color='darkgreen')
axes[0].set_xlabel('الأهمية')
axes[0].set_title('أهمية الميزات — الموافقة على القرض (الغابة العشوائية)')

cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['مرفوض', 'موافق']).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('مصفوفة الالتباس — الغابة العشوائية')

plt.tight_layout()
plt.show()""",

    43: """## المثال 2 — الخلية 8: التقييم

تقرير تصنيف كامل مع الدقة والدقة الإيجابية والاستدعاء وF1.

**ما تفعله هذه الخلية:** يحسب ويعرض مقاييس التقييم.""",

    44: """# الخطوة 8) التقييم

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'المقياس': ['الدقة', 'الدقة الإيجابية', 'الاستدعاء', 'مقياس F1'],
    'القيمة': [acc, prec, rec, f1],
    'الوصف': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع «موافق»، نسبة ما كان فعلاً موافقاً',
        'من «موافق» الفعلي، نسبة ما تنبأ به النموذج بشكل صحيح',
        'توازن بين الدقة الإيجابية والاستدعاء'
    ]
})

display(results.round(4))

print('\\nتقرير تصنيف مفصل:')
print(classification_report(y_test, y_pred_test, target_names=['مرفوض (0)', 'موافق (1)']))

print(f'\\nدقة اختبار المثال 2 = {acc:.4f}')""",

    45: """## لماذا تعمل الغابة العشوائية جيداً للموافقة على القرض؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **ميزات متعددة** | الغابة تقسم على الائتمان والدخل والقرض والسنوات عبر أشجار كثيرة |
| 2 | **قواعد غير خطية** | منطق موافقة معقد — المجموعة تلتقط التفاعلات |
| 3 | **أهمية الميزات** | تُظهر أن CreditScore غالباً هي الأبرز (متوسط عبر جميع الأشجار) |
| 4 | **لا حاجة للتحجيم** | القيم المالية الخام تعمل مباشرة |
| 5 | **دقة عالية** | التصويت عادةً يتفوق على شجرة قرار واحدة على هذه المجموعة |

> **المقارنة:** الغابة العشوائية مقابل CART — نفس المقايضة في التفسير، لكن RF عادةً **دقة أعلى** و **إفراط تلائم أقل**.""",

    46: """## فهم مقاييس التصنيف — المثال 2

| السيناريو | المقياس المركز |
|----------|--------------|
| **الموافقة الخاطئة مكلفة** | **الدقة الإيجابية** |
| **الرفض الخاطئ مكلف** | **الاستدعاء** |
| **تكلفة متوازنة** | **الدقة** أو **F1** |

> **فحص الإفراط في التلائم:** إذا كانت دقة التدريب ≈ 100% لكن دقة الاختبار أقل، قلّل `max_depth` أو زِد `min_samples_leaf`.""",
}
