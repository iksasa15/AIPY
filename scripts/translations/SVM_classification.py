CELLS = {
    0: """# آلة المتجهات الداعمة (SVM) — التصنيف (Google Colab)

**الهدف:** التنبؤ بـ **فئة ثنائية** (0/1) باستخدام **تصنيف المتجهات الداعمة (SVC)** — يجد **حد قرار** أمثل بأقصى **هامش** بين الفئات.

| مثال | الميزة/الميزات (X) | الهدف (y) | مجموعة البيانات |
|---------|----------------|------------|---------|
| **المثال 1** | Age, EstimatedSalary | Purchased (0/1) | `../Datasets/social_network_ads.csv` |
| **المثال 2** | CreditScore, Income, LoanAmount, YearsEmployed | Approved (0/1) | `../Datasets/loan_approval.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | المستوى الفائق، الهامش، النواة، C، gamma |
| المرحلة 1 | معالجة البيانات | التحميل ← التنظيف ← الترميز ← التقسيم |
| المرحلة 2 | الخوارزمية | التدريب ← التنبؤ ← التصور ← التقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)""",

    1: """---
# دليل الخوارزمية — آلة المتجهات الداعمة (التصنيف)

## ما هي SVM للتصنيف؟

**SVM (Support Vector Machine)** تجد **أفضل حد فاصل** (مستوى فائق) بين الفئات عبر **تعظيم الهامش** — المسافة إلى أقرب النقاط من كل فئة.

تُسمى تلك النقاط الأقرب **المتجهات الداعمة (Support Vectors)**.

| الخاصية | التفاصيل |
|----------|--------|
| النوع | مصنّف **تمييزي** — يركز على الحد الفاصل |
| الفكرة الأساسية | مستوى فائق بـ **أقصى هامش** |
| النقاط الرئيسية | **المتجهات الداعمة** — فقط هذه تحدد الحد |
| المخرجات | تسمية الفئة + `predict_proba()` (مع `probability=True`) |

## SVM خطية (فئتان)

**دالة القرار:**

`f(x) = w·x + b`

| القاعدة | التنبؤ |
|------|------------|
| f(x) ≥ 0 | الفئة **1** |
| f(x) < 0 | الفئة **0** |

**الهدف:** تعظيم الهامش بين الفئات مع تقليل التصنيف الخاطئ.

## الهامش

```
    الفئة 0  |  الهامش  |  الفئة 1
   ●●●●●●   |    ↑     |   ●●●●●●
            المستوى الفائق
```

**المتجهات الداعمة** = نقاط التدريب الأقرب إلى الحد (على الهامش أو داخله).

## المعاملات الفائقة الرئيسية

| المعامل | الدور | التأثير |
|-----------|------|--------|
| `C` | التنظيم | **C كبير** → هامش ضيق، يلتصق بالبيانات (خطر الإفراط في التعلّم) |
| | | **C صغير** → هامش أوسع، حد أبسط |
| `kernel` | شكل سطح القرار | `'linear'`, `'rbf'`, `'poly'` |
| `gamma` | مدى نواة RBF (`rbf` فقط) | **γ كبير** → حد ضيق ومعقد |
| | | **γ صغير** → حد أكثر نعومة |
| `probability` | تفعيل `predict_proba()` | `True` يستخدم Platt scaling (تدريب أبطأ) |

## حيلة النواة (SVM غير خطية)

عندما **لا تكون الفئات قابلة للفصل خطياً**، تُنقل الميزات إلى أبعاد أعلى:

| النواة | حالة الاستخدام |
|--------|----------|
| **`linear`** | بيانات قابلة للفصل خطياً |
| **`rbf`** (Gaussian) | **الاختيار الافتراضي** للحدود المنحنية |
| **`poly`** | علاقات متعددة الحدود |

**نواة RBF:** `K(x, x') = exp(−γ ||x − x'||²)`

## تحجيم الميزات — مطلوب!

تستخدم SVM **المسافات** في التحسين. الميزات ذات المقياس الأكبر تسيطر → استخدم دائماً **`StandardScaler`** قبل SVM.

## SVM مقابل مصنّفات أخرى

| | الانحدار اللوجستي | K-NN | SVM |
|---|---------------------|------|-----|
| الحد | خطي (log-odds) | محلي / مرن | **أقصى هامش**، قائم على النواة |
| التحجيم | موصى به | **مطلوب** | **مطلوب** |
| القوة | احتمالات، سريع | بسيط | **قوي** على مجموعات متوسطة |
| الضعف | خطي فقط | تنبؤ بطيء | بطيء على بيانات ضخمة |

## ما يجب أن يتذكره الطالب

1. SVM تعظّم **الهامش** بين الفئات.
2. **المتجهات الداعمة** هي نقاط التدريب الحاسمة.
3. **احجِم الميزات دائماً** قبل SVM.
4. استخدم **`kernel='rbf'`** للحدود غير الخطية.
5. اضبط **`C`** و **`gamma`** — القيم العالية قد تسبب إفراطاً في التعلّم.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات للمعالجة المسبقة وتصنيف SVM والتقييم.

**ما تفعله هذه الخلية:** تحمّل numpy و pandas و matplotlib و sklearn SVC و Pipeline والمقاييس.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء المخططات والرسوم
import seaborn as sns           # تصورات إحصائية (خرائط حرارية)

from sklearn.model_selection import train_test_split       # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.preprocessing import StandardScaler           # تحجيم الميزات (مطلوب لـ SVM)
from sklearn.pipeline import Pipeline                      # ربط المقياس + المصنّف
from sklearn.svm import SVC                                # تصنيف المتجهات الداعمة
from sklearn.metrics import (                              # مقاييس التصنيف
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

plt.rcParams['figure.figsize'] = (10, 6)  # حجم المخطط الافتراضي: عرض=10، ارتفاع=6 بوصة
plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لتقسيمات قابلة للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح""",

    6: """---
# المثال 1: إعلانات الشبكات الاجتماعية — تصنيف SVM

التنبؤ بما إذا كان المستخدم **Purchased** من **Age** و **EstimatedSalary** باستخدام SVM مع **نواة RBF**.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Age` | ميزة (X₁) | عمر المستخدم بالسنوات |
| `EstimatedSalary` | ميزة (X₂) | الراتب السنوي التقديري بالدولار |
| `Purchased` | الهدف (y) | 0 = لا، 1 = نعم |

**الملف:** `../Datasets/social_network_ads.csv`""",

    7: """---
# المرحلة 1: معالجة البيانات

تحضير البيانات قبل التدريب — نفس القالب يُعاد استخدامه لخوارزميات أخرى.""",

    8: """## المثال 1 — الخلية 1: تحميل واستكشاف البيانات

تحميل ملف CSV وإجراء استكشاف أولي (head، info، describe، shape).

**ما تفعله هذه الخلية:** تقرأ `../Datasets/social_network_ads.csv` وتعرض إحصائيات أساسية.""",

    9: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/social_network_ads.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['Age', 'EstimatedSalary']  # ميزتان رقميتان للإدخال
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

التحقق من القيم المفقودة، إزالة التكرارات، وتطبيق الاستبدال إذا لزم الأمر.

**ما تفعله هذه الخلية:** ينظّف مجموعة البيانات قبل النمذجة.""",

    11: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')
print(dataset.isnull().sum())

rows_before = len(dataset)
dataset = dataset.drop_duplicates().reset_index(drop=True)
rows_after = len(dataset)
print(f'\\nالتكرارات المحذوفة: {rows_before - rows_after}')

num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
if dataset.isnull().sum().sum() > 0:
    dataset[num_cols] = imputer.fit_transform(dataset[num_cols])
    print('تم استبدال القيم المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُستبدِل')

print(f'\\nالصفوف بعد التنظيف: {rows_after}')""",

    12: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة رقمية — يُتخطى الترميز.

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

**ما تفعله هذه الخلية:** ينشئ مصفوفات الميزات/الهدف ويطبّق train_test_split طبقي.""",

    15: """# الخطوة 4) تقسيم تدريب-اختبار (طبقي للتصنيف)

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

    16: """> **ملاحظة:** SVM **تتطلب تحجيم الميزات**. خط الأنابيب أدناه يستخدم `StandardScaler` قبل `SVC` — إلزامي لـ Age و Salary.""",

    17: """---
# المرحلة 2: تصنيف SVM — إعلانات الشبكات الاجتماعية

تدريب SVC بنواة RBF وتصور حد القرار.""",

    18: """## المثال 1 — الخلية 5: تدريب النموذج

بناء **Pipeline**: StandardScaler → SVC مع `kernel='rbf'`.

**ما تفعله هذه الخلية:** يدرّب SVM ويُبلغ عن عدد المتجهات الداعمة.""",

    19: """# الخطوة 5) تدريب مصنّف المتجهات الداعمة (SVM)

classifier = Pipeline([
    ('scaler', StandardScaler()),  # تحجيم Age و Salary — مطلوب لـ SVM
    ('model', SVC(
        kernel='rbf',          # دالة أساس شعاعية — حد غير خطي
        C=1.0,                   # التنظيم (موازنة الهامش مقابل التصنيف الخاطئ)
        gamma='scale',           # معامل النواة: 1 / (n_features * X.var())
        probability=True,        # تفعيل predict_proba() عبر Platt scaling
        random_state=42
    ))
])

classifier.fit(X_train, y_train)  # إيجاد حد أقصى هامش

model = classifier.named_steps['model']  # استخراج SVC المدرَّب
print('تم تدريب SVM (RBF) بنجاح.')
print(f'عدد المتجهات الداعمة (الفئة 0): {model.n_support_[0]}')
print(f'عدد المتجهات الداعمة (الفئة 1): {model.n_support_[1]}')
print(f'إجمالي المتجهات الداعمة: {model.n_support_.sum()} / {len(X_train)} عينة تدريب')""",

    20: """## المثال 1 — الخلية 6: التنبؤ

التنبؤ بتسميات الفئات والاحتمالات على مجموعة الاختبار.

**ما تفعله هذه الخلية:** يستخدم `predict()` و `predict_proba()` من خط أنابيب SVM.""",

    21: """# الخطوة 6) التنبؤ بالفئات والاحتمالات

y_pred_train = classifier.predict(X_train)            # تسميات الفئات لمجموعة التدريب
y_pred_test = classifier.predict(X_test)              # تسميات الفئات لمجموعة الاختبار
y_proba_test = classifier.predict_proba(X_test)[:, 1]  # P(Purchased=1) من Platt scaling

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    label = 'نعم' if y_pred_test[i] == 1 else 'لا'
    actual = 'نعم' if y_test[i] == 1 else 'لا'
    print(f'  الفعلي={actual}، المتوقع={label}، P(شراء)={y_proba_test[i]:.2%}')""",

    22: """## المثال 1 — الخلية 7: التصور

رسم نقاط البيانات و**حد القرار** وإبراز **المتجهات الداعمة**.

**ما تفعله هذه الخلية:** يعرض مناطق تصنيف SVM ونقاط المتجهات الداعمة الحاسمة.""",

    23: """# الخطوة 7) التصور — حد القرار + المتجهات الداعمة

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(grid).reshape(xx.shape)

# المتجهات الداعمة في فضاء الميزات الأصلي (الفهارس تشير إلى مجموعة التدريب المُقاسة)
svc = classifier.named_steps['model']
X_train_scaled = classifier.named_steps['scaler'].transform(X_train)
# إعادة المتجهات الداعمة إلى المقياس الأصلي للرسم
sv_indices = svc.support_
sv_points = X_train[sv_indices]

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlGn')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='لم يشتِر', alpha=0.4, s=35)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', label='اشترى', alpha=0.4, s=35)
plt.scatter(sv_points[:, 0], sv_points[:, 1], s=120, facecolors='none', edgecolors='black', linewidths=2, label='المتجهات الداعمة')
plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', edgecolors='k', s=80, label='نقاط الاختبار')
plt.xlabel('العمر')
plt.ylabel('الراتب التقديري (USD)')
plt.title('حد قرار SVM (RBF) — إعلانات الشبكات الاجتماعية')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()""",

    24: """## المثال 1 — الخلية 8: التقييم

التقييم بـ Accuracy و Precision و Recall و F1 ومصفوفة الالتباس.

**ما تفعله هذه الخلية:** يحسب مقاييس التصنيف على مجموعة الاختبار.""",

    25: """# الخطوة 8) التقييم — مقاييس التصنيف

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [acc, prec, rec, f1],
    'Description': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع نعم، النسبة الفعلية نعم',
        'من الفعلي نعم، النسبة المتوقعة بشكل صحيح',
        'التوازن بين الدقة والاسترجاع'
    ]
})

display(results.round(4))

cm = confusion_matrix(y_test, y_pred_test)
print('\\nمصفوفة الالتباس (الصفوف=فعلي، الأعمدة=متوقع):')
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['لا (0)', 'نعم (1)']).plot(ax=ax, cmap='Blues')
plt.title('مصفوفة الالتباس — SVM (RBF)')
plt.tight_layout()
plt.show()

print(f'\\nدقة اختبار المثال 1 = {acc:.4f}')""",

    26: """## لماذا تعمل SVM لإعلانات الشبكات الاجتماعية؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **حد غير خطي** | نواة RBF تلتف حول مجموعات العمر–الراتب |
| 2 | **أقصى هامش** | فصل متين بين المشترين وغير المشترين |
| 3 | **المتجهات الداعمة** | فقط النقاط الحاسمة تحدد الحد — نموذج فعّال |
| 4 | **تحجيم الميزات** | الراتب والعمر بمقاييس مختلفة — StandardScaler ضروري |
| 5 | **خط أساس قوي** | غالباً تنافسية مع Random Forest على بيانات منظمة |

> **الملخص:** SVM بنواة RBF تلتقط مناطق الشراء المعقدة أفضل من مصنّف خطي.""",

    27: """## فهم مقاييس التصنيف — المثال 1

| المقياس | سياق SVM |
|--------|-------------|
| **Accuracy** | القرارات الصحيحة الإجمالية القائمة على الهامش |
| **Precision** | عندما تتنبأ SVM بنعم، كم مرة تكون صحيحة |
| **Recall** | من كل نعم فعلية، كم تلتقطها SVM |
| **F1** | التوازن عندما تكون الفئات غير متوازنة |

> **نصيحة:** قارن SVM مع الانحدار اللوجستي على نفس البيانات — RBF غالباً تفوز عندما يكون الحد منحنياً.""",

    28: """---
# المثال 2: الموافقة على القرض — تصنيف SVM

التنبؤ بـ **Approved** مقابل **Rejected** من أربع ميزات مالية باستخدام SVM.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `CreditScore` | ميزة (X₁) | درجة الائتمان (300–850) |
| `Income` | ميزة (X₂) | الدخل السنوي بالدولار |
| `LoanAmount` | ميزة (X₃) | مبلغ القرض المطلوب بالدولار |
| `YearsEmployed` | ميزة (X₄) | سنوات العمل في الوظيفة الحالية |
| `Approved` | الهدف (y) | 1 = موافق، 0 = مرفوض |

**الملف:** `../Datasets/loan_approval.csv`""",

    29: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV الموافقة على القرض وفحص البيانات.

**ما تفعله هذه الخلية:** تقرأ `../Datasets/loan_approval.csv` وتعرض إحصائيات أساسية.""",

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

**ما تفعله هذه الخلية:** يتحقق من القيم الفارغة، يزيل التكرارات، ويستبدل القيم المفقودة.""",

    32: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')
print(dataset.isnull().sum())

rows_before = len(dataset)
dataset = dataset.drop_duplicates().reset_index(drop=True)
print(f'\\nالتكرارات المحذوفة: {rows_before - len(dataset)}')

if dataset[TARGET_COL].isnull().sum() > 0:
    dataset = dataset.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print('تم حذف الصفوف ذات الهدف المفقود')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
cols = FEATURE_COLS + [TARGET_COL]
if dataset[cols].isnull().sum().sum() > 0:
    dataset[cols] = imputer.fit_transform(dataset[cols])
    print('تم استبدال قيم الميزات المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُستبدِل')

print(f'\\nالصفوف بعد التنظيف: {len(dataset)}')""",

    33: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة رقمية — يُتخطى الترميز.

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

**ما تفعله هذه الخلية:** ينشئ مصفوفات الميزات/الهدف ويطبّق train_test_split طبقي.""",

    36: """# الخطوة 4) تقسيم تدريب-اختبار

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

تدريب SVM بنواة RBF على أربع ميزات مالية مُقاسة.

**ما تفعله هذه الخلية:** يدرّب خط الأنابيب ويُبلغ عن عدد المتجهات الداعمة.""",

    38: """# الخطوة 5) تدريب مصنّف المتجهات الداعمة (SVM)

classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(
        kernel='rbf',
        C=10.0,                # C أعلى قليلاً — حد أكثر تعقيداً لـ 4 ميزات
        gamma='scale',
        probability=True,
        random_state=42
    ))
])

classifier.fit(X_train, y_train)

model = classifier.named_steps['model']
print('تم تدريب SVM (RBF) بنجاح.')
print(f'المتجهات الداعمة: {model.n_support_.sum()} / {len(X_train)} عينة تدريب')
print(f'  مرفوض (0): {model.n_support_[0]}')
print(f'  موافق (1): {model.n_support_[1]}')""",

    39: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بالموافقة على القرض على مجموعة الاختبار.

**ما تفعله هذه الخلية:** يولّد تنبؤات SVM والاحتمالات.""",

    40: """# الخطوة 6) التنبؤ

y_pred_test = classifier.predict(X_test)
y_proba_test = classifier.predict_proba(X_test)[:, 1]

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    actual = 'موافق' if y_test[i] == 1 else 'مرفوض'
    pred = 'موافق' if y_pred_test[i] == 1 else 'مرفوض'
    print(f'  الفعلي={actual}، المتوقع={pred}، P(موافق)={y_proba_test[i]:.2%}')""",

    41: """## المثال 2 — الخلية 7: التصور

مقارنة دقة **النواة الخطية مقابل RBF** وعرض **مصفوفة الالتباس**.

**ما تفعله هذه الخلية:** يوضّح تأثير اختيار النواة على تصنيف الموافقة على القرض.""",

    42: """# الخطوة 7) التصور — مقارنة النواة + مصفوفة الالتباس

kernels = ['linear', 'rbf']
kernel_scores = {}

for k in kernels:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(kernel=k, C=10.0, gamma='scale', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    kernel_scores[k] = accuracy_score(y_test, pipe.predict(X_test))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- يسار: مخطط أعمدة مقارنة النواة ---
axes[0].bar(kernels, [kernel_scores[k] for k in kernels], color=['steelblue', 'seagreen'])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('دقة الاختبار')
axes[0].set_title('مقارنة نواة SVM — الموافقة على القرض')
for i, k in enumerate(kernels):
    axes[0].text(i, kernel_scores[k] + 0.02, f'{kernel_scores[k]:.2%}', ha='center')

# --- يمين: مصفوفة الالتباس لنموذج RBF ---
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['مرفوض', 'موافق']).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('مصفوفة الالتباس — SVM (RBF)')

plt.tight_layout()
plt.show()""",

    43: """## المثال 2 — الخلية 8: التقييم

تقرير تصنيف كامل مع Accuracy و Precision و Recall و F1.

**ما تفعله هذه الخلية:** يحسب ويعرض مقاييس التقييم.""",

    44: """# الخطوة 8) التقييم

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [acc, prec, rec, f1],
    'Description': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع موافق، النسبة الفعلية موافق',
        'من الفعلي موافق، النسبة المتوقعة بشكل صحيح',
        'التوازن بين الدقة والاسترجاع'
    ]
})

display(results.round(4))

print('\\nتقرير تصنيف مفصّل:')
print(classification_report(y_test, y_pred_test, target_names=['مرفوض (0)', 'موافق (1)']))

print(f'\\nدقة اختبار المثال 2 = {acc:.4f}')""",

    45: """## لماذا تعمل SVM جيداً للموافقة على القرض؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **حد متعدد الميزات** | SVM تفصل موافق/مرفوض في فضاء 4D مُقاس |
| 2 | **نواة RBF** | تلتقط قواعد موافقة غير خطية (تفاعلات الائتمان + القرض) |
| 3 | **التحجيم حاسم** | الدخل (~80k) مقابل YearsEmployed (~10) — يجب التقييس |
| 4 | **المتجهات الداعمة** | قد تكون ~30–50% من العينات فقط متجهات داعمة — نموذج مدمج |
| 5 | **مقارنة النواة** | RBF غالباً تتفوق على الخطية عندما يكون سطح القرار منحنياً |

> **قارن:** SVM مقابل Random Forest على بيانات القرض — كلاهما قوي؛ SVM تعطي تفسيراً بأقصى هامش.""",

    46: """## فهم مقاييس التصنيف — المثال 2

| الضبط | الإرشاد |
|--------|----------|
| **الإفراط في التعلّم** | قلّل `C` أو `gamma` |
| **نقص التعلّم** | زِد `C` أو جرّب `kernel='rbf'` |
| **بيانات خطية** | `kernel='linear'` أسرع وأكثر قابلية للتفسير |

> استخدم **GridSearchCV** في الإنتاج لضبط `C` و `gamma` و `kernel` معاً.""",
}
