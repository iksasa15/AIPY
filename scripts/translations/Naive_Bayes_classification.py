CELLS = {
    0: """# بايز الساذج — التصنيف (Google Colab)

**الهدف:** التنبؤ بـ **فئة ثنائية** (0/1) باستخدام **بايز الساذج** — يطبّق **نظرية بايز** مع افتراض **استقلال ساذج** بين الميزات.

| مثال | الميزة/الميزات (X) | الهدف (y) | مجموعة البيانات |
|---------|----------------|------------|---------|
| **المثال 1** | Age, EstimatedSalary | Purchased (0/1) | `../Datasets/social_network_ads.csv` |
| **المثال 2** | CreditScore, Income, LoanAmount, YearsEmployed | Approved (0/1) | `../Datasets/loan_approval.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | نظرية بايز، بايز الغاوسي، الاستقلال |
| المرحلة 1 | معالجة البيانات | التحميل ← التنظيف ← الترميز ← التقسيم |
| المرحلة 2 | الخوارزمية | التدريب ← التنبؤ ← التصور ← التقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)""",

    1: """---
# دليل الخوارزمية — بايز الساذج (الغاوسي)

## ما هو بايز الساذج؟

**بايز الساذج** هو مصنّف **احتمالي** يستند إلى **نظرية بايز**. يتنبأ بالفئة ذات **أعلى احتمال لاحق**.

| الخاصية | التفاصيل |
|----------|--------|
| النوع | نموذج **توليدي** — يتعلّم P(x \| class) و P(class) |
| «الساذج» | يفترض أن الميزات **مستقلة** بشرط الفئة |
| المخرجات | تسمية الفئة + **احتمالات الفئات** |
| السرعة | تدريب وتنبؤ **سريعان جداً** |

## نظرية بايز

`P(class | x) = P(x | class) · P(class) / P(x)`

للتصنيف نقارن:

`P(class | x) ∝ P(class) · P(x | class)`

نختار الفئة ذات **أكبر** احتمال لاحق.

## افتراض الاستقلال «الساذج»

يُفترض أن الميزات **مستقلة شرطياً** بشرط الفئة:

`P(x₁, x₂, … | class) = P(x₁ | class) · P(x₂ | class) · …`

| الواقع | افتراض بايز الساذج |
|---------|------------------------|
| الميزات غالباً مترابطة | يعامل كل ميزة **بشكل منفصل** |
| الأثر | يبسّط الرياضيات — ويعمل بشكل مدهش في الممارسة |

## بايز الساذج الغاوسي (ميزات متصلة)

لكل فئة **c** وميزة **j**، يقدّر sklearn:

| المعامل | المعنى |
|-----------|---------|
| **السابق** P(c) | نسبة عينات التدريب في الفئة c |
| **المتوسط μ_cj** | متوسط الميزة j في الفئة c |
| **التباين σ²_cj** | انتشار الميزة j في الفئة c |

تُنمذج كل ميزة كـ **منحنى غاوسي (طبيعي)** لكل فئة.

## خطوات التنبؤ

| الخطوة | الإجراء |
|------|--------|
| 1 | تقدير **السابقات** P(class) من تسميات التدريب |
| 2 | لكل فئة، تقدير **المتوسط والتباين** لكل ميزة |
| 3 | لنقطة جديدة **x**، حساب الاحتمالية تحت كل فئة (حاصل ضرب الغاوسيات) |
| 4 | الضرب في السابق → اختيار الفئة ذات **أعلى احتمال لاحق** |

## بايز الساذج مقابل مصنّفات أخرى

| | الانحدار اللوجستي | K-NN | **بايز الساذج** |
|---|---------------------|------|-----------------|
| التحجيم | موصى به | **مطلوب** | **غير مطلوب** (بايز الغاوسي) |
| سرعة التدريب | سريع | يخزّن كل البيانات | **سريع جداً** |
| الافتراض | لوغاريتم الاحتمالات خطي | لا شيء (محلي) | الاستقلال + الغاوسي |
| بيانات صغيرة | مقبول | مقبول | غالباً **يعمل جيداً** |

## الأنواع في sklearn

| الصنف | حالة الاستخدام |
|-------|----------|
| `GaussianNB` | ميزات رقمية متصلة |
| `MultinomialNB` | بيانات عدّية (مثل تكرار الكلمات في النص) |
| `BernoulliNB` | ميزات ثنائية (0/1) |

## ما يجب أن يتذكّره الطالب

1. بايز الساذج يستخدم **نظرية بايز** + **استقلال الميزات**.
2. **GaussianNB** يُلائم **منحنى جرسي** لكل ميزة لكل فئة.
3. **لا حاجة لتحجيم الميزات** مع بايز الساذج الغاوسي.
4. يُخرج **احتمالات** عبر `predict_proba()`.
5. خط أساس سريع — ممتاز لـ **كشف البريد المزعج** والنص والنماذج الأولية السريعة.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات لمعالجة البيانات وبايز الساذج الغاوسي والتقييم.

**ما تفعله هذه الخلية:** تحمّل numpy و pandas و matplotlib و sklearn GaussianNB والمقاييس.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء المخططات والرسوم
import seaborn as sns           # تصورات إحصائية (خرائط حرارية)

from sklearn.model_selection import train_test_split       # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.naive_bayes import GaussianNB                 # مصنّف بايز الساذج الغاوسي
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
# المثال 1: إعلانات الشبكات الاجتماعية — بايز الساذج

التنبؤ بما إذا كان المستخدم قد **اشترى** من **Age** و **EstimatedSalary** باستخدام بايز الساذج الغاوسي.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Age` | ميزة (X₁) | عمر المستخدم بالسنوات |
| `EstimatedSalary` | ميزة (X₂) | الراتب السنوي التقديري بالدولار الأمريكي |
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

FEATURE_COLS = ['Age', 'EstimatedSalary']  # ميزتان رقميتان مدخلتان
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

**ما تفعله هذه الخلية:** تنظّف مجموعة البيانات قبل النمذجة.""",

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

جميع الأعمدة رقمية — يُتخطَّى الترميز.

**ما تفعله هذه الخلية:** تؤكد عدم الحاجة إلى ترميز فئوي.""",

    13: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات: {FEATURE_COLS}')
    print(f'عمود الهدف: {TARGET_COL} (0/1 مسبقاً)')""",

    14: """## المثال 1 — الخلية 4: تقسيم البيانات

تحديد X و y، ثم التقسيم 80/20 مع **stratify** للحفاظ على نسبة الفئات.

**ما تفعله هذه الخلية:** تنشئ مصفوفات الميزات/الهدف وتطبّق train_test_split الطبقي.""",

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

    16: """> **ملاحظة:** **بايز الساذج الغاوسي** **لا يتطلب** تحجيم الميزات — ينمذج توزيع كل ميزة بشكل منفصل لكل فئة.""",

    17: """---
# المرحلة 2: بايز الساذج — إعلانات الشبكات الاجتماعية

تدريب GaussianNB والتقييم على مجموعة الاختبار.""",

    18: """## المثال 1 — الخلية 5: تدريب النموذج

ملاءمة `GaussianNB` — تقدير السابقات ومعاملات الغاوس لكل ميزة.

**ما تفعله هذه الخلية:** تدرّب المصنّف وتعرض السابقات والمتوسطات المتعلّمة.""",

    19: """# الخطوة 5) تدريب بايز الساذج الغاوسي

classifier = GaussianNB()  # بايز الساذج الغاوسي للميزات المتصلة
classifier.fit(X_train, y_train)  # تقدير السابقات والمتوسطات والتباينات لكل فئة

print('تم تدريب بايز الساذج الغاوسي بنجاح.')
print(f'تسميات الفئات: {classifier.classes_}')  # [0, 1]
print(f'السابقات P(c): {classifier.class_prior_.round(4)}')  # الاحتمال السابق لكل فئة

print('\\nمتوسطات الميزات لكل فئة (صفوف=فئة 0/1، أعمدة=Age، Salary):')
for i, cls in enumerate(classifier.classes_):
    label = 'لم يشترِ' if cls == 0 else 'اشترى'
    means = classifier.theta_[i]  # متوسط كل ميزة لهذه الفئة
    print(f'  الفئة {cls} ({label}): Age={means[0]:.1f}, Salary={means[1]:,.0f}')""",

    20: """## المثال 1 — الخلية 6: التنبؤ

التنبؤ بتسميات الفئات و**الاحتمالات اللاحقة** على مجموعة الاختبار.

**ما تفعله هذه الخلية:** تستخدم `predict()` و `predict_proba()` وفق نظرية بايز.""",

    21: """# الخطوة 6) التنبؤ بالفئات والاحتمالات

y_pred_train = classifier.predict(X_train)            # تسميات الفئات لمجموعة التدريب
y_pred_test = classifier.predict(X_test)              # تسميات الفئات لمجموعة الاختبار
y_proba_test = classifier.predict_proba(X_test)[:, 1]  # P(Purchased=1 | x) من بايز

print('عينات تنبؤات (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    label = 'نعم' if y_pred_test[i] == 1 else 'لا'
    actual = 'نعم' if y_test[i] == 1 else 'لا'
    print(f'  الفعلي={actual}، المتوقع={label}، P(شراء|x)={y_proba_test[i]:.2%}')""",

    22: """## المثال 1 — الخلية 7: التصور

رسم نقاط البيانات و**حدود القرار** لبايز الساذج.

**ما تفعله هذه الخلية:** تعرض كيف يفصل بايز الساذج الغاوسي بين اشترى ولم يشترِ.""",

    23: """# الخطوة 7) التصور — مبعثر + حدود قرار بايز الساذج

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(grid).reshape(xx.shape)  # الفئة المتوقعة عند كل نقطة شبكة

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlGn')  # مناطق القرار
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='لم يشترِ (تدريب)', alpha=0.5)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', label='اشترى (تدريب)', alpha=0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', edgecolors='k', s=80, label='نقاط الاختبار')
plt.xlabel('العمر')
plt.ylabel('الراتب التقديري (دولار أمريكي)')
plt.title('حدود قرار بايز الساذج — إعلانات الشبكات الاجتماعية')
plt.legend()
plt.tight_layout()
plt.show()""",

    24: """## المثال 1 — الخلية 8: التقييم

التقييم بالدقة والاستبانة والاستدعاء وF1 ومصفوفة الالتباس.

**ما تفعله هذه الخلية:** تحسب مقاييس التصنيف على مجموعة الاختبار.""",

    25: """# الخطوة 8) التقييم — مقاييس التصنيف

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'Metric': ['الدقة', 'الاستبانة', 'الاستدعاء', 'مقياس F1'],
    'Value': [acc, prec, rec, f1],
    'Description': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع نعم، النسبة الفعلية نعم',
        'من الفعلي نعم، النسبة المتوقعة بشكل صحيح',
        'توازن بين الاستبانة والاستدعاء'
    ]
})

display(results.round(4))

cm = confusion_matrix(y_test, y_pred_test)
print('\\nمصفوفة الالتباس (صفوف=فعلي، أعمدة=متوقع):')
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['لا (0)', 'نعم (1)']).plot(ax=ax, cmap='Blues')
plt.title('مصفوفة الالتباس — بايز الساذج')
plt.tight_layout()
plt.show()

print(f'\\nدقة اختبار المثال 1 = {acc:.4f}')""",

    26: """## لماذا يعمل بايز الساذج لإعلانات الشبكات الاجتماعية؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **مخرجات احتمالية** | P(شراء \| العمر، الراتب) مفيدة لحملات التسويق |
| 2 | **تدريب سريع** | يحتاج فقط متوسط/تباين لكل فئة — دون تحسين تكراري |
| 3 | **ميزات غاوسية** | العمر والراتب متصلان — GaussianNB يناسبهما طبيعياً |
| 4 | **لا حاجة للتحجيم** | لكل ميزة تباينها الخاص لكل فئة |
| 5 | **مصنّف خط أساس** | نقطة انطلاق جيدة قبل K-NN أو الانحدار اللوجستي |

> **الخلاصة:** المشترون يميلون لأن يكونوا أكبر سناً وبرواتب أعلى — الغاوسيات الشرطية للفئة تلتقط هذا النمط.""",

    27: """## فهم مقاييس التصنيف — المثال 1

| المقياس | سياق بايز الساذج |
|--------|---------------------|
| **الدقة** | قرارات لاحقة صحيحة إجمالاً |
| **الاستبانة** | عندما يتنبأ بايز الساذج بنعم، كم مرة يكون صحيحاً |
| **الاستدعاء** | من كل نعم الفعلية، كم يجدها بايز الساذج |
| **F1** | توازن عندما ~42% اشترت |

> قارن مع **الانحدار اللوجستي** و **K-NN** على نفس مجموعة البيانات — دقة متشابهة، افتراضات مختلفة.""",

    28: """---
# المثال 2: الموافقة على القروض — بايز الساذج

التنبؤ بـ **Approved** مقابل **Rejected** من أربع ميزات مالية باستخدام بايز الساذج الغاوسي.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `CreditScore` | ميزة (X₁) | درجة الائتمان (300–850) |
| `Income` | ميزة (X₂) | الدخل السنوي بالدولار الأمريكي |
| `LoanAmount` | ميزة (X₃) | مبلغ القرض المطلوب بالدولار الأمريكي |
| `YearsEmployed` | ميزة (X₄) | سنوات العمل في الوظيفة الحالية |
| `Approved` | الهدف (y) | 1 = موافق، 0 = مرفوض |

**الملف:** `../Datasets/loan_approval.csv`""",

    29: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV الموافقة على القروض وفحص البيانات.

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

تتضمن مجموعة البيانات قيماً مفقودة في الميزات لعرض `SimpleImputer`.

**ما تفعله هذه الخلية:** تتحقق من القيم الفارغة وتحذف التكرارات وتستبدل القيم المفقودة.""",

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

جميع الأعمدة رقمية — يُتخطَّى الترميز.

**ما تفعله هذه الخلية:** تؤكد عدم الحاجة إلى ترميز فئوي.""",

    34: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات: {FEATURE_COLS}')""",

    35: """## المثال 2 — الخلية 4: تقسيم البيانات

تحديد X (4 ميزات) و y (Approved)، ثم التقسيم 80/20 مع stratify.

**ما تفعله هذه الخلية:** تنشئ مصفوفات الميزات/الهدف وتطبّق train_test_split الطبقي.""",

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

تدريب GaussianNB على أربع ميزات مالية — يتعلّم 4 غاوسيات لكل فئة.

**ما تفعله هذه الخلية:** يُلائم النموذج ويعرض متوسطات الميزات الشرطية للفئة.""",

    38: """# الخطوة 5) تدريب بايز الساذج الغاوسي

classifier = GaussianNB()
classifier.fit(X_train, y_train)

print('تم تدريب بايز الساذج الغاوسي بنجاح.')
print(f'السابقات: مرفوض={classifier.class_prior_[0]:.3f}، موافق={classifier.class_prior_[1]:.3f}')

print('\\nمتوسطات الميزات لكل فئة:')
mean_df = pd.DataFrame(classifier.theta_, columns=FEATURE_COLS, index=['مرفوض (0)', 'موافق (1)'])
display(mean_df.round(2))  # theta_ = المتوسطات الشرطية للفئة""",

    39: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بالموافقة على القرض في مجموعة الاختبار مع الاحتمالات اللاحقة.

**ما تفعله هذه الخلية:** يولّد التنبؤات ويعرض عينات من المخرجات.""",

    40: """# الخطوة 6) التنبؤ

y_pred_test = classifier.predict(X_test)
y_proba_test = classifier.predict_proba(X_test)[:, 1]

print('عينات تنبؤات (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    actual = 'موافق' if y_test[i] == 1 else 'مرفوض'
    pred = 'موافق' if y_pred_test[i] == 1 else 'مرفوض'
    print(f'  الفعلي={actual}، المتوقع={pred}، P(موافق|x)={y_proba_test[i]:.2%}')""",

    41: """## المثال 2 — الخلية 7: التصور

رسم **متوسطات الميزات الشرطية للفئة** (خريطة حرارية) و**مصفوفة الالتباس**.

**ما تفعله هذه الخلية:** تقارن متوسط قيم الميزات للمتقدمين الموافق عليهم مقابل المرفوضين.""",

    42: """# الخطوة 7) التصور — خريطة حرارية لمتوسطات الميزات + مصفوفة الالتباس

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- اليسار: خريطة حرارية لمتوسطات الميزات لكل فئة ---
mean_df = pd.DataFrame(classifier.theta_, columns=FEATURE_COLS, index=['مرفوض', 'موافق'])
sns.heatmap(mean_df, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0])
axes[0].set_title('متوسطات الميزات الشرطية للفئة (بايز الساذج)')
axes[0].set_ylabel('الفئة')

# --- اليمين: مصفوفة الالتباس ---
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['مرفوض', 'موافق']).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('مصفوفة الالتباس — بايز الساذج')

plt.tight_layout()
plt.show()""",

    43: """## المثال 2 — الخلية 8: التقييم

تقرير تصنيف كامل بالدقة والاستبانة والاستدعاء وF1.

**ما تفعله هذه الخلية:** تحسب وتعرض مقاييس التقييم.""",

    44: """# الخطوة 8) التقييم

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'Metric': ['الدقة', 'الاستبانة', 'الاستدعاء', 'مقياس F1'],
    'Value': [acc, prec, rec, f1],
    'Description': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع موافق، النسبة الفعلية موافق',
        'من الفعلي موافق، النسبة المتوقعة بشكل صحيح',
        'توازن بين الاستبانة والاستدعاء'
    ]
})

display(results.round(4))

print('\\nتقرير تصنيف مفصّل:')
print(classification_report(y_test, y_pred_test, target_names=['مرفوض (0)', 'موافق (1)']))

print(f'\\nدقة اختبار المثال 2 = {acc:.4f}')""",

    45: """## لماذا يعمل بايز الساذج للموافقة على القروض؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **فروق واضحة بين الفئات** | المتقدمون الموافق عليهم لديهم ائتمان/دخل أعلى في المتوسط |
| 2 | **افتراض الاستقلال** | تقريب مقبول عندما توفّر الميزات إشارات مكمّلة |
| 3 | **سريع وبسيط** | لا معاملات فائقة للضبط — خط أساس جيد |
| 4 | **احتمالات** | P(موافق \| x) يدعم عتبات قائمة على المخاطر |
| 5 | **فئات متوازنة** | تقسيم 50/50 — الدقة ذات معنى |

> **رؤية:** الخريطة الحرارية تُظهر أن فئة الموافق لديها متوسطات أعلى لـ CreditScore و Income — يتوافق مع المنطق التجاري.""",

    46: """## فهم مقاييس التصنيف — المثال 2

| السيناريو | المقياس المركّز |
|----------|--------------|
| **الموافقة الخاطئة مكلفة** | **الاستبانة** |
| **الرفض الخاطئ مكلف** | **الاستدعاء** |
| **تكلفة متوازنة** | **الدقة** أو **F1** |

> **القيود:** إذا كانت الميزات **مترابطة بقوة** (Income مقابل LoanAmount)، يُنتهك افتراض الاستقلال — فكّر في الانحدار اللوجستي أو نماذج الأشجار.""",
}
