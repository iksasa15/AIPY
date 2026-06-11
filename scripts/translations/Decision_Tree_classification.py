# Arabic translations for Decision_Tree_classification.ipynb (47 cells)

CELLS = {
    0: """# تصنيف شجرة القرار (CART) — Google Colab

**الهدف:** التنبؤ بـ **فئة ثنائية** (0/1) باستخدام **تصنيف شجرة القرار (CART)** — قواعد **if/else** قابلة للقراءة تقسم البيانات لتعظيم نقاوة الفئة.

| مثال | المتغير(ات) (X) | الهدف (y) | مجموعة البيانات |
|---------|----------------|------------|---------|
| **المثال 1** | Age, EstimatedSalary | Purchased (0/1) | `../Datasets/social_network_ads.csv` |
| **المثال 2** | CreditScore, Income, LoanAmount, YearsEmployed | Approved (0/1) | `../Datasets/loan_approval.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | عدم نقاوة جيني، بنية الشجرة، CART |
| المرحلة 1 | معالجة البيانات | تحميل ← تنظيف ← ترميز ← تقسيم |
| المرحلة 2 | الخوارزمية | تدريب ← تنبؤ ← تصور ← تقييم |

> **التشغيل:** بيئة التشغيل ← تشغيل الكل (أو Ctrl+F9)""",

    1: """---
# دليل الخوارزمية — تصنيف شجرة القرار (CART)

## ما هي CART للتصنيف؟

**CART** = **Classification And Regression Trees** (أشجار التصنيف والانحدار).  
في التصنيف، تقسم الشجرة البيانات **لتعظيم نقاوة الفئة** وتتنبأ **بالفئة الأغلبية** في كل ورقة.

| الجزء | الاسم | الدور |
|------|------|------|
| **الجذر** | عقدة الجذر | أول تقسيم — أعلى الشجرة |
| **العقدة الداخلية** | فرع | قرار: `feature ≤ threshold?` |
| **الورقة** | عقدة طرفية | التنبؤ النهائي = **الفئة الأغلبية** في تلك الورقة |

## كيف يعمل التقسيم (عدم نقاوة جيني)

في كل عقدة، تجرب CART كل متغير وكل عتبة للعثور على التقسيم الذي **يقلل عدم نقاوة جيني بأقصى قدر**.

**عدم نقاوة جيني** (فئتان):

`Gini = 1 − Σ pᵢ²`

حيث **pᵢ** = نسبة الفئة **i** في العقدة.

| جيني | المعنى |
|------|---------|
| **0** | عقدة نقية — جميع العينات من نفس الفئة |
| **0.5** | أقصى عدم نقاوة — مزيج 50/50 (ثنائي) |

**درجة التقسيم:** اختر التقسيم ذا **أكبر تقليل في عدم النقاوة** (خوارزمية جشعة).

## معيار بديل: الإنتروبي

`Entropy = − Σ pᵢ · log₂(pᵢ)`

معامل sklearn: `criterion='gini'` (افتراضي) أو `'entropy'` (مكسب المعلومات).

## قاعدة التنبؤ

1. ابدأ من عقدة **الجذر**.
2. اتبع الفروع: `if X ≤ threshold` ← يسار، وإلا ← يمين.
3. عند الوصول إلى **ورقة**، تنبأ **بالفئة الأغلبية** لعينات التدريب في تلك الورقة.
4. `predict_proba()` = نسب الفئات في الورقة.

## المعاملات الفائقة الرئيسية

| المعامل | الدور | التأثير |
|-----------|------|--------|
| `max_depth` | أقصى عمق للشجرة | ضحلة = أبسط، عميقة = فرط ملاءمة |
| `min_samples_split` | الحد الأدنى للعينات لتقسيم عقدة | أعلى = شجرة أبسط |
| `min_samples_leaf` | الحد الأدنى للعينات في ورقة | أعلى = قرارات أكثر سلاسة |
| `max_leaf_nodes` | الحد الأقصى لعدد الأوراق | يحدّ من تعقيد الشجرة |
| `criterion` | `'gini'` أو `'entropy'` | مقياس جودة التقسيم |
| `random_state` | بذرة عشوائية | بنية شجرة قابلة للتكرار |

## شجرة القرار مقابل مصنّفات أخرى

| | الانحدار اللوجستي | K-NN | شجرة القرار (CART) |
|---|---------------------|------|----------------------|
| الحدود | خطية | محلية / مرنة | **محاذية للمحاور**، مناطق خطوات |
| التحجيم | موصى به | مطلوب | **غير مطلوب** |
| قابلية التفسير | المعاملات | منخفضة | **عالية** (`plot_tree`) |
| فرط الملاءمة | منخفض (بسيط) | يعتمد على K | **مرتفع** إذا كانت عميقة جدًا |

## ما يجب أن يتذكره الطالب

1. تستخدم CART **جيني** (أو الإنتروبي) لاختيار التقسيمات في التصنيف.
2. تنبؤ الورقة = **الفئة الأغلبية** في تلك المنطقة.
3. **لا حاجة لتحجيم المتغيرات** في أشجار القرار.
4. تحكم في **`max_depth`** لتجنب فرط الملاءمة.
5. استخدم **`plot_tree`** لقراءة قواعد if/else الدقيقة.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات لمعالجة البيانات وتصنيف شجرة القرار والتقييم.

**ما تفعله هذه الخلية:** تحمّل numpy و pandas و matplotlib و sklearn DecisionTreeClassifier و plot_tree والمقاييس.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # عمليات وأمصاف رقمية
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء الرسوم البيانية
import seaborn as sns           # تصورات إحصائية (خرائط حرارية)

from sklearn.model_selection import train_test_split       # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.tree import DecisionTreeClassifier, plot_tree # مصنّف CART ومخطط الشجرة
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
# المثال 1: إعلانات الشبكات الاجتماعية — تصنيف شجرة القرار

التنبؤ بما إذا **اشترى** المستخدم من **Age** و **EstimatedSalary** باستخدام CART.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Age` | متغير (X₁) | عمر المستخدم بالسنوات |
| `EstimatedSalary` | متغير (X₂) | الراتب السنوي التقديري بالدولار الأمريكي |
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

FEATURE_COLS = ['Age', 'EstimatedSalary']  # متغيران رقمان للإدخال
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

التحقق من القيم المفقودة، إزالة التكرارات، وتطبيق الاستكمال إذا لزم الأمر.

**ما تفعله هذه الخلية:** تنظف مجموعة البيانات قبل النمذجة.""",

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
    print('تم استكمال القيم المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُكمِّل')

print(f'\\nالصفوف بعد التنظيف: {rows_after}')""",

    12: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة رقمية — يُتخطى الترميز.

**ما تفعله هذه الخلية:** تؤكد أنه لا يلزم ترميز فئوي.""",

    13: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية وُجدت: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة المتغيرات: {FEATURE_COLS}')
    print(f'عمود الهدف: {TARGET_COL} (0/1 مسبقًا)')""",

    14: """## المثال 1 — الخلية 4: تقسيم البيانات

تعريف X و y، ثم تقسيم 80/20 مع **stratify** للحفاظ على نسبة الفئات.

**ما تفعله هذه الخلية:** تنشئ مصفوفات المتغيرات/الهدف وتطبّق train_test_split الطبقي.""",

    15: """# الخطوة 4) تقسيم تدريب-اختبار (طبقي للتصنيف)

X = dataset[FEATURE_COLS].values  # مصفوفة المتغيرات: Age و EstimatedSalary
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

    16: """> **ملاحظة:** تصنيف شجرة القرار **لا يتطلب** تحجيم المتغيرات. الأشجار تقسم على عتبات — المقياس لا يهم.""",

    17: """---
# المرحلة 2: تصنيف شجرة القرار — إعلانات الشبكات الاجتماعية

تدريب CART بعمق محكوم وتصور الشجرة.""",

    18: """## المثال 1 — الخلية 5: تدريب النموذج

تدريب `DecisionTreeClassifier` مع `max_depth=5` و `criterion='gini'`.

**ما تفعله هذه الخلية:** تُلائم نموذج CART وتطبع عمق الشجرة وعدد الأوراق.""",

    19: """# الخطوة 5) تدريب مصنّف شجرة القرار (CART)

classifier = DecisionTreeClassifier(
    criterion='gini',        # التقسيم بتقليل عدم نقاوة جيني
    max_depth=5,             # تحديد العمق لتجنب فرط الملاءمة
    min_samples_leaf=5,      # كل ورقة يجب أن تحتوي على 5 عينات على الأقل
    random_state=42          # بنية شجرة قابلة للتكرار
)

classifier.fit(X_train, y_train)  # بناء الشجرة: إيجاد أفضل تقسيمات جيني على بيانات التدريب

print('تم تدريب شجرة القرار (CART) بنجاح.')
print(f'عمق الشجرة: {classifier.get_depth()}')           # العمق الفعلي للشجرة المبنية
print(f'عدد الأوراق: {classifier.get_n_leaves()}')  # إجمالي عقد الأوراق (مناطق القرار)""",

    20: """## المثال 1 — الخلية 6: التنبؤ

التنبؤ بتسميات الفئات والاحتمالات على مجموعة الاختبار.

**ما تفعله هذه الخلية:** يوجّه كل عينة عبر الشجرة إلى ورقة.""",

    21: """# الخطوة 6) التنبؤ بالفئات والاحتمالات

y_pred_train = classifier.predict(X_train)            # تسميات الفئات لمجموعة التدريب
y_pred_test = classifier.predict(X_test)              # تسميات الفئات لمجموعة الاختبار
y_proba_test = classifier.predict_proba(X_test)[:, 1]  # P(Purchased=1) = نسبة الفئة في الورقة

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    label = 'نعم' if y_pred_test[i] == 1 else 'لا'
    actual = 'نعم' if y_test[i] == 1 else 'لا'
    print(f'  فعلي={actual}، متوقع={label}، P(شراء)={y_proba_test[i]:.2%}')""",

    22: """## المثال 1 — الخلية 7: التصور

رسم **مناطق القرار** ومخطط **الشجرة** (`plot_tree`).

**ما تفعله هذه الخلية:** يعرض حدود قرار محاذية للمحاور وقواعد if/else قابلة للقراءة.""",

    23: """# الخطوة 7) التصور — مناطق القرار + مخطط الشجرة

fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # رسمتان فرعيتان: المناطق والشجرة

# --- يسار: مناطق القرار ---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(grid).reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.25, cmap='RdYlGn')
axes[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='لم يشتري', alpha=0.5, s=40)
axes[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', label='اشترى', alpha=0.5, s=40)
axes[0].set_xlabel('العمر')
axes[0].set_ylabel('الراتب التقديري (USD)')
axes[0].set_title('مناطق قرار CART — إعلانات الشبكات الاجتماعية')
axes[0].legend(fontsize=8)

# --- يمين: مخطط الشجرة ---
plot_tree(
    classifier,                         # نموذج CART المدرب
    feature_names=FEATURE_COLS,         # أسماء على عقد التقسيم
    class_names=['لا', 'نعم'],          # تسميات الفئات في الأوراق
    filled=True,                        # تلوين العقد حسب الفئة الأغلبية
    rounded=True,                       # صناديق عقد مستديرة
    fontsize=8,                         # حجم خط قابل للقراءة
    ax=axes[1]                          # الرسم على الرسم الفرعي الثاني
)
axes[1].set_title('بنية شجرة CART')

plt.tight_layout()
plt.show()""",

    24: """## المثال 1 — الخلية 8: التقييم

التقييم بالدقة والدقة الإيجابية والاسترجاع وF1 ومصفوفة الالتباس.

**ما تفعله هذه الخلية:** تحسب مقاييس التصنيف على مجموعة الاختبار.""",

    25: """# الخطوة 8) التقييم — مقاييس التصنيف

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'Metric': ['الدقة', 'الدقة الإيجابية', 'الاسترجاع', 'مقياس F1'],
    'Value': [acc, prec, rec, f1],
    'Description': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع «نعم»، النسبة الفعلية «نعم»',
        'من الفعلي «نعم»، النسبة المتوقعة بشكل صحيح',
        'توازن بين الدقة الإيجابية والاسترجاع'
    ]
})

display(results.round(4))

cm = confusion_matrix(y_test, y_pred_test)
print('\\nمصفوفة الالتباس (صفوف=فعلي، أعمدة=متوقع):')
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['لا (0)', 'نعم (1)']).plot(ax=ax, cmap='Blues')
plt.title('مصفوفة الالتباس — شجرة القرار (CART)')
plt.tight_layout()
plt.show()

print(f'\\nدقة اختبار المثال 1 = {acc:.4f}')""",

    26: """## لماذا تعمل CART لإعلانات الشبكات الاجتماعية؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **مناطق غير خطية** | سلوك الشراء ينقسم بعتبات العمر والراتب |
| 2 | **لا حاجة للتحجيم** | الأشجار تقارن `Age ≤ 35?` — القيم الخام تعمل |
| 3 | **قواعد قابلة للقراءة** | `plot_tree` يعرض منطق if/else الدقيق للطلاب |
| 4 | **تقسيمات محاذية للمحاور** | كل تقسيم موازٍ لمحور — سهل التفسير |
| 5 | **التحكم في فرط الملاءمة** | `max_depth` و `min_samples_leaf` يحدّان تعقيد الشجرة |

> **الملخص:** تنشئ CART **مناطق مستطيلة** — مثل «Age > 40 AND Salary > 80k → Purchased».""",

    27: """## فهم مقاييس التصنيف — المثال 1

| المقياس | سياق شجرة القرار |
|--------|----------------------|
| **الدقة** | إجمالي التعيينات الصحيحة للأوراق |
| **الدقة الإيجابية** | عندما تتنبأ الشجرة «نعم»، كم مرة تكون صحيحة |
| **الاسترجاع** | من كل «نعم» الفعلية، كم تلتقطها الشجرة |
| **F1** | توازن عندما تكون الفئات غير متوازنة |

> **نصيحة:** إذا كانت دقة التدريب >> دقة الاختبار، فالشجرة **تفرط في الملاءمة** — قلّل `max_depth`.""",

    28: """---
# المثال 2: الموافقة على القروض — تصنيف شجرة القرار

التنبؤ **Approved** مقابل **Rejected** من أربعة متغيرات مالية باستخدام CART.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `CreditScore` | متغير (X₁) | درجة الائتمان (300–850) |
| `Income` | متغير (X₂) | الدخل السنوي بالدولار الأمريكي |
| `LoanAmount` | متغير (X₃) | مبلغ القرض المطلوب بالدولار الأمريكي |
| `YearsEmployed` | متغير (X₄) | سنوات العمل في الوظيفة الحالية |
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

تتضمن مجموعة البيانات قيمًا مفقودة في المتغيرات لعرض `SimpleImputer`.

**ما تفعله هذه الخلية:** تتحقق من القيم الفارغة، تزيل التكرارات، وتستكمل القيم المفقودة.""",

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
    print('تم استكمال قيم المتغيرات المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُكمِّل')

print(f'\\nالصفوف بعد التنظيف: {len(dataset)}')""",

    33: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة رقمية — يُتخطى الترميز.

**ما تفعله هذه الخلية:** تؤكد أنه لا يلزم ترميز فئوي.""",

    34: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية وُجدت: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة المتغيرات: {FEATURE_COLS}')""",

    35: """## المثال 2 — الخلية 4: تقسيم البيانات

تعريف X (4 متغيرات) و y (Approved)، ثم تقسيم 80/20 مع stratify.

**ما تفعله هذه الخلية:** تنشئ مصفوفات المتغيرات/الهدف وتطبّق train_test_split الطبقي.""",

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

تدريب CART بعدة متغيرات — الشجرة تقسم على الائتمان أو الدخل أو القرض أو سنوات العمل في كل عقدة.

**ما تفعله هذه الخلية:** تُلائم النموذج وتُبلغ أهمية المتغيرات.""",

    38: """# الخطوة 5) تدريب مصنّف شجرة القرار (CART)

classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

classifier.fit(X_train, y_train)

print('تم تدريب شجرة القرار (CART) بنجاح.')
print(f'عمق الشجرة: {classifier.get_depth()}')
print(f'عدد الأوراق: {classifier.get_n_leaves()}')

print('\\nأهمية المتغيرات:')
for name, imp in zip(FEATURE_COLS, classifier.feature_importances_):
    print(f'  {name:15s} -> {imp:.4f}')""",

    39: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بالموافقة على القرض على مجموعة الاختبار.

**ما تفعله هذه الخلية:** يولّد التنبؤات بتوجيه العينات عبر الشجرة.""",

    40: """# الخطوة 6) التنبؤ

y_pred_test = classifier.predict(X_test)
y_proba_test = classifier.predict_proba(X_test)[:, 1]

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    actual = 'موافق' if y_test[i] == 1 else 'مرفوض'
    pred = 'موافق' if y_pred_test[i] == 1 else 'مرفوض'
    print(f'  فعلي={actual}، متوقع={pred}، P(موافقة)={y_proba_test[i]:.2%}')""",

    41: """## المثال 2 — الخلية 7: التصور

رسم **أهمية المتغيرات** و**مصفوفة الالتباس**.

**ما تفعله هذه الخلية:** يعرض أي المتغيرات الأكثر أهمية لقرارات الموافقة.""",

    42: """# الخطوة 7) التصور — أهمية المتغيرات + مصفوفة الالتباس

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(FEATURE_COLS, classifier.feature_importances_, color='teal')
axes[0].set_xlabel('الأهمية')
axes[0].set_title('أهمية المتغيرات — الموافقة على القروض (CART)')

cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['مرفوض', 'موافق']).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('مصفوفة الالتباس — شجرة القرار (CART)')

plt.tight_layout()
plt.show()""",

    43: """## المثال 2 — الخلية 8: التقييم

تقرير تصنيف كامل بالدقة والدقة الإيجابية والاسترجاع وF1.

**ما تفعله هذه الخلية:** تحسب وتعرض مقاييس التقييم.""",

    44: """# الخطوة 8) التقييم

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

results = pd.DataFrame({
    'Metric': ['الدقة', 'الدقة الإيجابية', 'الاسترجاع', 'مقياس F1'],
    'Value': [acc, prec, rec, f1],
    'Description': [
        'نسبة التنبؤات الصحيحة',
        'من المتوقع «موافق»، النسبة الفعلية الموافقة',
        'من الفعلي «موافق»، النسبة المتوقعة بشكل صحيح',
        'توازن بين الدقة الإيجابية والاسترجاع'
    ]
})

display(results.round(4))

print('\\nتقرير تصنيف مفصّل:')
print(classification_report(y_test, y_pred_test, target_names=['مرفوض (0)', 'موافق (1)']))

print(f'\\nدقة اختبار المثال 2 = {acc:.4f}')""",

    45: """## لماذا تعمل CART جيدًا للموافقة على القروض؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **متغيرات متعددة** | CART تقسم تلقائيًا على الائتمان والدخل والقرض وسنوات العمل |
| 2 | **قواعد غير خطية** | مثل CreditScore > 620 AND LoanAmount < 100k → Approved |
| 3 | **أهمية المتغيرات** | تُظهر أي متغير يساهم أكثر في التقسيمات |
| 4 | **لا حاجة للتحجيم** | درجة الائتمان والدخل الخام يعملان مباشرة |
| 5 | **دقة عالية ممكنة** | مع ضبط `max_depth`، تلائم CART أنماط موافقة معقدة |

> **المقارنة:** CART مقابل الانحدار اللوجستي — الأشجار تلتقط قواعد **غير خطية** دون هندسة متغيرات يدوية.""",

    46: """## فهم مقاييس التصنيف — المثال 2

| السيناريو | المقياس المركز |
|----------|--------------|
| **الموافقة الخاطئة مكلفة** | **الدقة الإيجابية** |
| **الرفض الخاطئ مكلف** | **الاسترجاع** |
| **تكلفة متوازنة** | **الدقة** أو **F1** |

> **فحص فرط الملاءمة:** قارن `accuracy_score(y_train, y_pred_train)` مع دقة الاختبار — فجوة كبيرة تعني تقليل `max_depth`.""",
}
