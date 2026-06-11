CELLS = {
    0: """# أقرب الجيران (K-NN) — التصنيف (Google Colab)

**الهدف:** التنبؤ بـ **فئة ثنائية** (0/1) باستخدام **أقرب الجيران (K-NN)** — مصنّف **كسول** يعتمد على **المسافة** ويصوّت بين **أقرب K** نقاط تدريب.

| مثال | الميزة/الميزات (X) | الهدف (y) | مجموعة البيانات |
|---------|----------------|------------|---------|
| **المثال 1** | Age, EstimatedSalary | Purchased (0/1) | `../Datasets/social_network_ads.csv` |
| **المثال 2** | CreditScore, Income, LoanAmount, YearsEmployed | Approved (0/1) | `../Datasets/loan_approval.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | المسافة، K، التصويت، اختيار K |
| المرحلة 1 | معالجة البيانات | التحميل ← التنظيف ← الترميز ← التقسيم |
| المرحلة 2 | الخوارزمية | التدريب ← التنبؤ ← التصور ← التقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)""",

    1: """---
# دليل الخوارزمية — أقرب الجيران (K-NN)

## ما هو K-NN؟

**K-NN** يصنّف نقطة جديدة بالنظر إلى **أقرب K جيران** في بيانات التدريب وأخذ **تصويت بالأغلبية**.

| الخاصية | التفاصيل |
|----------|--------|
| النوع | **متعلّم كسول** — لا توجد مرحلة تدريب صريحة |
| قاعدة القرار | **تصويت بالأغلبية** بين أقرب K نقاط |
| المسافة | عادة **المسافة الإقليدية** |
| المخرج | تسمية الفئة (+ احتمال اختياري من أصوات الجيران) |

## كيف يعمل K-NN (خطوة بخطوة)

| الخطوة | الإجراء |
|------|--------|
| 1 | تخزين كل بيانات التدريب (لا تُتعلَّم معاملات نموذج) |
| 2 | لنقطة جديدة **x**، حساب المسافة إلى **كل** نقطة تدريب |
| 3 | اختيار **K** النقاط ذات أصغر مسافة |
| 4 | **التصويت:** التنبؤ بالفئة الأكثر شيوعاً بين هؤلاء الجيران |
| 5 | (كسر التعادل) استخدام تصويت مرجّح بالمسافة إذا `weights='distance'` |

## المسافة الإقليدية (ميزتان)

`d = √[(x₁ − x₁')² + (x₂ − x₂')²]`

الميزات ذات **مقياس أكبر** تهيمن على المسافة → **StandardScaler مطلوب**.

## اختيار K

| قيمة K | التأثير |
|---------|--------|
| **K = 1** | مرونة عالية — حساس للضجيج (**فرط التخصّص**) |
| **K صغير** | حدود قرار معقدة |
| **K كبير** | حدود أنعم — قد يحدث **نقص التخصّص** |
| **K زوجي** | تعادل محتمل — يُفضَّل **K فردي** (مثل 3، 5، 7) |

استخدم **التحقق المتقاطع** لاختيار K الذي يزيد دقة التحقق.

## المعاملات الفائقة الرئيسية (sklearn)

| المعامل | الدور |
|-----------|------|
| `n_neighbors` | **K** — عدد الجيران |
| `metric` | مقياس المسافة (`'euclidean'`، `'manhattan'`، ...) |
| `weights` | `'uniform'` (صوت متساوٍ) أو `'distance'` (الأقرب = وزن أكبر) |
| `n_jobs` | حساب المسافة بالتوازي (`-1` = كل الأنوية) |

## K-NN مقابل الانحدار اللوجستي

| | الانحدار اللوجستي | K-NN |
|---|---------------------|------|
| التدريب | يتعلّم المعاملات | **يخزّن** كل البيانات |
| الحدود | خطية (log-odds) | **غير خطية**، مرنة |
| التحجيم | موصى به | **مطلوب** |
| السرعة (التنبؤ) | سريع | أبطأ — يمسح كل نقاط التدريب |
| القابلية للتفسير | المعاملات | أصعب — يعتمد على الجيران |

## ما يجب أن يتذكّره الطالب

1. K-NN = **تصويت بالأغلبية** لـ **أقرب K** نقاط تدريب.
2. **احجِم الميزات دائماً** قبل K-NN (يعتمد على المسافة).
3. **اختر K** بعناية — استخدم التحقق المتقاطع.
4. **K فردي** يتجنّب تعادل الأصوات في التصنيف الثنائي.
5. K-NN **متعلّم كسول** — "التدريب" يخزّن مجموعة البيانات فقط.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات للمعالجة المسبقة وتصنيف K-NN والتقييم.

**ما تفعله هذه الخلية:** تحمّل numpy و pandas و matplotlib و KNeighborsClassifier من sklearn والمقاييس.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء المخططات والرسوم
import seaborn as sns           # تصورات إحصائية (خرائط حرارية)

from sklearn.model_selection import train_test_split, cross_val_score  # التقسيم والتحقق المتقاطع
from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.preprocessing import StandardScaler           # تحجيم الميزات (مطلوب لـ K-NN)
from sklearn.pipeline import Pipeline                      # ربط المحوّل + المصنّف
from sklearn.neighbors import KNeighborsClassifier         # مصنّف أقرب الجيران
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
# المثال 1: إعلانات الشبكات الاجتماعية — تصنيف K-NN

التنبؤ بما إذا **اشترى** المستخدم من **Age** و **EstimatedSalary** باستخدام K-NN.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Age` | ميزة (X₁) | عمر المستخدم بالسنوات |
| `EstimatedSalary` | ميزة (X₂) | الراتب السنوي التقديري بالدولار الأمريكي |
| `Purchased` | الهدف (y) | 0 = لا، 1 = نعم |

**الملف:** `../Datasets/social_network_ads.csv`""",

    7: """---
# المرحلة 1: معالجة البيانات

تحضير البيانات قبل تدريب النموذج — نفس القالب يُعاد استخدامه لخوارزميات أخرى.""",

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

التحقق من القيم المفقودة، إزالة التكرارات، وتطبيق الاستبدال عند الحاجة.

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

**ما تفعله هذه الخلية:** تؤكد أنه لا حاجة لترميز فئوي.""",

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

**ما تفعله هذه الخلية:** تنشئ مصفوفات الميزة/الهدف وتطبّق train_test_split طبقي.""",

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

    16: """> **ملاحظة:** K-NN يستخدم **المسافة** — **StandardScaler إلزامي** حتى يساهم العمر والراتب بشكل عادل.""",

    17: """---
# المرحلة 2: K-NN — إعلانات الشبكات الاجتماعية

اختر **K** بالتحقق المتقاطع، ثم درّب وقيّم.""",

    18: """## المثال 1 — الخلية 5: اختيار K وتدريب النموذج

جرّب K من 1 إلى 20 مع **تحقق متقاطع 5-fold** على مجموعة التدريب، ثم درّب بأفضل K.

**ما تفعله هذه الخلية:** تختار K الأمثل وتُلائِم خط أنابيب K-NN.""",

    19: """# الخطوة 5) اختيار K بالتحقق المتقاطع، ثم تدريب K-NN

K_range = range(1, 21)           # اختبار K من 1 إلى 20
cv_scores = []                   # تخزين متوسط دقة CV لكل K

for k in K_range:
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),                    # تحجيم الميزات قبل حساب المسافة
        ('model', KNeighborsClassifier(n_neighbors=k))   # K-NN مع K جيران
    ])
    scores = cross_val_score(knn_pipe, X_train, y_train, cv=5, scoring='accuracy')  # CV 5-fold
    cv_scores.append(scores.mean())  # متوسط الدقة عبر الطيات

best_k = list(K_range)[np.argmax(cv_scores)]  # K بأعلى دقة CV
print(f'أفضل K من التحقق المتقاطع: {best_k}')
print(f'أفضل دقة CV: {max(cv_scores):.4f}')

# رسم K مقابل الدقة
plt.figure(figsize=(9, 4))
plt.plot(list(K_range), cv_scores, marker='o', color='steelblue')
plt.axvline(best_k, color='red', linestyle='--', label=f'أفضل K = {best_k}')
plt.xlabel('K (عدد الجيران)')
plt.ylabel('دقة التحقق المتقاطع')
plt.title('اختيار K — إعلانات الشبكات الاجتماعية')
plt.xticks(list(K_range))
plt.legend()
plt.tight_layout()
plt.show()

# تدريب النموذج النهائي بأفضل K
classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1))
])
classifier.fit(X_train, y_train)  # تخزين بيانات التدريب المُحجَّمة للبحث عن الجيران

print(f'\\nتم تدريب K-NN بـ K = {best_k} (متعلّم كسول — يخزّن {len(X_train)} نقطة تدريب)')""",

    20: """## المثال 1 — الخلية 6: التنبؤ

التنبؤ بتسميات الفئات على مجموعة الاختبار. يمكن لـ K-NN أيضاً إخراج احتمالات من أصوات الجيران.

**ما تفعله هذه الخلية:** تستخدم `predict()` و `predict_proba()` على بيانات الاختبار.""",

    21: """# الخطوة 6) التنبؤ بالفئات والاحتمالات

y_pred_train = classifier.predict(X_train)            # تسميات الفئات لمجموعة التدريب
y_pred_test = classifier.predict(X_test)              # تسميات الفئات لمجموعة الاختبار
y_proba_test = classifier.predict_proba(X_test)[:, 1]  # نسبة جيران K الذين صوّتوا للفئة 1

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    label = 'نعم' if y_pred_test[i] == 1 else 'لا'
    actual = 'نعم' if y_test[i] == 1 else 'لا'
    print(f'  الفعلي={actual}، المتوقع={label}، تصويت الجيران P(نعم)={y_proba_test[i]:.2%}')""",

    22: """## المثال 1 — الخلية 7: التصور

رسم نقاط التدريب و**حدود قرار K-NN** (مناطق غير خطية).

**ما تفعله هذه الخلية:** تُظهر كيف يقسّم K-NN مستوى العمر–الراتب.""",

    23: """# الخطوة 7) التصور — نقاط مبعثرة + حدود قرار K-NN

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
plt.xlabel('Age')
plt.ylabel('Estimated Salary (USD)')
plt.title(f'حدود قرار K-NN (K={best_k}) — إعلانات الشبكات الاجتماعية')
plt.legend()
plt.tight_layout()
plt.show()""",

    24: """## المثال 1 — الخلية 8: التقييم

التقييم بالدقة والدقة الإيجابية والاسترجاع وF1 ومصفوفة الالتباه.

**ما تفعله هذه الخلية:** تحسب مقاييس التصنيف على مجموعة الاختبار.""",

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
        'من المتوقع نعم، نسبة الصحيح فعلاً',
        'من الفعلي نعم، نسبة ما تمّ التقاطه',
        'توازن بين الدقة الإيجابية والاسترجاع'
    ]
})

display(results.round(4))

cm = confusion_matrix(y_test, y_pred_test)
print('\\nمصفوفة الالتباه (الصفوف=الفعلي، الأعمدة=المتوقع):')
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['لا (0)', 'نعم (1)']).plot(ax=ax, cmap='Blues')
plt.title(f'مصفوفة الالتباه — K-NN (K={best_k})')
plt.tight_layout()
plt.show()

print(f'\\nدقة اختبار المثال 1 = {acc:.4f}')""",

    26: """## لماذا يعمل K-NN لإعلانات الشبكات الاجتماعية؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **حدود غير خطية** | أنماط الشراء تحتاج مناطق مرنة — K-NN يتكيّف محلياً |
| 2 | **ميزتان** | سهل تصور الجيران في فضاء العمر–الراتب |
| 3 | **المسافة + التحجيم** | نطاق الراتب كبير — StandardScaler يجعل العمر والراتب متكافئين |
| 4 | **اختيار K** | التحقق المتقاطع يمنع فرط التخصّص (K=1) ونقص التخصّص (K كبير جداً) |
| 5 | **لا افتراضات** | على عكس الانحدار اللوجستي، K-NN لا يفترض حدوداً خطية |

> **الخلاصة:** المستخدمون المتشابهون في العمر والراتب يميلون لمشاركة سلوك الشراء — K-NN يجدهم بالمسافة.""",

    27: """## فهم مقاييس التصنيف — المثال 1

| المقياس | سياق K-NN |
|--------|--------------|
| **Accuracy** | إجمالي الأصوات الصحيحة |
| **Precision** | عندما يتنبّأ K-NN بنعم، كم مرة يكون صحيحاً |
| **Recall** | من كل الفعلي نعم، كم يلتقط K-NN |
| **F1** | توازن عندما تكون الفئات غير متوازنة (~42% اشترى) |

> **نصيحة:** قارن حدود القرار — K-NN مقابل الانحدار اللوجستي على نفس مجموعة البيانات.""",

    28: """---
# المثال 2: الموافقة على القروض — تصنيف K-NN

التنبؤ بـ **Approved** مقابل **Rejected** من أربع ميزات مالية باستخدام K-NN.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `CreditScore` | ميزة (X₁) | درجة الائتمان (300–850) |
| `Income` | ميزة (X₂) | الدخل السنوي بالدولار الأمريكي |
| `LoanAmount` | ميزة (X₃) | مبلغ القرض المطلوب بالدولار الأمريكي |
| `YearsEmployed` | ميزة (X₄) | سنوات العمل في الوظيفة الحالية |
| `Approved` | الهدف (y) | 1 = Approved، 0 = Rejected |

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

تتضمن هذه المجموعة قيماً مفقودة في الميزات لعرض `SimpleImputer`.

**ما تفعله هذه الخلية:** تتحقق من القيم الفارغة، تزيل التكرارات، وتستبدل القيم المفقودة.""",

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

**ما تفعله هذه الخلية:** تؤكد أنه لا حاجة لترميز فئوي.""",

    34: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات: {FEATURE_COLS}')""",

    35: """## المثال 2 — الخلية 4: تقسيم البيانات

تحديد X (4 ميزات) و y (Approved)، ثم التقسيم 80/20 مع stratify.

**ما تفعله هذه الخلية:** تنشئ مصفوفات الميزة/الهدف وتطبّق train_test_split طبقي.""",

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

    37: """## المثال 2 — الخلية 5: اختيار K وتدريب النموذج

اختر K عبر التحقق المتقاطع على ميزات مُحجَّمة بـ 4 أبعاد.

**ما تفعله هذه الخلية:** تجد أفضل K وتُلائِم K-NN على بيانات الموافقة على القروض.""",

    38: """# الخطوة 5) اختيار K وتدريب K-NN

K_range = range(1, 21)
cv_scores = []

for k in K_range:
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(knn_pipe, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = list(K_range)[np.argmax(cv_scores)]
print(f'أفضل K من التحقق المتقاطع: {best_k}')
print(f'أفضل دقة CV: {max(cv_scores):.4f}')

classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1))
])
classifier.fit(X_train, y_train)

print(f'تم تدريب K-NN بـ K = {best_k}')""",

    39: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بالموافقة على القرض على مجموعة الاختبار.

**ما تفعله هذه الخلية:** يولّد تنبؤات الفئات واحتمالات تصويت الجيران.""",

    40: """# الخطوة 6) التنبؤ

y_pred_test = classifier.predict(X_test)
y_proba_test = classifier.predict_proba(X_test)[:, 1]

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    actual = 'موافق' if y_test[i] == 1 else 'مرفوض'
    pred = 'موافق' if y_pred_test[i] == 1 else 'مرفوض'
    print(f'  الفعلي={actual}، المتوقع={pred}، تصويت الجيران P(موافق)={y_proba_test[i]:.2%}')""",

    41: """## المثال 2 — الخلية 7: التصور

رسم **K مقابل دقة CV** و**مصفوفة الالتباه** على مجموعة الاختبار.

**ما تفعله هذه الخلية:** تُظهر اختيار المعامل الفائق والأداء النهائي.""",

    42: """# الخطوة 7) التصور — اختيار K + مصفوفة الالتباه

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_range), cv_scores, marker='o', color='steelblue')
axes[0].axvline(best_k, color='red', linestyle='--', label=f'أفضل K = {best_k}')
axes[0].set_xlabel('K (عدد الجيران)')
axes[0].set_ylabel('دقة التحقق المتقاطع')
axes[0].set_title('اختيار K — الموافقة على القروض')
axes[0].legend()

cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['مرفوض', 'موافق']).plot(ax=axes[1], cmap='Blues')
axes[1].set_title(f'مصفوفة الالتباه — K-NN (K={best_k})')

plt.tight_layout()
plt.show()""",

    43: """## المثال 2 — الخلية 8: التقييم

تقرير تصنيف كامل مع الدقة والدقة الإيجابية والاسترجاع وF1.

**ما تفعله هذه الخلية:** تحسب وتعرض مقاييس التقييم.""",

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
        'من المتوقع موافق، نسبة الموافقة الفعلية',
        'من الفعلي موافق، نسبة ما تمّ التنبؤ به صحيحاً',
        'توازن بين الدقة الإيجابية والاسترجاع'
    ]
})

display(results.round(4))

print('\\nتقرير تصنيف مفصّل:')
print(classification_report(y_test, y_pred_test, target_names=['مرفوض (0)', 'موافق (1)']))

print(f'\\nدقة اختبار المثال 2 = {acc:.4f}')""",

    45: """## لماذا يعمل K-NN للموافقة على القروض؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **مسافة متعددة الميزات** | الائتمان والدخل والقرض والتوظيف يحدّدون المتقدّمين "المتشابهين" |
| 2 | **لا افتراض خطي** | قواعد الموافقة قد تكون غير خطية — K-NN يتكيّف محلياً |
| 3 | **التحجيم حاسم** | الدخل (~80k) مقابل YearsEmployed (~10) — يجب التحجيم قبل المسافة |
| 4 | **ضبط K** | التحقق المتقاطع يختار K الذي يعمّم في فضاء 4D |
| 5 | **فئات متوازنة** | تقسيم 50/50 — الدقة مقياس عادل |

> **قارن:** K-NN مقابل الانحدار اللوجستي على نفس بيانات القروض — دقة متشابهة، منطق قرار مختلف.""",

    46: """## فهم مقاييس التصنيف — المثال 2

| السيناريو | المقياس المركّز |
|----------|--------------|
| **الموافقة الخاطئة مكلفة** | **Precision** |
| **الرفض الخاطئ مكلف** | **Recall** |
| **متوازن** | **Accuracy** أو **F1** |

> K-NN في **أبعاد عالية** (ميزات كثيرة) قد يعاني (لعنة الأبعاد) — 4 ميزات قابلة للإدارة.""",
}
