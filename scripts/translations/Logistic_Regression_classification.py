# Arabic translations for Logistic_Regression_classification.ipynb
CELLS = {
    0: """# الانحدار اللوجستي — التصنيف (Google Colab)

**الهدف:** التنبؤ بـ **فئة ثنائية** (0/1) باستخدام **الانحدار اللوجستي** — يُخرج **احتمالات** عبر دالة **السigmoid**.

| المثال | الميزة/الميزات (X) | الهدف (y) | مجموعة البيانات |
|---------|----------------|------------|---------|
| **المثال 1** | Age, EstimatedSalary | Purchased (0/1) | `../Datasets/social_network_ads.csv` |
| **المثال 2** | CreditScore, Income, LoanAmount, YearsEmployed | Approved (0/1) | `../Datasets/loan_approval.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | السigmoid، اللوغ-أودز، العتبة |
| المرحلة 1 | معالجة البيانات | التحميل ← التنظيف ← الترميز ← التقسيم |
| المرحلة 2 | الخوارزمية | التدريب ← التنبؤ ← التصور ← التقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)""",

    1: """---
# دليل الخوارزمية — الانحدار اللوجستي (التصنيف)

## ما هو التصنيف؟

**التصنيف** يتنبأ **بتسمية فئة منفصلة** (مثل Purchased = 0 أو 1)، وليس برقم متصل.

| المهمة | المخرج | مثال على خوارزمية |
|------|--------|-------------------|
| **الانحدار** | رقم متصل | الانحدار الخطي |
| **التصنيف** | تسمية فئة (0/1) | **الانحدار اللوجستي** |

## لماذا لا نستخدم الانحدار الخطي للتصنيف؟

| المشكلة | الشرح |
|---------|-------------|
| مخرجات غير صالحة | الانحدار الخطي قد يتنبأ بقيم **< 0** أو **> 1** |
| هدف خاطئ | يقلّل خطأ المربعات — غير مناسب لتسميات الفئات |
| لا احتمال | لا يُخرج احتمالات صالحة بشكل طبيعي |

## فكرة الانحدار اللوجستي

1. حساب **درجة خطية**: `z = β₀ + β₁x₁ + β₂x₂ + ...`
2. تمريرها عبر **السigmoid** للحصول على احتمال: `P(y=1|x) = σ(z) = 1 / (1 + e⁻ᶻ)`
3. تطبيق **عتبة** (افتراضي **0.5**): إذا P ≥ 0.5 → الفئة **1**، وإلا الفئة **0**

## دالة السigmoid

| z | σ(z) |
|---|------|
| موجب كبير | → 1 |
| 0 | → 0.5 |
| سالب كبير | → 0 |

**خاصية أساسية:** المخرج دائماً بين **0 و 1** — احتمال صالح.

## اللوغ-أودز (Logit)

`log(P / (1−P)) = β₀ + β₁x₁ + β₂x₂ + ...`

الانحدار اللوجستي يُنمذج **اللوغ-أودز** كدالة خطية من الميزات.

## المعاملات الفائقة الرئيسية (sklearn)

| المعامل | الدور |
|-----------|------|
| `C` | معكوس قوة التنظيم (C أصغر = عقوبة أقوى) |
| `max_iter` | أقصى عدد تكرارات لتقارب المحلّل |
| `random_state` | نتائج قابلة للتكرار |
| `class_weight` | معالجة الفئات غير المتوازنة (`'balanced'` عند الحاجة) |

## تحجيم الميزات

استخدم **`StandardScaler`** عندما تختلف مقاييس الميزات (العمر مقابل الراتب، الائتمان مقابل الدخل). التحجيم يساعد المحسّن ويجعل المعاملات قابلة للمقارنة.

## مقاييس التصنيف

| المقياس | الصيغة / المعنى |
|--------|-------------------|
| **الدقة (Accuracy)** | (TP + TN) / الإجمالي |
| **الدقة الإيجابية (Precision)** | TP / (TP + FP) — من المتنبأ بـ Yes، كم صحيح |
| **الاستدعاء (Recall)** | TP / (TP + FN) — من Yes الفعلي، كم اكتشفنا |
| **مقياس F1** | المتوسط التوافقي للدقة الإيجابية والاستدعاء |

## مصفوفة الارتباك

|  | تنبؤ 0 | تنبؤ 1 |
|--|--------|--------|
| **فعلي 0** | TN | FP |
| **فعلي 1** | FN | TP |

## ما يجب أن يتذكره الطالب

1. الانحدار اللوجستي يُخرج **احتمالات** في [0, 1].
2. الفئة النهائية تستخدم **عتبة** (عادة 0.5).
3. استخدم **مقاييس التصنيف** — وليس R² أو RMSE.
4. **حجّم الميزات** عندما تختلف الوحدات بشكل كبير.
5. استخدم **`stratify=y`** في train_test_split للحفاظ على توازن الفئات.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

Google Colab يتضمن عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبيت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات لمعالجة البيانات والانحدار اللوجستي وتقييم التصنيف.

**ما تفعله هذه الخلية:** تحميل numpy و pandas و matplotlib ومصنّفات sklearn ومقاييس التصنيف.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء المخططات والرسوم
import seaborn as sns           # تصورات إحصائية (خرائط حرارية)
from matplotlib.colors import ListedColormap  # ألوان مناطق القرار

from sklearn.model_selection import train_test_split       # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.preprocessing import StandardScaler           # تحجيم الميزات
from sklearn.pipeline import Pipeline                      # ربط المحوّل + المصنّف
from sklearn.linear_model import LogisticRegression        # مصنّف الانحدار اللوجستي
from sklearn.metrics import (                              # مقاييس التصنيف
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)  # حجم المخطط الافتراضي: عرض=10، ارتفاع=6 بوصة
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لتقسيمات قابلة للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح""",

    6: """---
# المثال 1: إعلانات الشبكات الاجتماعية — الانحدار اللوجستي

التنبؤ بما إذا كان المستخدم قد **Purchased** منتجاً من **Age** و **EstimatedSalary**.

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

**ما تفعله هذه الخلية:** قراءة `../Datasets/social_network_ads.csv` وعرض إحصائيات أساسية.""",

    9: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/social_network_ads.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['Age', 'EstimatedSalary']  # ميزتان عدديتان للإدخال
TARGET_COL = 'Purchased'                    # هدف ثنائي: 0 = لا، 1 = نعم

print('أول 5 صفوف:')          # طباعة تسمية للجدول أدناه
display(dataset.head())         # عرض أول 5 صفوف لفحص البيانات

print('\\nمعلومات مجموعة البيانات:')       # طباعة تسمية لأنواع الأعمدة وعدد القيم الفارغة
dataset.info()                  # عرض أسماء الأعمدة وأنواع البيانات وعدد القيم غير الفارغة

print('\\nملخص إحصائي:')  # طباعة تسمية للإحصائيات العددية
display(dataset.describe())       # عرض العدد والمتوسط والانحراف المعياري والحد الأدنى والأقصى والربيعيات

print(f'\\nتوازن الفئات ({TARGET_COL}):')  # عرض عدد 0 مقابل 1
print(dataset[TARGET_COL].value_counts())   # العدد لكل فئة

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')  # إجمالي الصفوف والأعمدة""",

    10: """## المثال 1 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

فحص القيم المفقودة وإزالة التكرارات وتطبيق الاستبدال عند الحاجة.

**ما تفعله هذه الخلية:** تنظيف مجموعة البيانات قبل النمذجة.""",

    11: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')  # طباعة تسمية
print(dataset.isnull().sum())        # عد قيم NaN في كل عمود

rows_before = len(dataset)                              # تخزين عدد الصفوف قبل التنظيف
dataset = dataset.drop_duplicates().reset_index(drop=True)  # إزالة الصفوف المكررة
rows_after = len(dataset)                               # تخزين عدد الصفوف بعد إزالة التكرار
print(f'\\nالتكرارات المحذوفة: {rows_before - rows_after}')  # عرض عدد التكرارات

num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()  # الأعمدة العددية فقط
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # ملء NaN بمتوسط العمود
if dataset.isnull().sum().sum() > 0:               # إذا وُجدت قيم مفقودة
    dataset[num_cols] = imputer.fit_transform(dataset[num_cols])  # استبدال الأعمدة العددية
    print('تم استبدال القيم المفقودة بالمتوسط')      # تأكيد الاستبدال
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المستبدِل')  # تخطي عند اكتمال البيانات

print(f'\\nالصفوف بعد التنظيف: {rows_after}')    # العدد النهائي للصفوف""",

    12: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة عددية — يُتخطى الترميز.

**ما تفعله هذه الخلية:** تأكيد عدم الحاجة لترميز فئوي.""",

    13: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()  # إيجاد أعمدة النص
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات: {FEATURE_COLS}')
    print(f'عمود الهدف: {TARGET_COL} (0/1 مسبقاً)')

""",

    14: """## المثال 1 — الخلية 4: تقسيم البيانات

تعريف X (الميزات) و y (Purchased)، ثم تقسيم 80/20 مع **stratify** للحفاظ على نسبة الفئات.

**ما تفعله هذه الخلية:** إنشاء مصفوفات الميزات/الهدف وتطبيق train_test_split الطبقي.""",

    15: """# الخطوة 4) تقسيم تدريب-اختبار (طبقي للتصنيف)

X = dataset[FEATURE_COLS].values  # مصفوفة الميزات: Age و EstimatedSalary
y = dataset[TARGET_COL].values    # متجه الهدف: Purchased (0 أو 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                  # البيانات للتقسيم
    test_size=0.2,         # 20% اختبار، 80% تدريب
    random_state=42,       # تقسيم قابل للتكرار
    stratify=y             # الحفاظ على نفس نسبة الفئات في التدريب والاختبار
)

print(f'شكل X_train: {X_train.shape}')  # شكل ميزات التدريب
print(f'شكل X_test:  {X_test.shape}')   # شكل ميزات الاختبار
print(f'عدد فئات y_train: {np.bincount(y_train)}')  # [عدد_0، عدد_1] في التدريب
print(f'عدد فئات y_test:  {np.bincount(y_test)}')   # [عدد_0، عدد_1] في الاختبار""",

    16: """> **ملاحظة:** يعمل الانحدار اللوجستي أفضل مع **ميزات محجّمة** عندما يختلف العمر والراتب في الوحدات — نستخدم `StandardScaler` في خط الأنابيب.""",

    17: """---
# المرحلة 2: الانحدار اللوجستي — إعلانات الشبكات الاجتماعية

تدريب مصنّف للتنبؤ بـ Purchased من Age و EstimatedSalary.""",

    18: """## المثال 1 — الخلية 5: تدريب النموذج

بناء **Pipeline**: StandardScaler → LogisticRegression.

**ما تفعله هذه الخلية:** ملاءمة المصنّف وطباعة المعاملات المُتعلَّمة.""",

    19: """# الخطوة 5) تدريب الانحدار اللوجستي

classifier = Pipeline([
    ('scaler', StandardScaler()),  # تحجيم العمر والراتب إلى mean=0، std=1
    ('model', LogisticRegression(max_iter=1000, random_state=42))  # مصنّف لوجستي ثنائي
])

classifier.fit(X_train, y_train)  # تعلّم المعاملات على بيانات التدريب

model = classifier.named_steps['model']  # استخراج LogisticRegression المُلائَم
print('تم تدريب الانحدار اللوجستي بنجاح.')
print(f'المقطع (β₀): {model.intercept_[0]:.4f}')  # حدود الانحياز
for name, coef in zip(FEATURE_COLS, model.coef_[0]):
    print(f'  {name:18s} -> β = {coef:.4f}')  # معامل لكل ميزة""",

    20: """## المثال 1 — الخلية 6: التنبؤ

التنبؤ بتسميات الفئات و**الاحتمالات** على مجموعة الاختبار.

**ما تفعله هذه الخلية:** استخدام `predict()` للفئات و `predict_proba()` لـ P(Purchased=1).""",

    21: """# الخطوة 6) التنبؤ بالفئات والاحتمالات

y_pred_train = classifier.predict(X_train)            # تسميات الفئات (0 أو 1) لمجموعة التدريب
y_pred_test = classifier.predict(X_test)              # تسميات الفئات لمجموعة الاختبار
y_proba_test = classifier.predict_proba(X_test)[:, 1]  # P(Purchased=1) لكل عينة اختبار

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    label = 'نعم' if y_pred_test[i] == 1 else 'لا'
    actual = 'نعم' if y_test[i] == 1 else 'لا'
    print(f'  فعلي={actual}، متنبأ={label}، P(شراء)={y_proba_test[i]:.2%}')""",

    22: """## المثال 1 — الخلية 7: التصور

رسم نقاط البيانات ملونة حسب الفئة و**حد القرار** (حيث P = 0.5).

**ما تفعله هذه الخلية:** عرض كيف يفصل الانحدار اللوجستي بين Purchased و Not Purchased.""",

    23: """# الخطوة 7) التصور — مبعثر + حد القرار

# --- بناء شبكة mesh لمناطق القرار ---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1          # نطاق العمر مع هامش
y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000   # نطاق الراتب مع هامش
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),      # 300 نقطة شبكة للعمر
    np.linspace(y_min, y_max, 300)       # 300 نقطة شبكة للراتب
)
grid = np.c_[xx.ravel(), yy.ravel()]     # تسطيح الشبكة إلى (n_points, 2)
Z = classifier.predict_proba(grid)[:, 1]  # P(Purchased=1) عند كل نقطة شبكة
Z = Z.reshape(xx.shape)                   # إعادة التشكيل إلى 2D لرسم contour

# --- الرسم ---
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlGn', levels=20)  # خريطة حرارية للاحتمال
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)  # حد القرار عند 0.5
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='red', label='لم يُشترَ (تدريب)', alpha=0.6)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', label='تم الشراء (تدريب)', alpha=0.6)
plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', edgecolors='k', s=80, label='نقاط الاختبار')
plt.xlabel('العمر')
plt.ylabel('الراتب التقديري (USD)')
plt.title('حد القرار — الانحدار اللوجستي — إعلانات الشبكات الاجتماعية')
plt.legend()
plt.tight_layout()
plt.show()""",

    24: """## المثال 1 — الخلية 8: التقييم

التقييم بـ **الدقة** و**الدقة الإيجابية** و**الاستدعاء** و**F1** و**مصفوفة الارتباك**.

**ما تفعله هذه الخلية:** حساب مقاييس التصنيف على مجموعة الاختبار.""",

    25: """# الخطوة 8) التقييم — مقاييس التصنيف

acc = accuracy_score(y_test, y_pred_test)                    # إجمالي التنبؤات الصحيحة
prec = precision_score(y_test, y_pred_test, zero_division=0) # TP / (TP + FP)
rec = recall_score(y_test, y_pred_test, zero_division=0)   # TP / (TP + FN)
f1 = f1_score(y_test, y_pred_test, zero_division=0)        # المتوسط التوافقي للدقة الإيجابية والاستدعاء

results = pd.DataFrame({
    'المقياس': ['الدقة', 'الدقة الإيجابية', 'الاستدعاء', 'مقياس F1'],
    'القيمة': [acc, prec, rec, f1],
    'الوصف': [
        'نسبة التنبؤات الصحيحة',
        'من المتنبأ بـ نعم، النسبة الفعلية نعم',
        'من نعم الفعلي، النسبة المتنبأ بها بشكل صحيح',
        'توازن بين الدقة الإيجابية والاستدعاء'
    ]
})

display(results.round(4))  # عرض جدول المقاييس

cm = confusion_matrix(y_test, y_pred_test)  # مصفوفة 2×2: TN, FP, FN, TP
print('\\nمصفوفة الارتباك (صفوف=فعلي، أعمدة=متنبأ):')
print(cm)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['لا (0)', 'نعم (1)']).plot(ax=ax, cmap='Blues')
plt.title('مصفوفة الارتباك — إعلانات الشبكات الاجتماعية')
plt.tight_layout()
plt.show()

print(f'\\nدقة اختبار المثال 1 = {acc:.4f}')""",

    26: """## لماذا يعمل الانحدار اللوجستي لإعلانات الشبكات الاجتماعية؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **هدف ثنائي** | Purchased هو 0/1 — مثالي للانحدار اللوجستي |
| 2 | **ميزتان** | العمر والراتب يجتمعان في درجة لوغ-أودز خطية |
| 3 | **حد القرار** | السigmoid ينشئ حداً سلساً في مستوى العمر–الراتب |
| 4 | **الاحتمالات** | فرق التسويق يحتاج P(شراء)، وليس نعم/لا فقط |
| 5 | **تحجيم الميزات** | الراتب (~50k) والعمر (~35) يختلفان في المقياس — StandardScaler يساعد |

> **الملخص:** المستخدمون الأكبر سناً ذوو الرواتب الأعلى يميلون للشراء — يتعلّم الانحدار اللوجستي هذا الحد.""",

    27: """## فهم مقاييس التصنيف — المثال 1

| المقياس | متى يهم |
|--------|-----------------|
| **الدقة** | فئات متوازنة — الأداء العام |
| **الدقة الإيجابية** | تكلفة الإنذارات الكاذبة (التنبؤ بنعم عند لا) |
| **الاستدعاء** | تكلفة تفويت المشترين (التنبؤ بلا عند نعم) |
| **F1** | توازن عندما تكون الفئات غير متوازنة قليلاً |

> **ليس R²:** التصنيف يستخدم **الدقة / الدقة الإيجابية / الاستدعاء / F1** — وليس مقاييس الانحدار.""",

    28: """---
# المثال 2: الموافقة على القروض — الانحدار اللوجستي

التنبؤ بما إذا كان القرض **Approved** (1) أو **Rejected** (0) من الميزات المالية للمتقدّم.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `CreditScore` | ميزة (X₁) | درجة الائتمان (300–850) |
| `Income` | ميزة (X₂) | الدخل السنوي بالدولار |
| `LoanAmount` | ميزة (X₃) | مبلغ القرض المطلوب بالدولار |
| `YearsEmployed` | ميزة (X₄) | سنوات العمل في الوظيفة الحالية |
| `Approved` | الهدف (y) | 1 = موافق، 0 = مرفوض |

**الملف:** `../Datasets/loan_approval.csv`""",

    29: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV الموافقة على القروض وفحص البيانات.

**ما تفعله هذه الخلية:** قراءة `../Datasets/loan_approval.csv` وعرض إحصائيات أساسية.""",

    30: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/loan_approval.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['CreditScore', 'Income', 'LoanAmount', 'YearsEmployed']  # أربع ميزات إدخال
TARGET_COL = 'Approved'  # هدف ثنائي: 0 = مرفوض، 1 = موافق

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

**ما تفعله هذه الخلية:** فحص القيم الفارغة وإزالة التكرارات واستبدال القيم المفقودة.""",

    32: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')
print(dataset.isnull().sum())

rows_before = len(dataset)
dataset = dataset.drop_duplicates().reset_index(drop=True)
rows_after = len(dataset)
print(f'\\nالتكرارات المحذوفة: {rows_before - rows_after}')

# حذف الصفوف التي يغيب فيها الهدف (لا يمكن التدريب بدون تسمية)
if dataset[TARGET_COL].isnull().sum() > 0:
    dataset = dataset.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print('تم حذف الصفوف ذات الهدف المفقود')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
feature_and_target = FEATURE_COLS + [TARGET_COL]
if dataset[feature_and_target].isnull().sum().sum() > 0:
    dataset[feature_and_target] = imputer.fit_transform(dataset[feature_and_target])
    print('تم استبدال قيم الميزات المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المستبدِل')

print(f'\\nالصفوف بعد التنظيف: {len(dataset)}')""",

    33: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة عددية — يُتخطى الترميز.

**ما تفعله هذه الخلية:** تأكيد عدم الحاجة لترميز فئوي.""",

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

    36: """# الخطوة 4) تقسيم تدريب-اختبار

X = dataset[FEATURE_COLS].values  # مصفوفة الميزات: 4 أعمدة
y = dataset[TARGET_COL].values.astype(int)  # الهدف: Approved كعدد صحيح 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'شكل X_train: {X_train.shape}')
print(f'شكل X_test:  {X_test.shape}')
print(f'عدد فئات y_train: {np.bincount(y_train)}')
print(f'عدد فئات y_test:  {np.bincount(y_test)}')""",

    37: """## المثال 2 — الخلية 5: تدريب النموذج

تدريب الانحدار اللوجستي مع StandardScaler على أربع ميزات مالية.

**ما تفعله هذه الخلية:** ملاءمة خط الأنابيب وطباعة المعاملات (تأثير قابل للتفسير).""",

    38: """# الخطوة 5) تدريب الانحدار اللوجستي

classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

classifier.fit(X_train, y_train)

model = classifier.named_steps['model']
print('تم تدريب الانحدار اللوجستي بنجاح.')
print(f'المقطع (β₀): {model.intercept_[0]:.4f}')
print('\\nالمعاملات (على الميزات المحجّمة):')
for name, coef in zip(FEATURE_COLS, model.coef_[0]):
    print(f'  {name:15s} -> β = {coef:+.4f}')""",

    39: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بالموافقة على القرض في مجموعة الاختبار مع الفئات والاحتمالات.

**ما تفعله هذه الخلية:** توليد التنبؤات وعرض عينات من المخرجات.""",

    40: """# الخطوة 6) التنبؤ

y_pred_test = classifier.predict(X_test)
y_proba_test = classifier.predict_proba(X_test)[:, 1]

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    actual = 'موافق' if y_test[i] == 1 else 'مرفوض'
    pred = 'موافق' if y_pred_test[i] == 1 else 'مرفوض'
    print(f'  فعلي={actual}، متنبأ={pred}، P(موافقة)={y_proba_test[i]:.2%}')""",

    41: """## المثال 2 — الخلية 7: التصور

رسم **مخطط أشرطة للمعاملات** و**مصفوفة الارتباك**.

**ما تفعله هذه الخلية:** عرض الميزات التي تدفع نحو الموافقة والأداء العام على الاختبار.""",

    42: """# الخطوة 7) التصور — المعاملات + مصفوفة الارتباك

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- يسار: أهمية المعاملات ---
coefs = classifier.named_steps['model'].coef_[0]
colors = ['seagreen' if c > 0 else 'crimson' for c in coefs]
axes[0].barh(FEATURE_COLS, coefs, color=colors)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('المعامل (ميزات محجّمة)')
axes[0].set_title('معاملات الانحدار اللوجستي — الموافقة على القروض')

# --- يمين: مصفوفة الارتباك ---
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['مرفوض', 'موافق']).plot(ax=axes[1], cmap='Blues')
axes[1].set_title('مصفوفة الارتباك — الموافقة على القروض')

plt.tight_layout()
plt.show()""",

    43: """## المثال 2 — الخلية 8: التقييم

تقرير تصنيف كامل مع الدقة والدقة الإيجابية والاستدعاء وF1.

**ما تفعله هذه الخلية:** حساب وعرض مقاييس التقييم.""",

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
        'من المتنبأ بموافقة، النسبة الموافقة فعلاً',
        'من الموافقة الفعلية، النسبة المتنبأ بها بشكل صحيح',
        'توازن بين الدقة الإيجابية والاستدعاء'
    ]
})

display(results.round(4))

print('\\nتقرير تصنيف مفصّل:')
print(classification_report(y_test, y_pred_test, target_names=['مرفوض (0)', 'موافق (1)']))

print(f'\\nدقة اختبار المثال 2 = {acc:.4f}')""",

    45: """## لماذا يعمل الانحدار اللوجستي للموافقة على القروض؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **قرار ثنائي واضح** | موافق مقابل مرفوض يُطابق 0/1 مباشرة |
| 2 | **ميزات عددية متعددة** | الائتمان والدخل وحجم القرض والتوظيف يجتمعون في درجة واحدة |
| 3 | **معاملات قابلة للتفسير** | β موجب → P(موافقة) أعلى؛ سالب → أقل |
| 4 | **مخرج احتمال** | البنوك يمكنها ضبط عتبة ≠ 0.5 حسب تحمل المخاطر |
| 5 | **التحجيم** | درجة الائتمان والدخل يختلفان في الحجم — StandardScaler مطلوب |

> **رؤية عملية:** درجة ائتمان أعلى ودخل أعلى يزيدان احتمال الموافقة؛ مبالغ قروض أكبر تنقصه.""",

    46: """## فهم مقاييس التصنيف — المثال 2

| السيناريو | المقياس المركز |
|----------|--------------|
| **الموافقة الخاطئة مكلفة** (قرض سيء) | **الدقة الإيجابية** — تقليل الموافقات الكاذبة |
| **الرفض الخاطئ مكلف** (عميل مفقود) | **الاستدعاء** — اكتشاف جميع المتقدّمين الجيدين |
| **تكلفة متوازنة** | **مقياس F1** أو **الدقة** |

> استخدم **`classification_report`** لرؤية الدقة الإيجابية والاستدعاء وF1 لكل فئة.""",
}
