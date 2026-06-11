# -*- coding: utf-8 -*-
"""Arabic translations for SVR_regression.ipynb cells."""

CELLS = {
    0: """# الانحدار باستخدام آلات الدعم (SVR) — Google Colab

**الهدف:** التنبؤ بقيمة مستهدفة متصلة باستخدام **الانحدار بآلات الدعم (SVR)** — قوي للعلاقات **غير الخطية**.

| المثال | الميزة (X) | الهدف (y) | مجموعة البيانات |
|---------|-------------|------------|---------|
| **المثال 1** | Level | Salary | `../Datasets/Position_Salaries.csv` |
| **المثال 2** | Speed_kmh | Braking_distance_m | `../Datasets/braking_distance.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | التعريفات، المعادلات، المعاملات الفائقة |
| المرحلة 1 | معالجة البيانات | تحميل ← تنظيف ← ترميز ← تقسيم ← **تحجيم** |
| المرحلة 2 | الخوارزمية | تدريب ← تنبؤ ← تصور ← تقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)""",

    1: """---
# دليل الخوارزمية — الانحدار بآلات الدعم (SVR)

## ما هو SVR؟

**الانحدار بآلات الدعم (SVR)** هو خوارزمية انحدار مبنية على **آلات الدعم (SVM)**.  
تجد دالة تلائم البيانات مع إبقاء التنبؤات داخل **أنبوب ε** (هامش التسامح).

## الفكرة (مبسّطة)

| المفهوم | المعنى |
|---------|--------|
| **متجهات الدعم** | أهم نقاط التدريب التي تحدد النموذج |
| **ε (إبسيلون)** | الأخطاء الأصغر من ε تُتجاهل — بلا عقوبة |
| **C** | يتحكم في المفاضلة: C كبير = ملاءمة وثيقة للبيانات، C صغير = نموذج أنعم |
| **النواة (Kernel)** | تحوّل الميزات لالتقاط أنماط غير خطية (مثل `rbf`) |

## المعادلة (نسخة الطالب)

**SVR خطي:**

`y = w·x + b`  (مع خسارة ε-غير حساسة)

**مع النواة (غير خطي):**

`y = Σ(αᵢ − αᵢ*) · K(xᵢ, x) + b`

حيث `K(xᵢ, x)` هي دالة النواة (مثل RBF: `exp(−γ||x − x'||²)`).

## هدف التحسين

تقليل:

` (1/2)||w||² + C · Σ(ξᵢ + ξᵢ*) `

مع بقاء التنبؤات ضمن **ε** من القيمة الحقيقية.

## المعاملات الفائقة الرئيسية

| المعامل | الدور | القيم النموذجية |
|-----------|------|----------------|
| `kernel` | شكل سطح القرار | `'rbf'`, `'linear'`, `'poly'` |
| `C` | قوة التنظيم | `1`, `10`, `100` |
| `epsilon` | عرض أنبوب ε | `0.01` – `0.2` |
| `gamma` | تأثير نواة RBF (`rbf` فقط) | `'scale'`, `'auto'`, أو عدد عشري |

## SVR مقابل الانحدار الخطي

| | الانحدار الخطي | SVR |
|---|-------------------|-----|
| الشكل | خط مستقيم / مستوى | يمكن أن يكون **منحنياً** (مع نواة RBF) |
| التحجيم | اختياري (ميزة واحدة) | **مطلوب** |
| القيم الشاذة | حساس | أكثر متانة (أنبوب ε) |
| القابلية للتفسير | سهلة (المعاملات) | أصعب (مبني على النواة) |

## ما يجب أن يتذكره الطالب

1. **طبّق دائماً تحجيم الميزات** قبل SVR (`StandardScaler`).
2. استخدم **`kernel='rbf'`** عندما تكون العلاقة غير خطية.
3. **`C` مرتفع جداً** ← فرط في الملاءمة؛ **`C` منخفض جداً** ← نقص في الملاءمة.
4. **`epsilon`** يحدد مقدار الخطأ الذي تتسامح معه بلا عقوبة.
5. قيّم باستخدام **MAE, RMSE, R²** — كباقي نماذج الانحدار.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبّت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات لمعالجة البيانات والمعالجة المسبقة والنمذجة والتقييم.

**ما تفعله هذه الخلية:** تحمّل numpy و pandas و matplotlib و seaborn و SVR من sklearn والمحوّلات.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء الرسوم البيانية
import seaborn as sns           # تصورات إحصائية (تنسيق اختياري)

from sklearn.model_selection import train_test_split  # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer              # ملء القيم المفقودة
from sklearn.preprocessing import StandardScaler      # تحجيم الميزات (مطلوب لـ SVR)
from sklearn.pipeline import Pipeline                 # ربط خطوات المحوّل + النموذج
from sklearn.svm import SVR                           # نموذج الانحدار بآلات الدعم
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # مقاييس التقييم

plt.rcParams['figure.figsize'] = (10, 6)  # حجم الرسم الافتراضي: عرض=10، ارتفاع=6 بوصة
plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لتقسيم قابل للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح""",

    6: """---
# المثال 1: رواتب المناصب — SVR

التنبؤ بـ **Salary** من **Level** الوظيفي. العلاقة **غير خطية** (الراتب ينمو أسرع عند المستويات الأعلى).

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Level` | الميزة (X) | مستوى الوظيفة (1–10) |
| `Salary` | الهدف (y) | الراتب السنوي بالدولار الأمريكي |

**الملف:** `../Datasets/Position_Salaries.csv`""",

    7: """---
# المرحلة 1: معالجة البيانات

تحضير البيانات قبل التدريب — نفس القالب يُعاد استخدامه لخوارزميات أخرى.""",

    8: """## المثال 1 — الخلية 1: تحميل واستكشاف البيانات

تحميل ملف CSV وإجراء استكشاف أولي (head, info, describe, shape).

**ما تفعله هذه الخلية:** تقرأ `../Datasets/Position_Salaries.csv` وتعرض إحصائيات أساسية.""",

    9: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/Position_Salaries.csv')  # قراءة CSV إلى DataFrame

FEATURE_COL = 'Level'   # المتغير المستقل (X) — مستوى الوظيفة
TARGET_COL = 'Salary'   # المتغير التابع (y) — الراتب المراد التنبؤ به

print('أول 5 صفوف:')          # طباعة تسمية للجدول أدناه
display(dataset.head())         # عرض أول 5 صفوف لفحص البيانات

print('\\nمعلومات مجموعة البيانات:')       # طباعة تسمية لأنواع الأعمدة وعدد القيم الفارغة
dataset.info()                  # عرض أسماء الأعمدة وأنواع البيانات وعدد القيم غير الفارغة

print('\\nملخص إحصائي:')  # طباعة تسمية للإحصائيات العددية
display(dataset.describe())       # عرض العدد والمتوسط والانحراف المعياري والحد الأدنى والأقصى والربيعيات

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')  # إجمالي الصفوف والأعمدة""",

    10: """## المثال 1 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

التحقق من القيم المفقودة، إزالة التكرارات، وتطبيق الإكمال عند الحاجة.

**ما تفعله هذه الخلية:** تنظف مجموعة البيانات قبل النمذجة.""",

    11: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')  # طباعة تسمية
print(dataset.isnull().sum())        # عد قيم NaN في كل عمود

rows_before = len(dataset)                              # تخزين عدد الصفوف قبل التنظيف
dataset = dataset.drop_duplicates().reset_index(drop=True)  # إزالة الصفوف المكررة
rows_after = len(dataset)                               # تخزين عدد الصفوف بعد إزالة التكرار
print(f'\\nالتكرارات المُزالة: {rows_before - rows_after}')  # عرض عدد التكرارات

num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()  # الأعمدة العددية فقط
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # ملء NaN بمتوسط العمود
if dataset.isnull().sum().sum() > 0:               # إذا وُجدت قيم مفقودة
    dataset[num_cols] = imputer.fit_transform(dataset[num_cols])  # إكمال الأعمدة العددية
    print('تم إكمال القيم المفقودة بالمتوسط')      # تأكيد الإكمال
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُكمِّل')  # تخطي عند اكتمال البيانات

print(f'\\nالصفوف بعد التنظيف: {rows_after}')    # العدد النهائي للصفوف""",

    12: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

نستخدم `Level` (عددي). عمود `Position` نصي — يُتخطى لأن Level يمثّل الرتبة بالفعل.

**ما تفعله هذه الخلية:** تتحقق من الأعمدة الفئوية وترمّزها عند الحاجة.""",

    13: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()  # إيجاد الأعمدة النصية
print(f'الأعمدة الفئوية (غير مستخدمة كـ X): {cat_cols}')  # أسماء المناصب — للمعلومات فقط
print(f'الميزة المستخدمة لـ SVR: {FEATURE_COL}')             # Level عددي — لا حاجة للترميز
print('لا يلزم ترميز — X عددي.')""",

    14: """## المثال 1 — الخلية 4: تقسيم البيانات

تعريف X (Level) و y (Salary)، ثم التقسيم 80/20.

**ما تفعله هذه الخلية:** تنشئ مصفوفات الميزات/الهدف وتطبّق train_test_split.""",

    15: """# الخطوة 4) تقسيم التدريب-الاختبار

X = dataset[[FEATURE_COL]].values  # مصفوفة الميزات: Level (مصفوفة ثنائية الأبعاد لـ sklearn)
y = dataset[TARGET_COL].values     # متجه الهدف: قيم الراتب

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                  # البيانات المراد تقسيمها
    test_size=0.2,         # 20% اختبار، 80% تدريب
    random_state=42        # تقسيم قابل للتكرار
)

print(f'شكل X_train: {X_train.shape}')  # شكل ميزات التدريب
print(f'شكل X_test:  {X_test.shape}')   # شكل ميزات الاختبار
print(f'شكل y_train: {y_train.shape}')  # شكل أهداف التدريب
print(f'شكل y_test:  {y_test.shape}')   # شكل أهداف الاختبار""",

    16: """## المثال 1 — الخلية 5: تحجيم الميزات (مطلوب لـ SVR)

SVR **حساس لمقياس الميزات**. نُوحّد X (واختيارياً y) باستخدام `StandardScaler`.

**المعادلة:** `z = (x − mean) / std`

**ما تفعله هذه الخلية:** تُلائم المحوّل على بيانات التدريب وتحوّل مجموعات التدريب/الاختبار.""",

    17: """# الخطوة 5) تحجيم الميزات — مطلوب لـ SVR

scaler_X = StandardScaler()  # إنشاء محوّل للميزات
X_train_scaled = scaler_X.fit_transform(X_train)  # تعلّم المتوسط/الانحراف من التدريب ثم التحويل
X_test_scaled = scaler_X.transform(X_test)        # تحويل الاختبار بإحصائيات التدريب

scaler_y = StandardScaler()  # إنشاء محوّل للهدف (يساعد تحسين SVR)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()  # تحجيم y للتدريب
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()      # تحجيم y للاختبار

print('تم تطبيق تحجيم الميزات (StandardScaler).')  # تأكيد اكتمال التحجيم
print(f'عينة من X_train المُحجَّم: {X_train_scaled[:3].flatten()}')  # عرض أول 3 قيم مُحجَّمة""",

    18: """---
# المرحلة 2: الانحدار بآلات الدعم

تدريب SVR بنواة RBF لالتقاط منحنى الراتب **غير الخطي**.""",

    19: """## المثال 1 — الخلية 6: تدريب النموذج

تدريب `SVR` بمعاملات فائقة: `kernel='rbf'`, `C=100`, `epsilon=0.1`.

**ما تفعله هذه الخلية:** تُلائم نموذج SVR على بيانات التدريب المُحجَّمة.""",

    20: """# الخطوة 6) تدريب نموذج SVR

regressor = SVR(
    kernel='rbf',    # دالة الأساس الشعاعية — تلتقط الأنماط غير الخطية
    C=100,           # التنظيم: أعلى = ملاءمة أوثق لبيانات التدريب
    epsilon=0.1,     # أنبوب ε: الأخطاء ضمن 0.1 (وحدات مُحجَّمة) بلا عقوبة
    gamma='scale'    # معامل نواة RBF: 1 / (n_features * X.var())
)

regressor.fit(X_train_scaled, y_train_scaled)  # التدريب على X و y المُحجَّمين

print('تم تدريب نموذج SVR بنجاح.')  # تأكيد اكتمال التدريب
print(f'متجهات الدعم المستخدمة: {regressor.n_support_}')  # عدد متجهات الدعم لكل فئة
print(f'المعاملات الفائقة: kernel=rbf, C=100, epsilon=0.1')  # عرض الإعدادات المختارة""",

    21: """## المثال 1 — الخلية 7: التنبؤ

التنبؤ بالراتب على مجموعة الاختبار. التنبؤات في **فضاء مُحجَّم** — إعادة التحويل العكسي إلى الدولار الأصلي.

**ما تفعله هذه الخلية:** تولّد التنبؤات وتحوّلها إلى مقياس الراتب الأصلي.""",

    22: """# الخطوة 7) التنبؤ

y_pred_scaled = regressor.predict(X_test_scaled)  # التنبؤ في الفضاء المُحجَّم
y_pred_test = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # العودة إلى الدولار

print('عينات تنبؤ (مجموعة الاختبار):')  # طباعة تسمية
for i in range(len(y_test)):             # عرض جميع تنبؤات الاختبار (مجموعة بيانات صغيرة)
    print(f'  Level={X_test[i][0]:.0f} -> فعلي=${y_test[i]:,.0f}, متوقع=${y_pred_test[i]:,.0f}')""",

    23: """## المثال 1 — الخلية 8: التصور

مقارنة **الانحدار الخطي مقابل SVR** على منحنى ناعم.

**ما تفعله هذه الخلية:** ترسم نقاط البيانات ومنحني النموذجين.""",

    24: """# الخطوة 8) التصور — خطي مقابل SVR
from sklearn.linear_model import LinearRegression  # استيراد للمقارنة فقط

X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)  # 200 نقطة لمنحنى ناعم
X_plot_scaled = scaler_X.transform(X_plot)                    # تحجيم نقاط الرسم

lin_reg = LinearRegression()           # نموذج خطي بسيط للمقارنة
lin_reg.fit(X_train, y_train)          # التدريب على البيانات الأصلية (غير المُحجَّمة)
y_lin = lin_reg.predict(X_plot)        # تنبؤات خطية

y_svr_scaled = regressor.predict(X_plot_scaled)  # تنبؤات SVR (مُحجَّمة)
y_svr = scaler_y.inverse_transform(y_svr_scaled.reshape(-1, 1)).ravel()  # العودة إلى الدولار

plt.figure(figsize=(10, 6))  # إنشاء الرسم
plt.scatter(X_train, y_train, color='blue', label='التدريب', s=80, zorder=3)   # نقاط التدريب
plt.scatter(X_test, y_test, color='green', label='الاختبار', s=80, zorder=3)       # نقاط الاختبار
plt.plot(X_plot, y_lin, color='orange', linewidth=2, label='الانحدار الخطي')  # الخط الخطي
plt.plot(X_plot, y_svr, color='red', linewidth=2, label='SVR (RBF)')             # منحنى SVR
plt.xlabel('Level')            # تسمية المحور السيني
plt.ylabel('الراتب (دولار أمريكي)')    # تسمية المحور الصادي
plt.title('المثال 1: خطي مقابل SVR — رواتب المناصب')  # عنوان الرسم
plt.legend()                   # عرض وسيلة الإيضاح
plt.tight_layout()             # ضبط التخطيط
plt.show()                     # عرض الرسم""",

    25: """## المثال 1 — الخلية 9: التقييم

تقييم SVR باستخدام MAE و RMSE و R² على مجموعة الاختبار.

**ما تفعله هذه الخلية:** تحسب وتعرض مقاييس التقييم.""",

    26: """# الخطوة 9) التقييم
mae = mean_absolute_error(y_test, y_pred_test)              # متوسط الخطأ المطلق بالدولار
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))     # جذر متوسط مربع الخطأ
r2 = r2_score(y_test, y_pred_test)                          # التباين المُفسَّر

results = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae, rmse, r2],
    'Description': [
        'متوسط الخطأ المطلق (دولار أمريكي)',
        'جذر متوسط مربع الخطأ (دولار أمريكي)',
        'معامل التحديد (1 = مثالي)'
    ]
})

display(results.round(4))  # عرض جدول المقاييس
print(f'\\nR² اختبار المثال 1 = {r2:.4f}')  # طباعة ملخص R²""",

    27: """## لماذا يعمل SVR جيداً لرواتب المناصب؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **علاقة غير خطية** | الراتب ينمو أسرع عند المستويات الأعلى — ليس خطاً مستقيماً |
| 2 | **نواة RBF** | SVR يمكنه الانحناء ليتابع النمو الأسيّ الشبيه للراتب |
| 3 | **تطبيق تحجيم الميزات** | StandardScaler يتيح لـ SVR التحسين بشكل صحيح |
| 4 | **أفضل من الانحدار الخطي** | النموذج الخطي يعاني نقص ملاءمة لهذا النمط المنحني |
| 5 | **مجموعة بيانات صغيرة** | SVR يستخدم متجهات الدعم — يعمل مع نقاط بيانات قليلة |

> **الملخص:** SVR بنواة RBF يُلائم علاقة الراتب-المستوى **المنحنية** أفضل من خط مستقيم.""",

    28: """---
# المثال 2: مسافة الفرملة — SVR

التنبؤ بـ **Braking Distance** من **Speed**. الفيزياء تخبرنا أن المسافة تنمو **بشكل غير خطي** مع السرعة.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Speed_kmh` | الميزة (X) | سرعة المركبة بالكم/س |
| `Braking_distance_m` | الهدف (y) | مسافة الفرملة بالأمتار |

**العلاقة الحقيقية:** `Distance ≈ 0.04×Speed² + 0.5×Speed + 5` (تربيعية)

**الملف:** `../Datasets/braking_distance.csv`""",

    29: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV مسافة الفرملة وفحص البيانات.

**ما تفعله هذه الخلية:** تقرأ `../Datasets/braking_distance.csv` وتعرض إحصائيات أساسية.""",

    30: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/braking_distance.csv')  # قراءة CSV إلى DataFrame

FEATURE_COL = 'Speed_kmh'            # المتغير المستقل (X)
TARGET_COL = 'Braking_distance_m'    # المتغير التابع (y)

print('أول 5 صفوف:')
display(dataset.head())

print('\\nمعلومات مجموعة البيانات:')
dataset.info()

print('\\nملخص إحصائي:')
display(dataset.describe())

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')""",

    31: """## المثال 2 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

تتضمن مجموعة البيانات قيماً مفقودة لعرض `SimpleImputer`.

**ما تفعله هذه الخلية:** تتحقق من القيم الفارغة، تزيل التكرارات، وتُكمِّل القيم المفقودة.""",

    32: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')
print(dataset.isnull().sum())

rows_before = len(dataset)
dataset = dataset.drop_duplicates().reset_index(drop=True)
rows_after = len(dataset)
print(f'\\nالتكرارات المُزالة: {rows_before - rows_after}')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
if dataset.isnull().sum().sum() > 0:
    dataset[[FEATURE_COL, TARGET_COL]] = imputer.fit_transform(dataset[[FEATURE_COL, TARGET_COL]])
    print('تم إكمال القيم المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُكمِّل')

print(f'\\nالصفوف بعد التنظيف: {rows_after}')""",

    33: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة عددية — يُتخطى الترميز.

**ما تفعله هذه الخلية:** تؤكد عدم الحاجة لترميز فئوي.""",

    34: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية وُجدت: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'الميزة: {FEATURE_COL}, الهدف: {TARGET_COL}')""",

    35: """## المثال 2 — الخلية 4: تقسيم البيانات

تعريف X (Speed) و y (Distance)، ثم التقسيم 80/20.

**ما تفعله هذه الخلية:** تنشئ مصفوفات الميزات/الهدف وتطبّق train_test_split.""",

    36: """# الخطوة 4) تقسيم التدريب-الاختبار

X = dataset[[FEATURE_COL]].values
y = dataset[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'شكل X_train: {X_train.shape}')
print(f'شكل X_test:  {X_test.shape}')
print(f'شكل y_train: {y_train.shape}')
print(f'شكل y_test:  {y_test.shape}')""",

    37: """## المثال 2 — الخلية 5: تحجيم الميزات (مطلوب لـ SVR)

توحيد الميزات والهدف قبل تدريب SVR.

**ما تفعله هذه الخلية:** تطبّق StandardScaler على مجموعات التدريب والاختبار.""",

    38: """# الخطوة 5) تحجيم الميزات

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

print('تم تطبيق تحجيم الميزات (StandardScaler).')""",

    39: """## المثال 2 — الخلية 6: تدريب النموذج

تدريب SVR بنواة RBF على البيانات المُحجَّمة.

**ما تفعله هذه الخلية:** تُلائم SVR وتطبع عدد متجهات الدعم.""",

    40: """# الخطوة 6) تدريب نموذج SVR

regressor = SVR(
    kernel='rbf',    # نواة غير خطية للعلاقة التربيعية
    C=100,           # قوة التنظيم
    epsilon=0.1,     # عرض أنبوب ε
    gamma='scale'    # معامل النواة
)

regressor.fit(X_train_scaled, y_train_scaled)

print('تم تدريب نموذج SVR بنجاح.')
print(f'متجهات الدعم المستخدمة: {regressor.n_support_}')""",

    41: """## المثال 2 — الخلية 7: التنبؤ

التنبؤ بمسافة الفرملة وإعادة التحويل العكسي إلى الأمتار.

**ما تفعله هذه الخلية:** تولّد التنبؤات على مجموعة الاختبار.""",

    42: """# الخطوة 7) التنبؤ

y_pred_scaled = regressor.predict(X_test_scaled)
y_pred_test = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    print(f'  Speed={X_test[i][0]:.1f} km/h -> فعلي={y_test[i]:.1f}م، متوقع={y_pred_test[i]:.1f}م')""",

    43: """## المثال 2 — الخلية 8: التصور

رسم البيانات مع منحنيات خطي مقابل SVR.

**ما تفعله هذه الخلية:** يُظهر كيف يلتقط SVR النمط التربيعي.""",

    44: """# الخطوة 8) التصور
from sklearn.linear_model import LinearRegression

X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
X_plot_scaled = scaler_X.transform(X_plot)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_lin = lin_reg.predict(X_plot)

y_svr_scaled = regressor.predict(X_plot_scaled)
y_svr = scaler_y.inverse_transform(y_svr_scaled.reshape(-1, 1)).ravel()

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='التدريب', alpha=0.7)
plt.scatter(X_test, y_test, color='green', label='الاختبار', alpha=0.7)
plt.plot(X_plot, y_lin, color='orange', linewidth=2, label='الانحدار الخطي')
plt.plot(X_plot, y_svr, color='red', linewidth=2, label='SVR (RBF)')
plt.xlabel('السرعة (كم/س)')
plt.ylabel('مسافة الفرملة (م)')
plt.title('المثال 2: خطي مقابل SVR — مسافة الفرملة')
plt.legend()
plt.tight_layout()
plt.show()""",

    45: """## المثال 2 — الخلية 9: التقييم

تقييم أداء SVR باستخدام MAE و RMSE و R².

**ما تفعله هذه الخلية:** تحسب وتعرض مقاييس التقييم.""",

    46: """# الخطوة 9) التقييم
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

results = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae, rmse, r2],
    'Description': [
        'متوسط الخطأ المطلق (أمتار)',
        'جذر متوسط مربع الخطأ (أمتار)',
        'معامل التحديد (1 = مثالي)'
    ]
})

display(results.round(4))
print(f'\\nR² اختبار المثال 2 = {r2:.4f}')""",

    47: """## لماذا يعمل SVR جيداً لمسافة الفرملة؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **فيزياء تربيعية** | مسافة الفرملة ∝ السرعة² — منحنية بطبيعتها |
| 2 | **نواة RBF تلائم المنحنيات** | SVR يتبع المنحنى الصاعد أفضل من خط |
| 3 | **تطبيق التحجيم** | السرعة والمسافة بمقاييس مختلفة — StandardScaler يصحح ذلك |
| 4 | **بيانات أكثر (50 صف)** | نقاط أكثر تساعد SVR على تحديد المنحنى بدقة |
| 5 | **R² أعلى من الخطي** | الانحدار الخطي لا يلتقط نمو السرعة² |

> **المقارنة:** الانحدار الخطي يعطي R² معتدلاً؛ **SVR (RBF)** يحقق عادةً **R² > 0.95** على هذه المجموعة.""",
}
