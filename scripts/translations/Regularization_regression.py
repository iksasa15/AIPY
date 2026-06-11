# -*- coding: utf-8 -*-
"""Arabic cell sources for Regularization_regression.ipynb"""

CELLS = {
    0: """# طرق التنظيم (Ridge، Lasso، Elastic Net) — Google Colab

**الهدف:** التحكم في **فرط التجهيز** في النماذج الخطية بإضافة **عقوبة** على المعاملات الكبيرة — مقارنة **Ridge (L2)** و **Lasso (L1)** و **Elastic Net (L1+L2)**.

| مثال | المتغير(ات) (X) | الهدف (y) | مجموعة البيانات |
|---------|----------------|------------|---------|
| **المثال 1** | Level (درجة كثير حدود 7) | Salary | `../Datasets/Position_Salaries.csv` |
| **المثال 2** | R&D، Admin، Marketing، State | Profit | `../Datasets/50_Startups.csv` |

| مرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | L1، L2، alpha، متى تستخدم كل منها |
| المرحلة 1 | معالجة البيانات | تحميل ← تنظيف ← ترميز ← تقسيم |
| المرحلة 2 | الخوارزمية | تدريب ← تنبؤ ← تصوير ← تقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)
""",

    1: """---
# دليل الخوارزمية — طرق التنظيم (Regularization)

## لماذا التنظيم؟

عندما يكون النموذج **مرنًا جدًا** (عدد كبير من المتغيرات، درجة كثير حدود عالية)، قد **يحفظ الضوضاء** بدلًا من تعلّم النمط الحقيقي.

| المشكلة | العرض | حل التنظيم |
|---------|---------|-------------------|
| **فرط التجهيز** | خطأ تدريب منخفض، خطأ اختبار مرتفع | معاقبة المعاملات الكبيرة |
| **التعدد الخطي** | معاملات غير مستقرة | Ridge يقلّص المتغيرات المترابطة معًا |
| **عدد كبير من المتغيرات** | تفسير ضعيف | Lasso يجعل بعض المعاملات **صفرًا** |

## دالة الهدف العامة للنموذج المُنظَّم

**تقليل:** خسارة البيانات + λ · العقوبة

`Loss = Σ(yᵢ − ŷᵢ)² + α · Ω(β)`

في scikit-learn، **`alpha`** يؤدي دور **λ** (قوة العقوبة).

## بدون تنظيم (OLS / الانحدار الخطي)

`minimize Σ(yᵢ − ŷᵢ)²`

المربعات الصغرى العادية — لا عقوبة على المعاملات.

## انحدار Ridge (عقوبة L2)

`minimize Σ(yᵢ − ŷᵢ)² + α · Σ βⱼ²`

| الخاصية | التفاصيل |
|----------|--------|
| العقوبة | **مجموع مربعات** المعاملات |
| التأثير | يقلّص المعاملات **باتجاه الصفر** بسلاسة |
| اختيار المتغيرات | **لا** — جميع المتغيرات تبقى في النموذج |
| الأفضل لـ | عدد كبير من المتغيرات، التعدد الخطي |

## انحدار Lasso (عقوبة L1)

`minimize Σ(yᵢ − ŷᵢ)² + α · Σ |βⱼ|`

| الخاصية | التفاصيل |
|----------|--------|
| العقوبة | **مجموع القيم المطلقة** للمعاملات |
| التأثير | قد يجعل بعض المعاملات **صفرًا بالضبط** |
| اختيار المتغيرات | **نعم** — اختيار تلقائي للمتغيرات |
| الأفضل لـ | نماذج متفرقة — عدد قليل من المتغيرات المهمة |

## Elastic Net (L1 + L2)

`minimize Σ(yᵢ − ŷᵢ)² + α · [l1_ratio · Σ|βⱼ| + (1 − l1_ratio) · Σ βⱼ²]`

| الخاصية | التفاصيل |
|----------|--------|
| العقوبة | **مزيج** من Lasso و Ridge |
| `l1_ratio=1` | Lasso خالص |
| `l1_ratio=0` | Ridge خالص |
| الأفضل لـ | متغيرات مترابطة + حاجة لبعض التفرّق |

## جدول المقارنة

| الطريقة | العقوبة | يقلّص المعاملات | معاملات صفرية | اختيار المتغيرات |
|--------|---------|---------------|------------|-------------------|
| **الانحدار الخطي** | لا شيء | لا | لا | لا |
| **Ridge** | L2 | نعم | نادرًا | لا |
| **Lasso** | L1 | نعم | **نعم** | **نعم** |
| **Elastic Net** | L1 + L2 | نعم | نعم | نعم |

## المعامل الفائق الرئيسي: `alpha`

| alpha | التأثير |
|-------|--------|
| **صغير** (→ 0) | عقوبة ضعيفة — أقرب إلى OLS |
| **كبير** | عقوبة قوية — نموذج أبسط، قد يحدث نقص التجهيز |

> اضبط دائمًا `alpha` باستخدام التحقق المتقاطع (`RidgeCV`، `LassoCV`) في الإنتاج.

## قياس المتغيرات — مطلوب!

النماذج المُنظَّمة **حساسة لمقياس المتغيرات**. طبّق دائمًا **`StandardScaler`** قبل Ridge أو Lasso أو Elastic Net حتى تُعامل العقوبة جميع المتغيرات بعدالة.

## ما يجب أن يتذكره الطالب

1. التنظيم = **عقوبة على المعاملات** لتقليل فرط التجهيز.
2. **Ridge (L2)** يقلّص؛ **Lasso (L1)** قد **يُزيل** متغيرات.
3. **Elastic Net** يجمع بينهما — خيار افتراضي جيد عندما تكون المتغيرات مترابطة.
4. **`alpha`** يتحكم في قوة العقوبة (أعلى = نموذج أبسط).
5. **قياس المتغيرات دائمًا** قبل تطبيق التنظيم.
""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

Google Colab يتضمّن عادةً معظم المكتبات. هذه الخلية تضمن توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبيت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.
""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn
""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات للمعالجة المسبقة والنماذج الخطية المُنظَّمة والتقييم.

**ما تفعله هذه الخلية:** تحميل numpy و pandas و matplotlib و sklearn Ridge/Lasso/ElasticNet و Pipeline والمقاييس.
""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء الرسوم البيانية
import seaborn as sns           # تصورات إحصائية (تنسيق اختياري)

from sklearn.model_selection import train_test_split       # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # هندسة المتغيرات والقياس
from sklearn.pipeline import Pipeline                      # ربط خطوات المعالجة + النموذج
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # OLS + نماذج مُنظَّمة
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # مقاييس التقييم

plt.rcParams['figure.figsize'] = (10, 6)  # حجم الرسم الافتراضي: عرض=10، ارتفاع=6 بوصة
plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لتقسيمات قابلة للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح
""",

    6: """---
# المثال 1: رواتب المناصب — التنظيم مع متغيرات كثير الحدود

التنبؤ بـ **Salary** من **Level** باستخدام متغيرات **كثير حدود درجة 7**. كثير الحدود ذات الدرجة العالية **يسبب فرط التجهيز** على مجموعات بيانات صغيرة — التنظيم يصلح ذلك.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Level` | متغير (X) | مستوى الوظيفة (1–10) ← يُوسَّع إلى x، x²، …، x⁷ |
| `Salary` | الهدف (y) | الراتب السنوي بالدولار الأمريكي |

**الملف:** `../Datasets/Position_Salaries.csv`

**النماذج المُقارَنة:** الانحدار الخطي (بدون تنظيم) · Ridge · Lasso · Elastic Net
""",

    7: """---
# المرحلة 1: معالجة البيانات

إعداد البيانات قبل التدريب — نفس القالب يُعاد استخدامه لخوارزميات أخرى.
""",

    8: """## المثال 1 — الخلية 1: تحميل واستكشاف البيانات

تحميل ملف CSV وإجراء استكشاف أولي (head، info، describe، shape).

**ما تفعله هذه الخلية:** قراءة `../Datasets/Position_Salaries.csv` وعرض إحصائيات أساسية.
""",

    9: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/Position_Salaries.csv')  # قراءة CSV إلى DataFrame

FEATURE_COL = 'Level'   # المتغير المستقل (X) — مستوى الوظيفة
TARGET_COL = 'Salary'   # المتغير التابع (y) — الراتب المراد التنبؤ به
POLY_DEG = 7            # درجة عالية — تُظهر فرط التجهيز على بيانات صغيرة (R² تدريب → 1.0 لـ OLS)

print('أول 5 صفوف:')          # طباعة تسمية للجدول أدناه
display(dataset.head())         # عرض أول 5 صفوف لفحص البيانات

print('\\nمعلومات مجموعة البيانات:')       # طباعة تسمية لأنواع الأعمدة وعدد القيم الفارغة
dataset.info()                  # عرض أسماء الأعمدة وأنواع البيانات وعدد القيم غير الفارغة

print('\\nملخص إحصائي:')  # طباعة تسمية للإحصائيات العددية
display(dataset.describe())       # عرض العدد والمتوسط والانحراف المعياري والحد الأدنى والأقصى والربيعيات

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')  # إجمالي الصفوف والأعمدة
""",

    10: """## المثال 1 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

التحقق من القيم المفقودة وإزالة التكرارات وتطبيق الإكمال عند الحاجة.

**ما تفعله هذه الخلية:** تنظيف مجموعة البيانات قبل النمذجة.
""",

    11: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')  # طباعة تسمية
print(dataset.isnull().sum())        # عد NaN في كل عمود

rows_before = len(dataset)                              # عدد الصفوف قبل التنظيف
dataset = dataset.drop_duplicates().reset_index(drop=True)  # إزالة الصفوف المكررة
rows_after = len(dataset)                               # عدد الصفوف بعد إزالة التكرار
print(f'\\nالتكرارات المُزالة: {rows_before - rows_after}')  # عرض عدد التكرارات

num_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()  # الأعمدة العددية فقط
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # ملء NaN بمتوسط العمود
if dataset.isnull().sum().sum() > 0:               # إذا وُجدت قيم مفقودة
    dataset[num_cols] = imputer.fit_transform(dataset[num_cols])  # إكمال الأعمدة العددية
    print('تم إكمال القيم المفقودة بالمتوسط')      # تأكيد الإكمال
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق الإكمال')  # تخطّ عند اكتمال البيانات

print(f'\\nالصفوف بعد التنظيف: {rows_after}')    # العدد النهائي للصفوف
""",

    12: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

نستخدم `Level` (عددي). عمود `Position` نصي — يُتخطّى لأن Level يُرمّز المرتبة بالفعل.

**ما تفعله هذه الخلية:** التحقق من الأعمدة الفئوية وترميزها عند الحاجة.
""",

    13: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()  # إيجاد الأعمدة النصية
print(f'الأعمدة الفئوية (غير مستخدمة كـ X): {cat_cols}')  # أسماء المناصب — للمعلومات فقط
print(f'المتغير المستخدم: {FEATURE_COL} (درجة كثير حدود {POLY_DEG})')  # يوسّع Level إلى قوى متعددة
print('لا يلزم ترميز — X عددي.')
""",

    14: """## المثال 1 — الخلية 4: تقسيم البيانات

تعريف X (Level) و y (Salary)، ثم تقسيم 80/20.

**ما تفعله هذه الخلية:** إنشاء مصفوفات المتغيرات/الهدف وتطبيق train_test_split.
""",

    15: """# الخطوة 4) تقسيم تدريب-اختبار

X = dataset[[FEATURE_COL]].values  # مصفوفة المتغيرات: Level (مصفوفة ثنائية الأبعاد لـ sklearn)
y = dataset[TARGET_COL].values     # متجه الهدف: قيم Salary

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                  # البيانات المراد تقسيمها
    test_size=0.2,         # 20% اختبار، 80% تدريب
    random_state=42        # تقسيم قابل للتكرار
)

print(f'شكل X_train: {X_train.shape}')  # شكل متغيرات التدريب
print(f'شكل X_test:  {X_test.shape}')   # شكل متغيرات الاختبار
print(f'شكل y_train: {y_train.shape}')  # شكل أهداف التدريب
print(f'شكل y_test:  {y_test.shape}')   # شكل أهداف الاختبار
""",

    16: """> **ملاحظة:** التنظيم يتطلب **قياس المتغيرات**. كل خط أنابيب نموذج يتضمن `StandardScaler` بعد توسيع كثير الحدود.
""",

    17: """---
# المرحلة 2: التنظيم — نموذج الراتب بكثير الحدود

تدريب ومقارنة أربعة نماذج على متغيرات كثير الحدود.
""",

    18: """## المثال 1 — الخلية 5: بناء خطوط الأنابيب وتدريب النماذج

إنشاء أربعة خطوط أنابيب: **بدون تنظيم (OLS)** و **Ridge** و **Lasso** و **Elastic Net** — كل منها مع PolynomialFeatures + StandardScaler.

**ما تفعله هذه الخلية:** تعريف خطوط الأنابيب وملاءمة جميع النماذج وتخزينها في قاموس.
""",

    19: """# الخطوة 5) بناء وتدريب نماذج كثير الحدود المُنظَّمة

# أربعة نماذج للمقارنة — كل خط أنابيب: PolynomialFeatures → StandardScaler → النموذج
models = {
    'No Reg (OLS)': Pipeline([
        ('poly', PolynomialFeatures(degree=POLY_DEG, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())  # المربعات الصغرى العادية — بدون عقوبة
    ]),
    'Ridge (L2)': Pipeline([
        ('poly', PolynomialFeatures(degree=POLY_DEG, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=10.0, random_state=42))  # عقوبة L2 — يقلّص المعاملات بسلاسة
    ]),
    'Lasso (L1)': Pipeline([
        ('poly', PolynomialFeatures(degree=POLY_DEG, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=500.0, max_iter=50000, random_state=42))  # L1 — قد يُصفّر بعض الحدود
    ]),
    'Elastic Net': Pipeline([
        ('poly', PolynomialFeatures(degree=POLY_DEG, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=50000, random_state=42))  # مزيج L1 + L2
    ])
}

fitted_models = {}  # تخزين خطوط الأنابيب المُدرَّبة
for name, pipe in models.items():
    pipe.fit(X_train, y_train)  # ملاءمة المعالجة + النموذج على بيانات التدريب
    fitted_models[name] = pipe    # حفظ خط الأنابيب المُدرَّب
    print(f'{name}: تم التدريب')   # تأكيد جاهزية كل نموذج
""",

    20: """## المثال 1 — الخلية 6: التنبؤ

إنشاء تنبؤات من جميع النماذج الأربعة على مجموعات التدريب والاختبار.

**ما تفعله هذه الخلية:** استدعاء `.predict()` على كل خط أنابيب مُدرَّب.
""",

    21: """# الخطوة 6) التنبؤ بجميع النماذج

predictions = {}  # قاموس لتخزين y_pred لكل نموذج
for name, pipe in fitted_models.items():
    predictions[name] = {
        'train': pipe.predict(X_train),  # تنبؤات التدريب
        'test': pipe.predict(X_test)     # تنبؤات الاختبار
    }

print('عينة من تنبؤات الاختبار (Level -> Salary):')  # طباعة تسمية
for i in range(len(y_test)):
    level = X_test[i][0]
    print(f'  Level={level:.0f} | الفعلي=${y_test[i]:,.0f}')
    for name in fitted_models:
        pred = predictions[name]['test'][i]
        print(f'    {name:15s} -> ${pred:,.0f}')
""",

    22: """## المثال 1 — الخلية 7: التصوير

رسم منحنيات التنبؤ لجميع النماذج الأربعة — لرؤية كيف يُنعّم التنظيم فرط التجهيز.

**ما تفعله هذه الخلية:** إنشاء subplot 2×2 لمقارنة OLS مقابل Ridge مقابل Lasso مقابل Elastic Net.
""",

    23: """# الخطوة 7) التصوير — مقارنة منحنيات التنبؤ

X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)  # محور x سلس للمنحنيات
plot_order = ['No Reg (OLS)', 'Ridge (L2)', 'Lasso (L1)', 'Elastic Net']
colors = {'No Reg (OLS)': 'crimson', 'Ridge (L2)': 'seagreen', 'Lasso (L1)': 'darkorange', 'Elastic Net': 'purple'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # أربعة رسوم فرعية
axes = axes.ravel()  # تسطيح إلى 1D للفهرسة السهلة

for ax, name in zip(axes, plot_order):
    pipe = fitted_models[name]
    y_plot = pipe.predict(X_plot)  # منحنى النموذج على شبكة كثيفة
    ax.scatter(X_train, y_train, color='blue', s=70, label='تدريب', zorder=3)   # نقاط التدريب
    ax.scatter(X_test, y_test, color='green', s=70, label='اختبار', zorder=3)     # نقاط الاختبار
    ax.plot(X_plot, y_plot, color=colors[name], linewidth=2, label=name)        # منحنى التنبؤ
    ax.set_xlabel('Level')           # تسمية المحور x
    ax.set_ylabel('Salary (USD)')   # تسمية المحور y
    ax.set_title(f'{name} (درجة={POLY_DEG})')  # عنوان الرسم الفرعي
    ax.legend(fontsize=8)            # وسيلة الإيضاح

plt.suptitle('أثر التنظيم — رواتب المناصب (كثير الحدود)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
""",

    24: """## المثال 1 — الخلية 8: التقييم

مقارنة **R² تدريب مقابل R² اختبار** لجميع النماذج — فرط التجهيز يظهر كـ R² تدريب مرتفع و R² اختبار منخفض.

**ما تفعله هذه الخلية:** حساب MAE و RMSE و R² لكل نموذج على مجموعة الاختبار.
""",

    25: """# الخطوة 8) التقييم — مقارنة جميع النماذج

rows = []
for name, pipe in fitted_models.items():
    y_tr_pred = predictions[name]['train']  # تنبؤات التدريب
    y_te_pred = predictions[name]['test']   # تنبؤات الاختبار
    rows.append({
        'النموذج': name,
        'R² تدريب': r2_score(y_train, y_tr_pred),
        'R² اختبار': r2_score(y_test, y_te_pred),
        'MAE اختبار': mean_absolute_error(y_test, y_te_pred),
        'RMSE اختبار': np.sqrt(mean_squared_error(y_test, y_te_pred))
    })

results = pd.DataFrame(rows)
display(results.round(4))  # عرض جدول المقارنة

best_name = results.loc[results['R² اختبار'].idxmax(), 'النموذج']  # النموذج ذو أعلى R² اختبار
r2 = results.loc[results['R² اختبار'].idxmax(), 'R² اختبار']       # أفضل قيمة R² اختبار
print(f'\\nأفضل نموذج على مجموعة الاختبار: {best_name}')
print(f'المثال 1 أفضل R² اختبار = {r2:.4f}')
""",

    26: """## لماذا يعمل التنظيم لرواتب المناصب؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **فرط تجهيز كثير الحدود** | درجة 7 على 10 صفوف — OLS يصل إلى R² تدريب = 1.0 (يحفظ كل نقطة) |
| 2 | **Ridge ينعّم المنحنى** | عقوبة L2 تقلّص معاملات كثير الحدود الكبيرة — تنبؤات أنعم |
| 3 | **Lasso يبسّط** | L1 قد يُصفّر حدود x³ أو x⁴ — يبقي القوى المفيدة فقط |
| 4 | **القياس ضروري** | حدود x⁴ كبيرة بدون StandardScaler — العقوبة ستكون غير عادلة |
| 5 | **فجوة تدريب مقابل اختبار** | OLS غالبًا R² تدريب ≈ 1 لكن R² اختبار أسوأ — التنظيم يُغلق الفجوة |

> **الخلاصة:** على بيانات صغيرة + درجة كثير حدود عالية، **النماذج المُنظَّمة تُعمّم أفضل** من OLS العادي.
""",

    27: """## فهم R² — المثال 1 (رواتب المناصب)

قارن **R² تدريب** مقابل **R² اختبار** عبر النماذج:

| النمط | المعنى |
|---------|---------|
| R² تدريب مرتفع، R² اختبار منخفض | **فرط التجهيز** — OLS بدون تنظيم |
| R² تدريب واختبار معقولان | **تعميم جيد** — Ridge/Lasso/Elastic Net |
| كلاهما R² منخفض | **نقص التجهيز** — alpha كبير جدًا |

> مع **عينتين اختبار فقط**، ركّز على **شكل المنحنى** وفجوة التدريب/الاختبار، وليس على رقم R² واحد فقط.
""",

    28: """---
# المثال 2: أرباح الشركات الناشئة — التنظيم مع متغيرات متعددة

التنبؤ بـ **Profit** من أعمدة الإنفاق و **State** باستخدام Ridge و Lasso و Elastic Net على **عدد كبير من المتغيرات المُرمَّزة**.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `R&D Spend` | متغير (X₁) | ميزانية البحث والتطوير |
| `Administration` | متغير (X₂) | المصاريف الإدارية |
| `Marketing Spend` | متغير (X₃) | ميزانية التسويق |
| `State` | متغير (X₄) | ولاية أمريكية (ترميز One-Hot) |
| `Profit` | الهدف (y) | الربح السنوي بالدولار الأمريكي |

**الملف:** `../Datasets/50_Startups.csv`
""",

    29: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV الشركات الناشئة وفحص البيانات.

**ما تفعله هذه الخلية:** قراءة `../Datasets/50_Startups.csv` وعرض إحصائيات أساسية.
""",

    30: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/50_Startups.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['R&D Spend', 'Administration', 'Marketing Spend']  # متغيرات عددية
CAT_COL = 'State'                                                   # متغير فئوي
TARGET_COL = 'Profit'                                               # المتغير الهدف

print('أول 5 صفوف:')
display(dataset.head())

print('\\nمعلومات مجموعة البيانات:')
dataset.info()

print('\\nملخص إحصائي:')
display(dataset.describe())

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')
""",

    31: """## المثال 2 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

التحقق من القيم المفقودة وإزالة التكرارات.

**ما تفعله هذه الخلية:** تنظيف مجموعة البيانات قبل النمذجة.
""",

    32: """# الخطوة 2) تنظيف البيانات

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
    print('تم إكمال القيم المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق الإكمال')

print(f'\\nالصفوف بعد التنظيف: {rows_after}')
""",

    33: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

ترميز **State** بـ **One-Hot Encoding** (حذف الأول لتجنب فخ المتغير الوهمي).

**ما تفعله هذه الخلية:** إنشاء أعمدة وهمية لـ State.
""",

    34: """# الخطوة 3) الترميز الفئوي — One-Hot لـ State

dataset_encoded = pd.get_dummies(dataset, columns=[CAT_COL], drop_first=True)  # State -> أعمدة ثنائية
feature_cols = [c for c in dataset_encoded.columns if c != TARGET_COL]  # جميع الأعمدة ما عدا Profit

print(f'الأعمدة الأصلية: {list(dataset.columns)}')
print(f'أعمدة المتغيرات المُرمَّزة ({len(feature_cols)}): {feature_cols}')
""",

    35: """## المثال 2 — الخلية 4: تقسيم البيانات

تعريف X (المتغيرات المُرمَّزة) و y (Profit)، ثم تقسيم 80/20.

**ما تفعله هذه الخلية:** إنشاء مصفوفات المتغيرات/الهدف وتطبيق train_test_split.
""",

    36: """# الخطوة 4) تقسيم تدريب-اختبار

X = dataset_encoded[feature_cols].values  # مصفوفة المتغيرات بعد الترميز
y = dataset_encoded[TARGET_COL].values    # متجه الهدف: Profit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'شكل X_train: {X_train.shape}')  # (n_train, n_features)
print(f'شكل X_test:  {X_test.shape}')
print(f'شكل y_train: {y_train.shape}')
print(f'شكل y_test:  {y_test.shape}')
""",

    37: """## المثال 2 — الخلية 5: بناء خطوط الأنابيب وتدريب النماذج

تدريب **Ridge** و **Lasso** و **Elastic Net** مع StandardScaler. مقارنة مع خط أساس OLS.

**ما تفعله هذه الخلية:** ملاءمة أربعة نماذج خطية مُقاسة على بيانات الشركات الناشئة.
""",

    38: """# الخطوة 5) بناء وتدريب النماذج المُنظَّمة (متعددة المتغيرات)

models_ex2 = {
    'No Reg (OLS)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge (L2)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ]),
    'Lasso (L1)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=100.0, max_iter=50000, random_state=42))
    ]),
    'Elastic Net': Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000, random_state=42))
    ])
}

fitted_ex2 = {}
for name, pipe in models_ex2.items():
    pipe.fit(X_train, y_train)
    fitted_ex2[name] = pipe
    print(f'{name}: تم التدريب')
""",

    39: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بـ Profit على مجموعة الاختبار بجميع النماذج.

**ما تفعله هذه الخلية:** إنشاء التنبؤات وعرض عينات من المخرجات.
""",

    40: """# الخطوة 6) التنبؤ

pred_ex2 = {}
for name, pipe in fitted_ex2.items():
    pred_ex2[name] = pipe.predict(X_test)

print('عينة من التنبؤات (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    print(f'  Profit الفعلي=${y_test[i]:,.0f}')
    for name in fitted_ex2:
        print(f'    {name:15s} -> ${pred_ex2[name][i]:,.0f}')
""",

    41: """## المثال 2 — الخلية 7: التصوير

رسم **الفعلي مقابل المتنبأ** (أفضل نموذج) و **مقارنة المعاملات** عبر طرق التنظيم.

**ما تفعله هذه الخلية:** إظهار اختيار متغيرات Lasso عبر معاملات صفرية.
""",

    42: """# الخطوة 7) التصوير — المعاملات + الفعلي مقابل المتنبأ

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# --- يسار: مقارنة المعاملات (بعد القياس، من كل نموذج) ---
coef_data = {}
for name, pipe in fitted_ex2.items():
    model = pipe.named_steps['model']  # استخراج النموذج الخطي من خط الأنابيب
    coef_data[name] = model.coef_       # متجه المعاملات

coef_df = pd.DataFrame(coef_data, index=feature_cols)  # الصفوف = المتغيرات، الأعمدة = النماذج
coef_df.plot(kind='barh', ax=axes[0], width=0.8)         # مخطط أشرطة أفقي
axes[0].set_xlabel('قيمة المعامل (متغيرات مُقاسة)')
axes[0].set_title('مقارنة المعاملات — طرق التنظيم')
axes[0].axvline(0, color='black', linewidth=0.8)  # خط الصفر — Lasso قد يصل إليه

# --- يمين: الفعلي مقابل المتنبأ (Ridge — عادةً قوي على هذه البيانات) ---
best_pipe = fitted_ex2['Ridge (L2)']
y_pred_ridge = best_pipe.predict(X_test)
axes[1].scatter(y_test, y_pred_ridge, color='seagreen', alpha=0.7)
min_val = min(y_test.min(), y_pred_ridge.min())
max_val = max(y_test.max(), y_pred_ridge.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='تنبؤ مثالي')
axes[1].set_xlabel('Profit الفعلي (USD)')
axes[1].set_ylabel('Profit المتنبأ (USD)')
axes[1].set_title('الفعلي مقابل المتنبأ — Ridge (L2)')
axes[1].legend()

plt.tight_layout()
plt.show()

# عد المعاملات غير الصفرية (اختيار متغيرات Lasso)
print('\\nالمعاملات غير الصفرية لكل نموذج:')
for name, pipe in fitted_ex2.items():
    coefs = pipe.named_steps['model'].coef_
    n_nonzero = int(np.sum(np.abs(coefs) > 1e-6))
    print(f'  {name:15s}: {n_nonzero} / {len(coefs)} متغير')
""",

    43: """## المثال 2 — الخلية 8: التقييم

مقارنة جميع النماذج بـ MAE و RMSE و R² على مجموعة الاختبار.

**ما تفعله هذه الخلية:** حساب وعرض مقاييس التقييم.
""",

    44: """# الخطوة 8) التقييم
rows = []
for name, pipe in fitted_ex2.items():
    y_pred = pipe.predict(X_test)
    rows.append({
        'النموذج': name,
        'R² اختبار': r2_score(y_test, y_pred),
        'MAE اختبار': mean_absolute_error(y_test, y_pred),
        'RMSE اختبار': np.sqrt(mean_squared_error(y_test, y_pred))
    })

results = pd.DataFrame(rows)
display(results.round(4))

best_name = results.loc[results['R² اختبار'].idxmax(), 'النموذج']
r2 = results.loc[results['R² اختبار'].idxmax(), 'R² اختبار']
print(f'\\nأفضل نموذج على مجموعة الاختبار: {best_name}')
print(f'المثال 2 أفضل R² اختبار = {r2:.4f}')
""",

    45: """## لماذا يعمل التنظيم لأرباح الشركات الناشئة؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **إنفاقات متعددة مترابطة** | R&D والتسويق قد يترابطان — Ridge يُثبّت المعاملات |
| 2 | **أعمدة State بـ One-Hot** | تضيف متغيرات إضافية — Lasso قد يحذف ولايات أقل فائدة |
| 3 | **قياس المتغيرات** | أعمدة الإنفاق بمقاييس مختلفة — StandardScaler قبل العقوبة |
| 4 | **تقليص المعاملات** | يمنع أوزانًا متطرفة على متغيرات ذات ضوضاء |
| 5 | **R² مشابه لـ OLS** | على هذه البيانات OLS يُلائم جيدًا — التنظيم يضيف استقرارًا |

> **قارن:** Lasso قد يجعل بعض معاملات State الوهمية **صفرًا** — اختيار تلقائي للمتغيرات.
""",

    46: """## فهم R² — المثال 2 (أرباح الشركات الناشئة)

| قيمة R² | المعنى |
|----------|---------|
| **> 0.90** | ممتاز — الإنفاق + الولاية يفسّران معظم تباين الربح |
| **0.70–0.90** | ملاءمة جيدة |
| **< 0.50** | ضعيف — جرّب متغيرات أكثر أو اضبط alpha |

> على هذه البيانات، **Ridge/Lasso/Elastic Net** غالبًا تطابق R² لـ OLS مع إنتاج معاملات **أصغر وأكثر استقرارًا** — الفائدة الرئيسية هي **التعميم**، وليس دائمًا R² أعلى على نفس التقسيم.
""",
}
