CELLS = {
    0: """# الانحدار الخطي البسيط — Google Colab

**الهدف:** التنبؤ بقيمة مستهدفة متصلة من ميزة رقمية واحدة باستخدام الانحدار الخطي البسيط.

| مثال | الميزة (X) | الهدف (y) | مجموعة البيانات |
|---------|-------------|------------|---------|
| **المثال 1** | SAT | GPA | `../Datasets/Simple linear regression.csv` |
| **المثال 2** | YearsExperience | Salary | `../Datasets/experience_salary.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | الإعداد والاستيراد |
| المرحلة 1 | معالجة البيانات | التحميل ← التنظيف ← الترميز ← التقسيم |
| المرحلة 2 | الخوارزمية | التدريب ← التنبؤ ← التصور ← التقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)""",

    1: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    2: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    3: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات لمعالجة البيانات والتحضير والنمذجة والتقييم.

**ما تفعله هذه الخلية:** تحمّل numpy و pandas و matplotlib و seaborn ووحدات sklearn.""",

    4: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء المخططات والرسوم
import seaborn as sns           # تصورات إحصائية (تنسيق اختياري)

from sklearn.model_selection import train_test_split  # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer              # ملء القيم المفقودة
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # ترميز الفئات
from sklearn.compose import ColumnTransformer         # تطبيق التحويلات على الأعمدة
from sklearn.linear_model import LinearRegression     # نموذج الانحدار الخطي البسيط
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # مقاييس التقييم

plt.rcParams['figure.figsize'] = (10, 6)  # حجم المخطط الافتراضي: عرض=10، ارتفاع=6 بوصة
plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لتقسيمات قابلة للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح""",

    5: """---
# المرحلة 1: معالجة البيانات

تحضير البيانات قبل تدريب النموذج — نفس القالب يُعاد استخدامه لخوارزميات أخرى.""",

    6: """## المرحلة 1 — الخلية 1: تحميل واستكشاف البيانات

تحميل ملف CSV وإجراء استكشاف أولي (head، info، describe، shape).

**ما تفعله هذه الخلية:** تقرأ `../Datasets/Simple linear regression.csv` وتعرض إحصائيات أساسية.""",

    7: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/Simple linear regression.csv')  # قراءة CSV إلى DataFrame

print('أول 5 صفوف:')          # طباعة تسمية للجدول أدناه
display(dataset.head())         # عرض أول 5 صفوف لفحص البيانات

print('\\nمعلومات مجموعة البيانات:')       # طباعة تسمية لأنواع الأعمدة وعدد القيم الفارغة
dataset.info()                  # عرض أسماء الأعمدة وأنواع البيانات وعدد القيم غير الفارغة

print('\\nملخص إحصائي:')  # طباعة تسمية للإحصائيات العددية
display(dataset.describe())       # عرض العدد والمتوسط والانحراف المعياري والحد الأدنى والأقصى والربيعيات

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')  # إجمالي الصفوف والأعمدة""",

    8: """## المرحلة 1 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

يشمل تنظيف البيانات التحقق من القيم المفقودة وإزالة التكرارات واستخدام SimpleImputer عند الحاجة.

**ما تفعله هذه الخلية:** تتحقق من القيم الفارغة، تحذف الصفوف المكررة، وتجهّز قالب imputer لمجموعات بيانات أخرى.""",

    9: """# الخطوة 2) تنظيف البيانات

# 1) التحقق من القيم المفقودة
print('القيم المفقودة لكل عمود:')  # طباعة تسمية
print(dataset.isnull().sum())        # عد قيم NaN في كل عمود

# 2) إزالة الصفوف المكررة
rows_before = len(dataset)                              # تخزين عدد الصفوف قبل التنظيف
dataset = dataset.drop_duplicates().reset_index(drop=True)  # إزالة التكرارات وإعادة تعيين فهرس الصف
rows_after = len(dataset)                               # تخزين عدد الصفوف بعد التنظيف
print(f'\\nالتكرارات المحذوفة: {rows_before - rows_after}')  # عرض عدد التكرارات المحذوفة

# 3) SimpleImputer — قالب قابل لإعادة الاستخدام لمجموعات بيانات أخرى
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # استبدال NaN بمتوسط العمود
if dataset.isnull().sum().sum() > 0:               # إذا وُجدت قيم مفقودة
    dataset.iloc[:, :] = imputer.fit_transform(dataset)  # تعلّم المتوسطات وملء القيم المفقودة
    print('تم استبدال القيم المفقودة بالمتوسط')      # تأكيد تطبيق الاستبدال
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُستبدِل')  # تخطي الاستبدال عند اكتمال البيانات

print(f'\\nالصفوف بعد التنظيف: {rows_after}')    # العدد النهائي للصفوف في مجموعة البيانات""",

    10: """## المرحلة 1 — الخلية 3: ترميز البيانات الفئوية

تتطلب خوارزميات التعلم الآلي مدخلات رقمية. تُرمَّز الأعمدة الفئوية بـ One-Hot أو Label Encoding. بالنسبة لـ SAT/GPA، جميع الأعمدة رقمية، لذا يُتخطَّى الترميز.

**ما تفعله هذه الخلية:** تكتشف الأعمدة الفئوية وتطبّق OneHotEncoder عند الحاجة؛ وإلا تتخطى.""",

    11: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()  # إيجاد أعمدة النص/الفئة
num_cols = dataset.select_dtypes(exclude=['object', 'category']).columns.tolist()  # إيجاد الأعمدة الرقمية

if cat_cols:  # تشغيل الترميز فقط عند وجود أعمدة فئوية
    print(f'أعمدة فئوية موجودة: {cat_cols}')  # سرد الأعمدة التي تحتاج ترميزاً
    preprocessor = ColumnTransformer(  # بناء خط أنابيب للأعمدة المحددة
        transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)],  # ترميز one-hot للفئات
        remainder='passthrough'  # الإبقاء على الأعمدة الأخرى دون تغيير
    )
    dataset = pd.DataFrame(  # تحويل المخرجات المرمّزة إلى DataFrame
        preprocessor.fit_transform(dataset),  # تعلّم الفئات وتحويل البيانات
        columns=preprocessor.get_feature_names_out()  # استخدام أسماء الأعمدة المرمّزة الجديدة
    )
    print('تم تطبيق ترميز One-Hot')  # تأكيد اكتمال الترميز
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')  # لا شيء للترميز في مجموعة SAT/GPA
    print(f'الأعمدة الرقمية: {num_cols}')  # عرض الأعمدة الرقمية المستخدمة في النموذج""",

    12: """## المرحلة 1 — الخلية 4: تقسيم البيانات

تحديد الميزة X (SAT) والهدف y (GPA)، ثم التقسيم إلى 80% تدريب و 20% اختبار.

**ما تفعله هذه الخلية:** تنشئ مصفوفات X و y وتطبّق train_test_split مع random_state=42.""",

    13: """# الخطوة 4) تقسيم التدريب-الاختبار

# X = الميزة (SAT)
X = dataset[['SAT']].values  # مصفوفة الميزات المدخلة: درجات SAT (مصفوفة ثنائية الأبعاد مطلوبة من sklearn)

# y = الهدف (GPA)
y = dataset['GPA'].values    # متجه الهدف: قيم GPA للتنبؤ بها

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                  # البيانات للتقسيم
    test_size=0.2,         # 20% للاختبار، 80% للتدريب
    random_state=42        # بذرة ثابتة لتقسيم قابل للتكرار في كل تشغيل
)

print(f'شكل X_train: {X_train.shape}')  # شكل ميزات التدريب: (n_train, 1)
print(f'شكل X_test:  {X_test.shape}')   # شكل ميزات الاختبار: (n_test, 1)
print(f'شكل y_train: {y_train.shape}')  # شكل أهداف التدريب: (n_train,)
print(f'شكل y_test:  {y_test.shape}')   # شكل أهداف الاختبار: (n_test,)""",

    14: """> **ملاحظة:** الانحدار الخطي البسيط **لا يتطلب** توسيع الميزات (scaling) مع ميزة واحدة. التوسيع مطلوب لـ SVR و KNN.""",

    15: """---
# المرحلة 2: الانحدار الخطي البسيط

تطبيق خوارزمية الانحدار الخطي البسيط: `GPA = b0 + b1 x SAT`""",

    16: """## المرحلة 2 — الخلية 5: تدريب النموذج

تدريب `LinearRegression` على بيانات التدريب. يتعلّم النموذج الميل (b1) والمقطع (b0).

**ما تفعله هذه الخلية:** يُلائم نموذج الانحدار ويطبع معاملات المقطع والميل.""",

    17: """# الخطوة 5) تدريب النموذج
regressor = LinearRegression()      # إنشاء نموذج انحدار خطي بسيط فارغ
regressor.fit(X_train, y_train)     # تدريب النموذج: تعلّم أفضل b0 (مقطع) و b1 (ميل)

print(f'المقطع (b0): {regressor.intercept_:.4f}')   # الحد الثابت عند SAT = 0
print(f'الميل (b1):       {regressor.coef_[0]:.6f}')  # التغيّر في GPA لكل زيادة نقطة واحدة في SAT
print(f'\\nالمعادلة: GPA = {regressor.intercept_:.4f} + {regressor.coef_[0]:.6f} x SAT')  # معادلة الانحدار النهائية""",

    18: """## المرحلة 2 — الخلية 6: التنبؤ

بعد التدريب، يتنبأ النموذج بقيم GPA لمجموعتي التدريب والاختبار.

**ما تفعله هذه الخلية:** يُنشئ y_pred_train و y_pred_test باستخدام النموذج المُدرَّب.""",

    19: """# الخطوة 6) التنبؤ
y_pred_train = regressor.predict(X_train)  # التنبؤ بـ GPA لدرجات SAT في التدريب
y_pred_test = regressor.predict(X_test)    # التنبؤ بـ GPA لدرجات SAT غير المرئية في الاختبار

print('عينات تنبؤ (مجموعة الاختبار):')  # طباعة تسمية
for i in range(min(5, len(y_test))):     # حلقة على أول 5 عينات اختبار
    print(f'  SAT={X_test[i][0]:.0f} -> GPA فعلي={y_test[i]:.2f}، متوقع={y_pred_test[i]:.2f}')  # مقارنة الفعلي مقابل المتوقع""",

    20: """## المرحلة 2 — الخلية 7: التصور

مخطط انتشار للبيانات مع خط الانحدار — التدريب (أزرق)، الاختبار (أخضر)، الخط (أحمر).

**ما تفعله هذه الخلية:** يرسم SAT مقابل GPA مع خط الانحدار المُلائَم.""",

    21: """# الخطوة 7) التصور
plt.figure(figsize=(10, 6))  # إنشاء شكل جديد بحجم مخصص
plt.scatter(X_train, y_train, color='blue', label='مجموعة التدريب', alpha=0.7)  # رسم نقاط التدريب
plt.scatter(X_test, y_test, color='green', label='مجموعة الاختبار', alpha=0.7)       # رسم نقاط الاختبار

# خط الانحدار على كامل نطاق SAT
X_line = np.linspace(dataset['SAT'].min(), dataset['SAT'].max(), 100).reshape(-1, 1)  # 100 قيمة SAT من الحد الأدنى إلى الأقصى
y_line = regressor.predict(X_line)  # التنبؤ بـ GPA لكل قيمة SAT على الخط
plt.plot(X_line, y_line, color='red', linewidth=2, label='خط الانحدار')  # رسم خط الانحدار المُلائَم

plt.xlabel('درجة SAT')  # تسمية المحور السيني
plt.ylabel('GPA')       # تسمية المحور الصادي
plt.title('الانحدار الخطي البسيط — SAT مقابل GPA')  # عنوان المخطط
plt.legend()             # عرض وسيلة الإيضاح (التدريب، الاختبار، خط الانحدار)
plt.tight_layout()       # ضبط المسافات حتى لا تُقص التسميات
plt.show()               # عرض المخطط""",

    22: """## المرحلة 2 — الخلية 8: التقييم

تقييم النموذج على مجموعة الاختبار باستخدام MAE (متوسط الخطأ المطلق) و RMSE (جذر متوسط مربع الخطأ) و R² (معامل التحديد).

**ما تفعله هذه الخلية:** يحسب ويعرض مقاييس تقييم الانحدار.""",

    23: """# الخطوة 8) التقييم
mae = mean_absolute_error(y_test, y_pred_test)              # متوسط الخطأ المطلق في التنبؤ
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))     # الجذر التربيعي لمتوسط مربع الخطأ
r2 = r2_score(y_test, y_pred_test)                          # نسبة التباين التي يفسّرها النموذج

results = pd.DataFrame({  # بناء جدول مقاييس التقييم
    'Metric': ['MAE', 'RMSE', 'R²'],  # أسماء المقاييس
    'Value': [mae, rmse, r2],         # قيم المقاييس المحسوبة
    'Description': [                   # شرح مختصر لكل مقياس
        'متوسط الخطأ المطلق',
        'جذر متوسط مربع الخطأ',
        'معامل التحديد (1 = مثالي)'
    ]
})

display(results.round(4))  # عرض المقاييس مقربة إلى 4 خانات عشرية""",

    24: """## لماذا R² ≈ 0.3782؟

**R² = 0.3782** يعني أن النموذج يفسّر فقط نحو **38%** من التباين في GPA. الأسباب المحتملة:

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **ميزة واحدة (SAT فقط)** | GPA يتأثر بعوامل كثيرة (عادات الدراسة، المقررات، الحضور، المدرسة...) — SAT وحده لا يكفي |
| 2 | **العلاقة ليست خطية تماماً** | يفترض الانحدار الخطي البسيط خطاً مستقيماً؛ قد تكون العلاقة الحقيقية منحنية |
| 3 | **ضجيج في البيانات** | طلاب بنفس SAT قد يحصلون على GPA مختلف — هذا يخفض R² |
| 4 | **حذف الصفوف المكررة** | بعد التنظيف، يتبقى 83 صفاً فقط — عينة أصغر قد تؤثر على الاستقرار |
| 5 | **القيم الشاذة** | بعض الطلاب لديهم SAT عالٍ و GPA منخفض (أو العكس) — تبعد النقاط عن الخط |
| 6 | **نموذج بسيط (SLR)** | نموذج خطي بميزة واحدة قد لا يلتقط كل الأنماط — ليس بالضرورة نقص ملاءمة، بل **ملاءمة متوسطة** |

> **الخلاصة:** R² = 0.3782 = **ملاءمة متوسطة** — يتعلّم النموذج اتجاهاً (SAT ↑ → GPA ↑) لكنه لا يتنبأ بدقة عالية.""",

    25: """---
# المثال 2: الخبرة مقابل الراتب — مجموعة بيانات مثالية للانحدار الخطي البسيط

صُممت هذه المجموعة لـ **الانحدار الخطي البسيط**:

| العمود | الدور | الوصف |
|--------|------|-------------|
| `YearsExperience` | الميزة (X) | سنوات الخبرة العملية |
| `Salary` | الهدف (y) | الراتب السنوي بالدولار الأمريكي |

**العلاقة الحقيقية:** `Salary ≈ 25000 + 9500 × YearsExperience` (+ ضجيج بسيط)

**الملف:** `../Datasets/experience_salary.csv`""",

    26: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV الخبرة مقابل الراتب وفحص البيانات.

**ما تفعله هذه الخلية:** تقرأ `../Datasets/experience_salary.csv` وتعرض إحصائيات أساسية.""",

    27: """# تحميل مجموعة البيانات المناسبة للانحدار الخطي البسيط
dataset = pd.read_csv('../Datasets/experience_salary.csv')  # قراءة CSV إلى DataFrame

FEATURE_COL = 'YearsExperience'  # المتغير المستقل (X)
TARGET_COL = 'Salary'              # المتغير التابع (y)

print('أول 5 صفوف:')          # طباعة تسمية للجدول أدناه
display(dataset.head())         # عرض أول 5 صفوف لفحص البيانات

print('\\nمعلومات مجموعة البيانات:')       # طباعة تسمية لأنواع الأعمدة وعدد القيم الفارغة
dataset.info()                  # عرض أسماء الأعمدة وأنواع البيانات وعدد القيم غير الفارغة

print('\\nملخص إحصائي:')  # طباعة تسمية للإحصائيات العددية
display(dataset.describe())       # عرض العدد والمتوسط والانحراف المعياري والحد الأدنى والأقصى والربيعيات

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')  # إجمالي الصفوف والأعمدة""",

    28: """## المثال 2 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

تتضمن هذه المجموعة بعض القيم المفقودة لعرض `SimpleImputer`.

**ما تفعله هذه الخلية:** تتحقق من القيم الفارغة، تزيل التكرارات، وتستبدل القيم المفقودة.""",

    29: """# تنظيف البيانات للخبرة مقابل الراتب

print('القيم المفقودة لكل عمود:')  # طباعة تسمية
print(dataset.isnull().sum())        # عد قيم NaN في كل عمود

rows_before = len(dataset)                              # تخزين عدد الصفوف قبل التنظيف
dataset = dataset.drop_duplicates().reset_index(drop=True)  # إزالة الصفوف المكررة
rows_after = len(dataset)                               # تخزين عدد الصفوف بعد إزالة التكرار
print(f'\\nالتكرارات المحذوفة: {rows_before - rows_after}')  # عرض عدد التكرارات

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # ملء NaN بمتوسط العمود
if dataset.isnull().sum().sum() > 0:               # إذا وُجدت قيم مفقودة
    dataset[[FEATURE_COL, TARGET_COL]] = imputer.fit_transform(dataset[[FEATURE_COL, TARGET_COL]])  # استبدال الأعمدة الرقمية
    print('تم استبدال القيم المفقودة بالمتوسط')      # تأكيد الاستبدال
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُستبدِل')  # تخطي عند اكتمال البيانات

print(f'\\nالصفوف بعد التنظيف: {rows_after}')    # العدد النهائي للصفوف""",

    30: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

كلا العمودين رقميان — يُتخطَّى الترميز.

**ما تفعله هذه الخلية:** تكتشف الأعمدة الفئوية وتطبّق الترميز عند الحاجة.""",

    31: """# الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()  # إيجاد أعمدة النص/الفئة
num_cols = dataset.select_dtypes(exclude=['object', 'category']).columns.tolist()  # إيجاد الأعمدة الرقمية

if cat_cols:  # الترميز فقط عند وجود أعمدة فئوية
    print(f'أعمدة فئوية موجودة: {cat_cols}')
    preprocessor = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )
    dataset = pd.DataFrame(
        preprocessor.fit_transform(dataset),
        columns=preprocessor.get_feature_names_out()
    )
    print('تم تطبيق ترميز One-Hot')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'الأعمدة الرقمية: {num_cols}')""",

    32: """## المثال 2 — الخلية 4: تقسيم البيانات

تحديد X (`YearsExperience`) و y (`Salary`)، ثم التقسيم 80/20.

**ما تفعله هذه الخلية:** تنشئ مصفوفات الميزة/الهدف وتطبّق train_test_split.""",

    33: """# تقسيم التدريب-الاختبار

X = dataset[[FEATURE_COL]].values  # مصفوفة الميزات: سنوات الخبرة
y = dataset[TARGET_COL].values     # متجه الهدف: قيم الراتب

X_train, X_test, y_train, y_test = train_test_split(
    X, y,                  # البيانات للتقسيم
    test_size=0.2,         # 20% اختبار، 80% تدريب
    random_state=42        # تقسيم قابل للتكرار
)

print(f'شكل X_train: {X_train.shape}')  # شكل ميزات التدريب
print(f'شكل X_test:  {X_test.shape}')   # شكل ميزات الاختبار
print(f'شكل y_train: {y_train.shape}')  # شكل أهداف التدريب
print(f'شكل y_test:  {y_test.shape}')   # شكل أهداف الاختبار""",

    34: """## المثال 2 — الخلية 5: تدريب النموذج

تدريب الانحدار الخطي البسيط: `Salary = b0 + b1 × YearsExperience`

**ما تفعله هذه الخلية:** يُلائم النموذج ويطبع معادلة الانحدار.""",

    35: """# تدريب النموذج
regressor = LinearRegression()      # إنشاء نموذج انحدار خطي بسيط
regressor.fit(X_train, y_train)     # تعلّم المقطع (b0) والميل (b1)

print(f'المقطع (b0): {regressor.intercept_:.2f}')    # الراتب الأساسي عند الخبرة = 0
print(f'الميل (b1):       {regressor.coef_[0]:.2f}')   # زيادة الراتب لكل سنة إضافية
print(f'\\nالمعادلة: {TARGET_COL} = {regressor.intercept_:.2f} + {regressor.coef_[0]:.2f} x {FEATURE_COL}')""",

    36: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بالراتب لقيم الخبرة في التدريب والاختبار.

**ما تفعله هذه الخلية:** يُنشئ التنبؤات ويعرض مقارنات عينة.""",

    37: """# التنبؤ
y_pred_train = regressor.predict(X_train)  # التنبؤ بالراتب لبيانات التدريب
y_pred_test = regressor.predict(X_test)    # التنبؤ بالراتب لبيانات الاختبار

print('عينات تنبؤ (مجموعة الاختبار):')
for i in range(min(5, len(y_test))):
    print(f'  {FEATURE_COL}={X_test[i][0]:.1f} -> فعلي={y_test[i]:,.0f}، متوقع={y_pred_test[i]:,.0f}')""",

    38: """## المثال 2 — الخلية 7: التصور

مخطط انتشار مع خط الانحدار للخبرة مقابل الراتب.

**ما تفعله هذه الخلية:** يرسم نقاط البيانات والخط المُلائَم.""",

    39: """# التصور
plt.figure(figsize=(10, 6))  # إنشاء شكل
plt.scatter(X_train, y_train, color='blue', label='مجموعة التدريب', alpha=0.7)  # نقاط التدريب
plt.scatter(X_test, y_test, color='green', label='مجموعة الاختبار', alpha=0.7)       # نقاط الاختبار

X_line = np.linspace(dataset[FEATURE_COL].min(), dataset[FEATURE_COL].max(), 100).reshape(-1, 1)  # قيم x للخط
y_line = regressor.predict(X_line)  # قيم y للخط من النموذج
plt.plot(X_line, y_line, color='red', linewidth=2, label='خط الانحدار')  # رسم خط الانحدار

plt.xlabel('سنوات الخبرة')  # تسمية المحور السيني
plt.ylabel('الراتب (USD)')          # تسمية المحور الصادي
plt.title('الانحدار الخطي البسيط — الخبرة مقابل الراتب')  # عنوان المخطط
plt.legend()             # عرض وسيلة الإيضاح
plt.tight_layout()       # ضبط التخطيط
plt.show()               # عرض المخطط""",

    40: """## المثال 2 — الخلية 8: التقييم

تقييم أداء النموذج بـ MAE و RMSE و R².

**ما تفعله هذه الخلية:** يحسب ويعرض مقاييس التقييم على مجموعة الاختبار.""",

    41: """# التقييم
mae = mean_absolute_error(y_test, y_pred_test)              # متوسط الخطأ المطلق بالدولار
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))     # جذر متوسط مربع الخطأ
r2 = r2_score(y_test, y_pred_test)                          # التباين المفسَّر (الأقرب إلى 1 أفضل)

results = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae, rmse, r2],
    'Description': [
        'متوسط الخطأ المطلق (USD)',
        'جذر متوسط مربع الخطأ (USD)',
        'معامل التحديد (1 = مثالي)'
    ]
})

display(results.round(4))""",

    42: """## لماذا تعمل هذه المجموعة جيداً للانحدار الخطي البسيط؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **علاقة خطية قوية** | يزداد الراتب بثبات مع سنوات الخبرة |
| 2 | **ميزة واحدة وهدف واحد** | يطابق متطلبات SLR تماماً |
| 3 | **ضجيج منخفض** | تبقى التنبؤات قريبة من الخط الحقيقي → **R² عالٍ** |
| 4 | **تفسير واضح** | الميل = الراتب الإضافي لكل سنة خبرة |
| 5 | **قيم مفقودة مُضمَّنة** | يعرض تنظيف البيانات بـ `SimpleImputer` |

> **المقارنة:** مجموعة SAT/GPA كانت **R² ≈ 0.38** (متوسط). هذه المجموعة تحقق عادةً **R² > 0.90** (ملاءمة ممتازة).""",
}
