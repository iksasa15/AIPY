# Arabic translations for Multiple_linear_regression.ipynb
CELLS = {
    0: """# الانحدار الخطي المتعدد — Google Colab

**الهدف:** التنبؤ بقيمة مستهدفة متصلة من **عدة** ميزات باستخدام الانحدار الخطي المتعدد.

| المثال | الميزات (X) | المستهدف (y) | مجموعة البيانات |
|---------|--------------|------------|---------|
| **المثال 1** | R&D Spend, Administration, Marketing Spend, State | Profit | `../Datasets/50_Startups.csv` |
| **المثال 2** | Area_sqft, Bedrooms, Age_years | Price | `../Datasets/house_price.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| المرحلة 1 | معالجة البيانات | التحميل ← التنظيف ← الترميز ← التقسيم |
| المرحلة 2 | الخوارزمية | التدريب ← التنبؤ ← التصور ← التقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)
""",

    1: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

Google Colab يتضمن عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبيت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.
""",

    2: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn
""",

    3: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات لمعالجة البيانات والتحضير والنمذجة والتقييم.

**ما تفعله هذه الخلية:** تحميل وحدات numpy و pandas و matplotlib و seaborn و sklearn.
""",

    4: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء المخططات والرسوم
import seaborn as sns           # تصورات إحصائية (تنسيق اختياري)

from sklearn.model_selection import train_test_split  # تقسيم البيانات إلى تدريب/اختبار
from sklearn.impute import SimpleImputer              # ملء القيم المفقودة
from sklearn.preprocessing import OneHotEncoder         # ترميز الأعمدة الفئوية
from sklearn.compose import ColumnTransformer         # تطبيق التحويلات على الأعمدة
from sklearn.linear_model import LinearRegression     # نموذج الانحدار الخطي المتعدد
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # مقاييس التقييم

plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)  # حجم المخطط الافتراضي: عرض=10، ارتفاع=6 بوصة
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لتقسيمات قابلة للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح
""",

    5: """---
# المثال 1: أرباح الشركات الناشئة — الانحدار الخطي المتعدد

التنبؤ بـ **Profit** من ثلاثة أعمدة للإنفاق و**State** التي تعمل فيها الشركة الناشئة.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `R&D Spend` | ميزة (X₁) | ميزانية البحث والتطوير |
| `Administration` | ميزة (X₂) | المصاريف الإدارية |
| `Marketing Spend` | ميزة (X₃) | ميزانية التسويق |
| `State` | ميزة (X₄) | ولاية أمريكية (فئوية ← One-Hot) |
| `Profit` | مستهدف (y) | الربح السنوي بالدولار |

**الملف:** `../Datasets/50_Startups.csv`
""",

    6: """---
# المرحلة 1: معالجة البيانات

تحضير البيانات قبل تدريب النموذج — نفس القالب يُعاد استخدامه لخوارزميات أخرى.
""",

    7: """## المثال 1 — الخلية 1: تحميل واستكشاف البيانات

تحميل ملف CSV وإجراء استكشاف أولي (head, info, describe, shape).

**ما تفعله هذه الخلية:** قراءة `../Datasets/50_Startups.csv` وعرض إحصائيات أساسية.
""",

    8: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/50_Startups.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']  # المتغيرات المستقلة
TARGET_COL = 'Profit'  # المتغير التابع المراد التنبؤ به

print('أول 5 صفوف:')          # طباعة تسمية للجدول أدناه
display(dataset.head())         # عرض أول 5 صفوف لفحص البيانات

print('\\nمعلومات مجموعة البيانات:')       # طباعة تسمية لأنواع الأعمدة وعدد القيم غير الفارغة
dataset.info()                  # عرض أسماء الأعمدة وأنواع البيانات وعدد القيم غير الفارغة

print('\\nملخص إحصائي:')  # طباعة تسمية للإحصائيات العددية
display(dataset.describe())       # عرض العدد والمتوسط والانحراف المعياري والحد الأدنى والأقصى والربيعيات

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')  # إجمالي الصفوف والأعمدة
""",

    9: """## المثال 1 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

تنظيف البيانات يشمل فحص القيم المفقودة وإزالة التكرارات واستخدام SimpleImputer عند الحاجة.

**ما تفعله هذه الخلية:** فحص القيم الفارغة وحذف الصفوف المكررة وملء القيم العددية المفقودة.
""",

    10: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')  # طباعة تسمية
print(dataset.isnull().sum())        # عد قيم NaN في كل عمود

rows_before = len(dataset)                              # تخزين عدد الصفوف قبل التنظيف
dataset = dataset.drop_duplicates().reset_index(drop=True)  # إزالة التكرارات وإعادة تعيين فهرس الصفوف
rows_after = len(dataset)                               # تخزين عدد الصفوف بعد التنظيف
print(f'\\nالتكرارات المحذوفة: {rows_before - rows_after}')  # عرض عدد التكرارات المحذوفة

numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()  # قائمة الأعمدة العددية فقط
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # استبدال NaN بمتوسط العمود
if dataset.isnull().sum().sum() > 0:               # إذا وُجدت قيم مفقودة
    dataset[numeric_cols] = imputer.fit_transform(dataset[numeric_cols])  # ملء الأعمدة العددية فقط
    print('تم ملء القيم المفقودة بالمتوسط')      # تأكيد تطبيق الملء
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُملِئ')  # تخطي الملء عندما تكون البيانات كاملة

print(f'\\nالصفوف بعد التنظيف: {rows_after}')    # العدد النهائي للصفوف في مجموعة البيانات
""",

    11: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

خوارزميات التعلم الآلي تتطلب مدخلات عددية. عمود `State` فئوي وسيُرمَّز بـ One-Hot Encoding.

**ما تفعله هذه الخلية:** ترميز `State` بـ One-Hot وبناء مصفوفة الميزات X بجميع الأعمدة العددية.
""",

    12: """# الخطوة 3) الترميز الفئوي للانحدار الخطي المتعدد

cat_cols = [col for col in FEATURE_COLS if dataset[col].dtype == 'object']  # إيجاد أعمدة النص في الميزات
num_feature_cols = [col for col in FEATURE_COLS if col not in cat_cols]  # إيجاد أعمدة الميزات العددية

if cat_cols:  # تشغيل الترميز عند وجود أعمدة ميزات فئوية
    print(f'أعمدة الميزات الفئوية: {cat_cols}')  # قائمة الأعمدة التي تحتاج ترميزاً
    preprocessor = ColumnTransformer(  # بناء خط أنابيب للأعمدة المحددة
        transformers=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)],  # ترميز one-hot
        remainder='passthrough'  # الإبقاء على أعمدة الميزات العددية دون تغيير
    )
    X_array = preprocessor.fit_transform(dataset[FEATURE_COLS])  # تعلّم الفئات وتحويل الميزات
    feature_names = preprocessor.get_feature_names_out()  # الحصول على أسماء الأعمدة المرمّزة الجديدة
    X = pd.DataFrame(X_array, columns=feature_names)  # تحويل المخرجات المرمّزة إلى DataFrame
    print('تم تطبيق One-Hot Encoding (drop=first يتجنب فخ المتغير الوهمي)')  # تأكيد اكتمال الترميز
    print(f'أعمدة الميزات بعد الترميز: {list(X.columns)}')  # عرض أسماء الميزات النهائية
else:
    X = dataset[FEATURE_COLS].copy()  # استخدام الميزات العددية مباشرة عند عدم وجود فئات
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')  # لا شيء للترميز

y = dataset[TARGET_COL].values  # متجه المستهدف: قيم الربح المراد التنبؤ بها
""",

    13: """## المثال 1 — الخلية 4: تقسيم البيانات

تعريف مصفوفة الميزات X والمستهدف y، ثم التقسيم إلى 80% تدريب و 20% اختبار.

**ما تفعله هذه الخلية:** إنشاء مصفوفات X و y وتطبيق train_test_split مع random_state=42.
""",

    14: """# الخطوة 4) تقسيم التدريب-الاختبار

X_values = X.values  # تحويل DataFrame الميزات إلى مصفوفة numpy ثنائية الأبعاد لـ sklearn
y_values = y       # المستهدف مصفوفة numpy أحادية البعد بالفعل

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values,  # البيانات المراد تقسيمها
    test_size=0.2,       # 20% للاختبار، 80% للتدريب
    random_state=42      # بذرة ثابتة لتقسيم قابل للتكرار في كل تشغيل
)

print(f'شكل X_train: {X_train.shape}')  # شكل ميزات التدريب: (n_train, n_features)
print(f'شكل X_test:  {X_test.shape}')   # شكل ميزات الاختبار: (n_test, n_features)
print(f'شكل y_train: {y_train.shape}')  # شكل مستهدفات التدريب: (n_train,)
print(f'شكل y_test:  {y_test.shape}')   # شكل مستهدفات الاختبار: (n_test,)
""",

    15: """> **ملاحظة:** الانحدار الخطي المتعدد **لا يتطلب** تحجيم الميزات. المعاملات ما زالت قابلة للتفسير على المقياس الأصلي عندما تستخدم الميزات وحدات متقاربة.
""",

    16: """---
# المرحلة 2: الانحدار الخطي المتعدد

تطبيق الانحدار الخطي المتعدد: `Profit = b₀ + b₁·R&D + b₂·Admin + b₃·Marketing + b₄·State + ...`
""",

    17: """## المثال 1 — الخلية 5: تدريب النموذج

تدريب `LinearRegression` على بيانات التدريب. يتعلّم النموذج معاملاً واحداً لكل ميزة بالإضافة إلى الثابت.

**ما تفعله هذه الخلية:** ملاءمة `LinearRegression` وطباعة الثابت وجميع معاملات الميزات.
""",

    18: """# الخطوة 5) تدريب النموذج
regressor = LinearRegression()      # إنشاء نموذج انحدار خطي متعدد فارغ
regressor.fit(X_train, y_train)     # تدريب النموذج: تعلّم الثابت وجميع المعاملات

print(f'الثابت (b0): {regressor.intercept_:.2f}')  # الحد الثابت عندما تكون جميع الميزات صفراً
print('\\nالمعاملات:')  # طباعة تسمية لجدول المعاملات
for name, coef in zip(X.columns, regressor.coef_):  # التكرار على كل ميزة ووزنها
    print(f'  {name}: {coef:.6f}')  # عرض تغيّر الربح لكل زيادة بوحدة واحدة في تلك الميزة

equation_terms = ' + '.join([f'{c:.4f} x {n}' for n, c in zip(X.columns, regressor.coef_)])  # بناء سلسلة المعادلة
print(f'\\nالمعادلة: {TARGET_COL} = {regressor.intercept_:.2f} + {equation_terms}')  # معادلة الانحدار الكاملة
""",

    19: """## المثال 1 — الخلية 6: التنبؤ

بعد التدريب، يتنبأ النموذج بقيم Profit لمجموعتي التدريب والاختبار.

**ما تفعله هذه الخلية:** إنشاء y_pred_train و y_pred_test باستخدام النموذج المُدرَّب.
""",

    20: """# الخطوة 6) التنبؤ
y_pred_train = regressor.predict(X_train)  # التنبؤ بالربح لعينات التدريب
y_pred_test = regressor.predict(X_test)    # التنبؤ بالربح لعينات الاختبار غير المرئية

print('عينات تنبؤ (مجموعة الاختبار):')  # طباعة تسمية
for i in range(min(5, len(y_test))):     # التكرار على أول 5 عينات اختبار
    print(f'  الربح الفعلي={y_test[i]:,.2f}، المتوقع={y_pred_test[i]:,.2f}')  # مقارنة الفعلي مقابل المتوقع
""",

    21: """## المثال 1 — الخلية 7: التصور

مخطط مبعثر **الفعلي مقابل المتوقع** للربح — النقاط القريبة من الخط القطري تشير إلى تنبؤات جيدة.

**ما تفعله هذه الخلية:** رسم الربح الحقيقي (محور x) مقابل تنبؤات النموذج (محور y).
""",

    22: """# الخطوة 7) التصور — الفعلي مقابل المتوقع
plt.figure(figsize=(10, 6))  # إنشاء شكل جديد بحجم مخصص
plt.scatter(y_test, y_pred_test, color='steelblue', alpha=0.8, edgecolors='white', label='تنبؤات الاختبار')  # رسم النقاط

min_val = min(y_test.min(), y_pred_test.min())  # إيجاد القيمة الدنيا لنطاق الخط القطري
max_val = max(y_test.max(), y_pred_test.max())  # إيجاد القيمة العليا لنطاق الخط القطري
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='خط التنبؤ المثالي')  # خط مثالي بزاوية 45°

plt.xlabel('الربح الفعلي (USD)')       # تسمية المحور x
plt.ylabel('الربح المتوقع (USD)')    # تسمية المحور y
plt.title('الانحدار الخطي المتعدد — الربح الفعلي مقابل المتوقع (الشركات الناشئة)')  # عنوان المخطط
plt.legend()             # عرض وسيلة الإيضاح
plt.tight_layout()       # ضبط المسافات حتى لا تُقص التسميات
plt.show()               # عرض المخطط
""",

    23: """## المثال 1 — الخلية 8: التقييم

تقييم النموذج على مجموعة الاختبار باستخدام MAE و RMSE و R² (معامل التحديد).

**ما تفعله هذه الخلية:** حساب وعرض مقاييس تقييم الانحدار.
""",

    24: """# الخطوة 8) التقييم
mae = mean_absolute_error(y_test, y_pred_test)              # متوسط خطأ التنبؤ المطلق
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))     # الجذر التربيعي لمتوسط مربع الخطأ
r2 = r2_score(y_test, y_pred_test)                          # نسبة التباين التي يفسرها النموذج

results = pd.DataFrame({  # بناء جدول مقاييس التقييم
    'Metric': ['MAE', 'RMSE', 'R²'],  # أسماء المقاييس
    'Value': [mae, rmse, r2],         # قيم المقاييس المحسوبة
    'Description': [                   # شرح مختصر لكل مقياس
        'متوسط الخطأ المطلق (USD)',
        'جذر متوسط مربع الخطأ (USD)',
        'معامل التحديد (1 = مثالي)'
    ]
})

display(results.round(4))  # عرض المقاييس مقربة إلى 4 خانات عشرية
print(f'\\nR² اختبار المثال 1 = {r2:.4f}')  # طباعة R² للمرجعية السريعة
""",

    25: """## لماذا R² ≈ 0.8987؟

مجموعة بيانات 50 Startups مثال **كلاسيكي** للانحدار الخطي المتعدد. صُمِّم الربح ليعتمد خطياً على R&D و Administration و Marketing و State.

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **ميزات متعددة ذات صلة** | يُفسَّر الربح بالإنفاق في ثلاثة أقسام بالإضافة إلى الموقع — وليس متغيراً واحداً |
| 2 | **علاقة خطية** | العلاقة الحقيقية بين الإنفاق والربح تقريباً خطية |
| 3 | **ترميز State الفئوي** | One-Hot Encoding يتيح للنموذج التقاط الفروق الإقليمية في الربحية |
| 4 | **ضوضاء منخفضة** | تتجمع نقاط البيانات قرب مستوى الانحدار → **R² مرتفع** |
| 5 | **معاملات قابلة للتفسير** | كل معامل يوضح كيف يتغيّر الربح عند زيادة ميزة واحدة (مع ثبات الباقي) |

> **الخلاصة:** R² ≈ 0.8987 = **ملاءمة ممتازة** — يفسّر النموذج نحو **90%** من تباين الربح لأن المشكلة تتوافق مع افتراضات الانحدار الخطي المتعدد.
""",

    26: """---
# المثال 2: سعر المنزل — الانحدار الخطي المتعدد

التنبؤ بـ **Price** من ثلاث ميزات عددية للعقار.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Area_sqft` | ميزة (X₁) | مساحة المعيشة بالقدم المربع |
| `Bedrooms` | ميزة (X₂) | عدد غرف النوم |
| `Age_years` | ميزة (X₃) | عمر المنزل بالسنوات |
| `Price` | مستهدف (y) | سعر البيع بالدولار |

**الملف:** `../Datasets/house_price.csv`
""",

    27: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV أسعار المنازل وفحص البيانات.

**ما تفعله هذه الخلية:** قراءة `../Datasets/house_price.csv` وعرض إحصائيات أساسية.
""",

    28: """# تحميل مجموعة بيانات أسعار المنازل
dataset = pd.read_csv('../Datasets/house_price.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['Area_sqft', 'Bedrooms', 'Age_years']  # المتغيرات المستقلة
TARGET_COL = 'Price'  # المتغير التابع المراد التنبؤ به

print('أول 5 صفوف:')          # طباعة تسمية للجدول أدناه
display(dataset.head())         # عرض أول 5 صفوف لفحص البيانات

print('\\nمعلومات مجموعة البيانات:')       # طباعة تسمية لأنواع الأعمدة وعدد القيم غير الفارغة
dataset.info()                  # عرض أسماء الأعمدة وأنواع البيانات وعدد القيم غير الفارغة

print('\\nملخص إحصائي:')  # طباعة تسمية للإحصائيات العددية
display(dataset.describe())       # عرض العدد والمتوسط والانحراف المعياري والحد الأدنى والأقصى والربيعيات

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')  # إجمالي الصفوف والأعمدة
""",

    29: """## المثال 2 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

فحص القيم المفقودة وإزالة التكرارات والملء عند الحاجة.

**ما تفعله هذه الخلية:** تنظيف مجموعة بيانات أسعار المنازل قبل النمذجة.
""",

    30: """# تنظيف البيانات لمجموعة أسعار المنازل

print('القيم المفقودة لكل عمود:')  # طباعة تسمية
print(dataset.isnull().sum())        # عد قيم NaN في كل عمود

rows_before = len(dataset)                              # تخزين عدد الصفوف قبل التنظيف
dataset = dataset.drop_duplicates().reset_index(drop=True)  # إزالة الصفوف المكررة
rows_after = len(dataset)                               # تخزين عدد الصفوف بعد إزالة التكرار
print(f'\\nالتكرارات المحذوفة: {rows_before - rows_after}')  # عرض عدد التكرارات

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # ملء NaN بمتوسط العمود
if dataset.isnull().sum().sum() > 0:               # إذا وُجدت قيم مفقودة
    dataset[FEATURE_COLS + [TARGET_COL]] = imputer.fit_transform(dataset[FEATURE_COLS + [TARGET_COL]])  # ملء
    print('تم ملء القيم المفقودة بالمتوسط')      # تأكيد الملء
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق المُملِئ')  # تخطي عند اكتمال البيانات

print(f'\\nالصفوف بعد التنظيف: {rows_after}')    # العدد النهائي للصفوف
""",

    31: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

جميع أعمدة الميزات عددية — يُتخطَّى الترميز.

**ما تفعله هذه الخلية:** اكتشاف الأعمدة الفئوية وتطبيق الترميز عند الحاجة.
""",

    32: """# الترميز الفئوي

cat_cols = [col for col in FEATURE_COLS if dataset[col].dtype == 'object']  # إيجاد أعمدة النص في الميزات

if cat_cols:  # الترميز فقط عند وجود أعمدة فئوية
    print(f'أعمدة فئوية موجودة: {cat_cols}')  # قائمة الأعمدة التي تحتاج ترميزاً
    preprocessor = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )
    X_array = preprocessor.fit_transform(dataset[FEATURE_COLS])
    feature_names = preprocessor.get_feature_names_out()
    X = pd.DataFrame(X_array, columns=feature_names)
    print('تم تطبيق One-Hot Encoding')
else:
    X = dataset[FEATURE_COLS].copy()  # جميع الميزات عددية بالفعل
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')  # لا شيء للترميز
    print(f'أعمدة الميزات العددية: {list(X.columns)}')  # عرض أسماء الميزات

y = dataset[TARGET_COL].values  # متجه المستهدف: أسعار المنازل
""",

    33: """## المثال 2 — الخلية 4: تقسيم البيانات

تعريف X (Area, Bedrooms, Age) و y (Price)، ثم التقسيم 80/20.

**ما تفعله هذه الخلية:** إنشاء مصفوفات الميزات/المستهدف وتطبيق train_test_split.
""",

    34: """# تقسيم التدريب-الاختبار

X_values = X.values  # مصفوفة الميزات كمصفوفة numpy
y_values = y         # متجه المستهدف

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values,  # البيانات المراد تقسيمها
    test_size=0.2,       # 20% اختبار، 80% تدريب
    random_state=42      # تقسيم قابل للتكرار
)

print(f'شكل X_train: {X_train.shape}')  # شكل ميزات التدريب
print(f'شكل X_test:  {X_test.shape}')   # شكل ميزات الاختبار
print(f'شكل y_train: {y_train.shape}')  # شكل مستهدفات التدريب
print(f'شكل y_test:  {y_test.shape}')   # شكل مستهدفات الاختبار
""",

    35: """## المثال 2 — الخلية 5: تدريب النموذج

تدريب الانحدار الخطي المتعدد: `Price = b₀ + b₁·Area + b₂·Bedrooms + b₃·Age`

**ما تفعله هذه الخلية:** ملاءمة النموذج وطباعة معادلة الانحدار.
""",

    36: """# تدريب النموذج
regressor = LinearRegression()      # إنشاء نموذج انحدار خطي متعدد
regressor.fit(X_train, y_train)     # تعلّم الثابت وجميع المعاملات

print(f'الثابت (b0): {regressor.intercept_:.2f}')  # السعر الأساسي عندما تكون جميع الميزات صفراً
print('\\nالمعاملات:')  # طباعة تسمية
for name, coef in zip(X.columns, regressor.coef_):  # التكرار على معامل كل ميزة
    print(f'  {name}: {coef:.2f}')  # عرض تغيّر السعر لكل زيادة بوحدة واحدة في الميزة

equation_terms = ' + '.join([f'{c:.2f} x {n}' for n, c in zip(X.columns, regressor.coef_)])  # حدود المعادلة
print(f'\\nالمعادلة: {TARGET_COL} = {regressor.intercept_:.2f} + {equation_terms}')  # المعادلة الكاملة
""",

    37: """## المثال 2 — الخلية 6: التنبؤ

التنبؤ بأسعار المنازل لعينات التدريب والاختبار.

**ما تفعله هذه الخلية:** إنشاء التنبؤات وعرض مقارنات عينة.
""",

    38: """# التنبؤ
y_pred_train = regressor.predict(X_train)  # التنبؤ بالسعر لبيانات التدريب
y_pred_test = regressor.predict(X_test)    # التنبؤ بالسعر لبيانات الاختبار

print('عينات تنبؤ (مجموعة الاختبار):')  # طباعة تسمية
for i in range(min(5, len(y_test))):     # عرض أول 5 تنبؤات اختبار
    print(f'  السعر الفعلي=${y_test[i]:,.0f}، المتوقع=${y_pred_test[i]:,.0f}')  # مقارنة الفعلي مقابل المتوقع
""",

    39: """## المثال 2 — الخلية 7: التصور

مخطط مبعثر للأسعار الفعلية مقابل المتوقعة على مجموعة الاختبار.

**ما تفعله هذه الخلية:** رسم الأسعار الحقيقية مقابل تنبؤات النموذج.
""",

    40: """# التصور — الفعلي مقابل المتوقع
plt.figure(figsize=(10, 6))  # إنشاء الشكل
plt.scatter(y_test, y_pred_test, color='darkorange', alpha=0.8, edgecolors='white', label='تنبؤات الاختبار')  # مبعثر

min_val = min(y_test.min(), y_pred_test.min())  # الحد الأدنى للخط القطري
max_val = max(y_test.max(), y_pred_test.max())  # الحد الأقصى للخط القطري
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='خط التنبؤ المثالي')  # خط مثالي

plt.xlabel('السعر الفعلي (USD)')        # تسمية المحور x
plt.ylabel('السعر المتوقع (USD)')     # تسمية المحور y
plt.title('الانحدار الخطي المتعدد — السعر الفعلي مقابل المتوقع للمنزل')  # عنوان المخطط
plt.legend()             # عرض وسيلة الإيضاح
plt.tight_layout()       # ضبط التخطيط
plt.show()               # عرض المخطط
""",

    41: """## المثال 2 — الخلية 8: التقييم

تقييم أداء النموذج بـ MAE و RMSE و R².

**ما تفعله هذه الخلية:** حساب وعرض مقاييس التقييم على مجموعة الاختبار.
""",

    42: """# التقييم
mae = mean_absolute_error(y_test, y_pred_test)              # متوسط الخطأ المطلق بالدولار
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))     # جذر متوسط مربع الخطأ
r2 = r2_score(y_test, y_pred_test)                          # التباين المفسَّر (الأقرب إلى 1 أفضل)

results = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Value': [mae, rmse, r2],
    'Description': [
        'متوسط الخطأ المطلق (USD)',
        'جذر متوسط مربع الخطأ (USD)',
        'معامل التحديد (1 = مثالي)'
    ]
})

display(results.round(4))
print(f'\\nR² اختبار المثال 2 = {r2:.4f}')  # طباعة R² للمرجعية السريعة
""",

    43: """## لماذا R² ≈ 0.9202؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **المساحة تدفع السعر** | المنازل الأكبر (قدم مربع) تكلف عادةً أكثر — معامل موجب قوي |
| 2 | **غرف النوم تضيف قيمة** | المزيد من غرف النوم يزيد عادةً سعر البيع |
| 3 | **العمر يخفض السعر** | المنازل الأقدم تُباع غالباً بأقل (يُتوقَّع معامل Age سالب) |
| 4 | **ميزات متعددة معاً** | MLR يجمع الإشارات الثلاث — أفضل من أي ميزة منفردة |
| 5 | **بيانات خطية تركيبية** | هذه المجموعة مُولَّدة بصيغة سعر شبه خطية مع ضوضاء صغيرة |

> **الخلاصة:** R² ≈ 0.9202 = **ملاءمة ممتازة** — يفسّر النموذج نحو **92%** من تباين السعر لأن المساحة وغرف النوم والعمر معاً تلتقط العلاقة الخطية.
""",
}
