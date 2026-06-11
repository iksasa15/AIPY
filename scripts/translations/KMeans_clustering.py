CELLS = {
    0: """# تجميع K-Means — Google Colab

**الهدف:** تجميع العملاء المتشابهين في **مجموعات (clusters)** باستخدام **K-Means** — خوارزمية **غير مُشرَفة** (لا حاجة لعمود هدف).

| مثال | الميزة/الميزات (X) | مجموعة البيانات |
|---------|----------------|---------|
| **المثال 1** | Age, Annual_Income | `../Datasets/mall_customers.csv` |
| **المثال 2** | Income, Spending_Score, Age, Num_Purchases | `../Datasets/customer_segments.csv` |

| المرحلة | الموضوع | الخلايا |
|-------|-------|-------|
| المرحلة 0 | الإعداد | التثبيت والاستيراد |
| — | دليل الخوارزمية | النقاط المركزية، القصور الذاتي، طريقة الكوع، معامل Silhouette |
| المرحلة 1 | معالجة البيانات | التحميل ← التنظيف ← الترميز ← تحضير X |
| المرحلة 2 | الخوارزمية | اختيار K ← التدريب ← التنبؤ ← التصور ← التقييم |

> **التشغيل:** Runtime → Run all (أو Ctrl+F9)

> **ملاحظة:** التجميع **غير مُشرَف** — لا يوجد عمود **y (هدف)**.""",

    1: """---
# دليل الخوارزمية — تجميع K-Means

## ما هو التجميع (Clustering)؟

**التجميع** يجمع نقاط البيانات المتشابهة معًا **بدون تسميات**. تكتشف الخوارزمية أنماطًا مخفية في البيانات.

| نوع التعلم | هل توجد تسميات؟ | مثال |
|---------------|-------------|---------|
| **مُشرَف** | نعم (y) | التصنيف، الانحدار |
| **غير مُشرَف** | **لا** | **تجميع K-Means** |

## ما هو K-Means؟

**K-Means** يقسم البيانات إلى **K مجموعة** بتقليل المسافة من كل نقطة إلى **النقطة المركزية (centroid)** لمجموعتها (المركز).

| المصطلح | المعنى |
|------|---------|
| **K** | عدد المجموعات (أنت تختاره) |
| **Centroid** | متوسط جميع النقاط في المجموعة |
| **Assignment** | كل نقطة تنتمي إلى أقرب نقطة مركزية |
| **Iteration** | إعادة حساب النقاط المركزية ← إعادة التعيين ← التكرار حتى الاستقرار |

## خوارزمية K-Means (خطوة بخطوة)

| الخطوة | الإجراء |
|------|--------|
| 1 | اختر **K** (عدد المجموعات) |
| 2 | تهيئة **K نقطة مركزية** عشوائيًا |
| 3 | عيّن كل نقطة إلى **أقرب** نقطة مركزية |
| 4 | حدّث كل نقطة مركزية = **متوسط** النقاط المعيّنة |
| 5 | كرّر الخطوتين 3–4 حتى **تتوقف** النقاط المركزية عن الحركة |

## دالة الهدف — القصور الذاتي (Inertia)

**Inertia** = مجموع المسافات التربيعية من كل نقطة إلى النقطة المركزية لمجموعتها.

`Inertia = Σ ||xᵢ − μ_c||²`

| Inertia | المعنى |
|---------|---------|
| **أقل** | مجموعات أكثر تماسكًا وإحكامًا |
| **K = n** | Inertia = 0 (كل نقطة مجموعة بمفردها — عديمة الفائدة) |

## اختيار K

| الطريقة | كيف تعمل |
|--------|--------------|
| **Elbow Method** | ارسم K مقابل inertia — ابحث عن **الكوع** (عائد متناقص) |
| **Silhouette Score** | يقيس مدى ملاءمة كل نقطة لمجموعتها (−1 إلى **1**، الأعلى أفضل) |
| **Domain knowledge** | القرار من منظور الأعمال (مثلاً 5 شرائح عملاء) |

## مقياس الميزات — مطلوب!

K-Means يستخدم **المسافة الإقليدية**. الميزات ذات المقياس الأكبر تسيطر → طبّق دائمًا **`StandardScaler`** قبل K-Means.

## المعاملات الفائقة الرئيسية (sklearn)

| المعامل | الدور |
|-----------|------|
| `n_clusters` | **K** — عدد المجموعات |
| `init` | تهيئة النقاط المركزية (`'k-means++'` افتراضي — بداية ذكية) |
| `n_init` | عدد مرات التشغيل ببذور مختلفة (اختر أفضل inertia) |
| `max_iter` | الحد الأقصى للتكرارات في كل تشغيل |
| `random_state` | نتائج قابلة للتكرار |

## K-Means مقابل التصنيف

| | K-Means | K-NN Classification |
|---|---------|---------------------|
| النوع | **غير مُشرَف** | مُشرَف |
| التسميات | **غير مطلوبة** | مطلوبة |
| المخرجات | معرّف المجموعة (0, 1, …, K−1) | تسمية الفئة |
| المقياس | **مطلوب** | مطلوب |

## ما يجب أن يتذكره الطالب

1. K-Means = **غير مُشرَف** — لا عمود هدف.
2. **قيّم الميزات دائمًا** قبل التجميع.
3. اختر **K** باستخدام **طريقة الكوع** و**معامل Silhouette**.
4. المخرجات = **تسمية المجموعة** لكل صف (`predict` أو `fit_predict`).
5. النقاط المركزية = **متوسط** النقاط في كل مجموعة.""",

    2: """## المرحلة 0 — الخلية 0: تثبيت المكتبات

يتضمن Google Colab عادةً معظم المكتبات. تضمن هذه الخلية توفر الحزم المطلوبة.

**ما تفعله هذه الخلية:** تثبت scikit-learn و pandas و matplotlib و numpy و seaborn بصمت.""",

    3: """# تثبيت المكتبات المطلوبة بصمت (-q يخفي المخرجات)
!pip install -q scikit-learn pandas matplotlib numpy seaborn""",

    4: """## المرحلة 0 — الخلية 1: استيراد المكتبات

استيراد المكتبات لمعالجة البيانات وتجميع K-Means والتقييم.

**ما تفعله هذه الخلية:** تحمّل numpy و pandas و matplotlib و sklearn KMeans و StandardScaler والمقاييس.""",

    5: """# --- استيراد المكتبات ---
import numpy as np              # العمليات العددية والمصفوفات
import pandas as pd             # تحميل ومعالجة البيانات الجدولية
import matplotlib.pyplot as plt # إنشاء المخططات والرسوم
import seaborn as sns           # تصورات إحصائية

from sklearn.impute import SimpleImputer                   # ملء القيم المفقودة
from sklearn.preprocessing import StandardScaler           # مقياس الميزات (مطلوب لـ K-Means)
from sklearn.cluster import KMeans                         # خوارزمية تجميع K-Means
from sklearn.metrics import silhouette_score, silhouette_samples  # تقييم التجميع

plt.rcParams['figure.figsize'] = (10, 6)  # حجم المخطط الافتراضي: عرض=10، ارتفاع=6 بوصة
plt.rcParams['font.family'] = ['Segoe UI', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid')            # خلفية بيضاء نظيفة مع خطوط شبكة
np.random.seed(42)                          # تثبيت البذرة العشوائية لنتائج قابلة للتكرار

print('المكتبات جاهزة')                      # تأكيد تحميل جميع الاستيرادات بنجاح""",

    6: """---
# المثال 1: عملاء المول — تجميع K-Means

تقسيم عملاء المول حسب **Age** و **Annual Income** إلى **K مجموعة** للتسويق المستهدف.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Age` | ميزة (X₁) | عمر العميل بالسنوات |
| `Annual_Income` | ميزة (X₂) | الدخل السنوي بالدولار الأمريكي |
| `Spending_Score` | معلومات فقط (لا يُستخدم في التجميع) | درجة الإنفاق 1–100 — للمرجع |

**الملف:** `../Datasets/mall_customers.csv`

**التجميع يستخدم:** Age + Annual_Income فقط (ميزتان لسهولة التصور).""",

    7: """---
# المرحلة 1: معالجة البيانات

تحضير الميزات للتجميع — **لا عمود هدف** في التعلم غير المُشرَف.""",

    8: """## المثال 1 — الخلية 1: تحميل واستكشاف البيانات

تحميل ملف CSV وإجراء استكشاف أولي (head، info، describe، shape).

**ما تفعله هذه الخلية:** تقرأ `../Datasets/mall_customers.csv` وتعرض إحصائيات أساسية.""",

    9: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/mall_customers.csv')  # قراءة CSV إلى DataFrame

FEATURE_COLS = ['Age', 'Annual_Income']  # الميزات المستخدمة في التجميع
INFO_COL = 'Spending_Score'              # عمود إضافي — لا يُستخدم في K-Means (للمرجع فقط)

print('أول 5 صفوف:')
display(dataset.head())

print('\\nمعلومات مجموعة البيانات:')
dataset.info()

print('\\nملخص إحصائي:')
display(dataset.describe())

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')
print('تعلم غير مُشرَف — لا عمود هدف (y).')""",

    10: """## المثال 1 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

التحقق من القيم المفقودة وإزالة التكرارات وتطبيق الإكمال إذا لزم الأمر.

**ما تفعله هذه الخلية:** تنظّف مجموعة البيانات قبل التجميع.""",

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
    print('تم إكمال القيم المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق الإكمال')

print(f'\\nالصفوف بعد التنظيف: {rows_after}')""",

    12: """## المثال 1 — الخلية 3: ترميز البيانات الفئوية

جميع ميزات التجميع رقمية — يُتخطى الترميز.

**ما تفعله هذه الخلية:** تؤكد أنه لا حاجة لترميز فئوي.""",

    13: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات للتجميع: {FEATURE_COLS}')""",

    14: """## المثال 1 — الخلية 4: تحضير الميزات والمقياس

تعريف **X** (الميزات فقط) وتطبيق **StandardScaler** — مطلوب لـ K-Means.

**ما تفعله هذه الخلية:** ينشئ مصفوفة الميزات ويقيّمها للتجميع القائم على المسافة.""",

    15: """# الخطوة 4) تحضير الميزات وقياسها (بدون تقسيم تدريب/اختبار — التجميع يستخدم كل البيانات)

X = dataset[FEATURE_COLS].values  # مصفوفة الميزات فقط — بدون y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # قياس إلى متوسط=0، انحراف=1

print(f'شكل X: {X.shape}')
print(f'شكل X_scaled: {X_scaled.shape}')
print('تم قياس الميزات بـ StandardScaler — جاهز لـ K-Means.')""",

    16: """> **ملاحظة:** K-Means يستخدم **المسافة الإقليدية** — **StandardScaler إلزامي** عندما يكون Age و Income بوحدات مختلفة.""",

    17: """---
# المرحلة 2: K-Means — عملاء المول

اختر **K** بطريقة الكوع، ثم درّب K-Means وصوّر المجموعات.""",

    18: """## المثال 1 — الخلية 5: اختيار K (طريقة الكوع + Silhouette)

جرّب K من 1 إلى 10 — ارسم **inertia** (الكوع) و**معامل Silhouette** لاختيار أفضل K.

**ما تفعله هذه الخلية:** يجد K الأمثل قبل تدريب النموذج النهائي.""",

    19: """# الخطوة 5) اختيار K — طريقة الكوع ومعامل Silhouette

K_range = range(2, 11)  # يجب أن يكون K على الأقل 2 لمجموعات ذات معنى
inertias = []
silhouettes = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

best_k = list(K_range)[np.argmax(silhouettes)]  # K بأعلى معامل Silhouette
print(f'أفضل K (أعلى Silhouette): {best_k}')
print(f'Silhouette عند K={best_k}: {max(silhouettes):.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_range), inertias, marker='o', color='steelblue')
axes[0].axvline(best_k, color='red', linestyle='--', label=f'أفضل K = {best_k}')
axes[0].set_xlabel('K (عدد المجموعات)')
axes[0].set_ylabel('Inertia (مجموع مربعات المسافات داخل المجموعة)')
axes[0].set_title('طريقة الكوع — عملاء المول')
axes[0].legend()

axes[1].plot(list(K_range), silhouettes, marker='o', color='seagreen')
axes[1].axvline(best_k, color='red', linestyle='--', label=f'أفضل K = {best_k}')
axes[1].set_xlabel('K (عدد المجموعات)')
axes[1].set_ylabel('معامل Silhouette')
axes[1].set_title('معامل Silhouette مقابل K')
axes[1].legend()

plt.tight_layout()
plt.show()""",

    20: """## المثال 1 — الخلية 6: تدريب K-Means وتعيين المجموعات

درّب K-Means بالـ **K** المختار وعيّن لكل عميل تسمية مجموعة.

**ما تفعله هذه الخلية:** يدرّب النموذج ويضيف تسميات المجموعات إلى مجموعة البيانات.""",

    21: """# الخطوة 6) تدريب K-Means وتعيين تسميات المجموعات

kmeans = KMeans(
    n_clusters=best_k,     # عدد المجموعات من الخطوة 5
    init='k-means++',      # تهيئة ذكية للنقاط المركزية
    n_init=10,             # تشغيل 10 مرات — الاحتفاظ بأفضل inertia
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)  # التدريب وتعيين معرّف المجموعة (0 إلى K-1)

dataset['Cluster'] = cluster_labels  # إضافة عمود المجموعة إلى DataFrame

print(f'تم تدريب K-Means بـ K = {best_k}')
print(f'Inertia: {kmeans.inertia_:.2f}')
print(f'\\nالعملاء لكل مجموعة:')
print(dataset['Cluster'].value_counts().sort_index())

print('\\nمراكز المجموعات (فضاء مقيّس):')
print(kmeans.cluster_centers_.round(3))""",

    22: """## المثال 1 — الخلية 7: التصور

ارسم العملاء ملوّنين حسب **المجموعة** مع تحديد **النقاط المركزية**.

**ما تفعله هذه الخلية:** يصوّر شرائح K-Means في مستوى Age–Income.""",

    23: """# الخطوة 7) التصور — مخطط مبعثر مع النقاط المركزية

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)  # العودة للمقياس الأصلي

plt.figure(figsize=(10, 6))
palette = sns.color_palette('tab10', best_k)

for c in range(best_k):
    mask = cluster_labels == c
    plt.scatter(
        X[mask, 0], X[mask, 1],
        c=[palette[c]], label=f'المجموعة {c}', alpha=0.7, s=60
    )

plt.scatter(
    centroids_original[:, 0], centroids_original[:, 1],
    c='black', marker='X', s=200, linewidths=2, label='النقاط المركزية', zorder=5
)

plt.xlabel('Age')
plt.ylabel('Annual Income (USD)')
plt.title(f'تجميع K-Means (K={best_k}) — عملاء المول')
plt.legend()
plt.tight_layout()
plt.show()""",

    24: """## المثال 1 — الخلية 8: التقييم

قيّم جودة التجميع بـ **Silhouette Score** و **Inertia**.

**ما تفعله هذه الخلية:** يحسب مقاييس التجميع غير المُشرَف (وليس Accuracy — لا تسميات).""",

    25: """# الخطوة 8) التقييم — مقاييس التجميع (غير مُشرَف)

sil_avg = silhouette_score(X_scaled, cluster_labels)  # Silhouette الإجمالي (−1 إلى 1)
sil_samples = silhouette_samples(X_scaled, cluster_labels)  # Silhouette لكل نقطة

results = pd.DataFrame({
    'Metric': ['Inertia', 'Silhouette Score', 'Number of Clusters'],
    'Value': [kmeans.inertia_, sil_avg, best_k],
    'Description': [
        'مجموع مربعات المسافات داخل المجموعة (أقل = أكثر إحكامًا)',
        'جودة المجموعات: 1 = مثالي، 0 = متداخلة، <0 = مجموعة خاطئة',
        'K المختار بمعامل Silhouette'
    ]
})

display(results.round(4))

print(f'\\nمعامل Silhouette للمثال 1 = {sil_avg:.4f}')
print('(لا Accuracy — التجميع غير مُشرَف، لا تسميات حقيقية للمقارنة)')""",

    26: """## لماذا يعمل K-Means لعملاء المول؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **شرائح طبيعية** | Age و Income تشكل مجموعات واضحة في فضاء ثنائي الأبعاد |
| 2 | **مجموعات كروية** | K-Means يفترض مجموعات دائرية — يعمل جيدًا هنا |
| 3 | **المقياس** | Income (~50k) مقابل Age (~40) — StandardScaler يجعل المسافة عادلة |
| 4 | **استخدام تسويقي** | كل مجموعة = شريحة عملاء لحملات مستهدفة |
| 5 | **الكوع + Silhouette** | اختيار K مبني على البيانات بدل التخمين |

> **الخلاصة:** K-Means يكتشف **شرائح العملاء** بدون الحاجة لتسمية "Purchased".""",

    27: """## فهم مقاييس التجميع — المثال 1

| المقياس | المعنى |
|--------|---------|
| **Inertia** | مدى إحكام المجموعات — قارن عبر قيم K (الكوع) |
| **Silhouette** | مدى فصل المجموعات — **> 0.5** = معقول |
| **لا Accuracy** | غير مُشرَف — لا y للمقارنة مع التنبؤات |

> **نصيحة:** لوّن النقاط حسب `Spending_Score` بشكل منفصل لترى إن كانت المجموعات تطابق سلوك الإنفاق.""",

    28: """---
# المثال 2: شرائح العملاء — K-Means (متعدد الميزات)

تقسيم العملاء باستخدام **أربع ميزات** — الدخل، الإنفاق، العمر، وتكرار الشراء.

| العمود | الدور | الوصف |
|--------|------|-------------|
| `Income` | ميزة (X₁) | الدخل السنوي بالدولار الأمريكي |
| `Spending_Score` | ميزة (X₂) | درجة الإنفاق 1–100 |
| `Age` | ميزة (X₃) | عمر العميل |
| `Num_Purchases` | ميزة (X₄) | عدد المشتريات سنويًا |

**الملف:** `../Datasets/customer_segments.csv`""",

    29: """## المثال 2 — الخلية 1: تحميل واستكشاف البيانات

تحميل CSV شرائح العملاء وفحص البيانات.

**ما تفعله هذه الخلية:** تقرأ `../Datasets/customer_segments.csv` وتعرض إحصائيات أساسية.""",

    30: """# الخطوة 1) تحميل مجموعة البيانات
dataset = pd.read_csv('../Datasets/customer_segments.csv')

FEATURE_COLS = ['Income', 'Spending_Score', 'Age', 'Num_Purchases']

print('أول 5 صفوف:')
display(dataset.head())

print('\\nمعلومات مجموعة البيانات:')
dataset.info()

print('\\nملخص إحصائي:')
display(dataset.describe())

print(f'\\nالشكل: {dataset.shape[0]} صف × {dataset.shape[1]} عمود')""",

    31: """## المثال 2 — الخلية 2: تنظيف البيانات (معالجة القيم المفقودة)

التحقق من القيم المفقودة وإزالة التكرارات.

**ما تفعله هذه الخلية:** تنظّف مجموعة البيانات قبل التجميع.""",

    32: """# الخطوة 2) تنظيف البيانات

print('القيم المفقودة لكل عمود:')
print(dataset.isnull().sum())

rows_before = len(dataset)
dataset = dataset.drop_duplicates().reset_index(drop=True)
print(f'\\nالتكرارات المحذوفة: {rows_before - len(dataset)}')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
if dataset.isnull().sum().sum() > 0:
    dataset[FEATURE_COLS] = imputer.fit_transform(dataset[FEATURE_COLS])
    print('تم إكمال القيم المفقودة بالمتوسط')
else:
    print('لا توجد قيم مفقودة — لم يُطبَّق الإكمال')

print(f'\\nالصفوف بعد التنظيف: {len(dataset)}')""",

    33: """## المثال 2 — الخلية 3: ترميز البيانات الفئوية

جميع الأعمدة رقمية — يُتخطى الترميز.

**ما تفعله هذه الخلية:** تؤكد أنه لا حاجة لترميز فئوي.""",

    34: """# الخطوة 3) الترميز الفئوي

cat_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f'أعمدة فئوية موجودة: {cat_cols}')
else:
    print('لا توجد أعمدة فئوية — تم تخطي الترميز.')
    print(f'أعمدة الميزات: {FEATURE_COLS}')""",

    35: """## المثال 2 — الخلية 4: تحضير الميزات والمقياس

تعريف **X** وقياس جميع الميزات الأربع بـ StandardScaler.

**ما تفعله هذه الخلية:** يحضّر مصفوفة ميزات مقيّسة لـ K-Means في فضاء 4D.""",

    36: """# الخطوة 4) تحضير الميزات وقياسها

X = dataset[FEATURE_COLS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'شكل X: {X.shape}')
print('تم قياس جميع الميزات الأربع — جاهز لـ K-Means.')""",

    37: """## المثال 2 — الخلية 5: اختيار K وتدريب K-Means

اختر K عبر معامل Silhouette، ثم درّب النموذج النهائي.

**ما تفعله هذه الخلية:** يختار K ويعيّن تسميات المجموعات لجميع العملاء.""",

    38: """# الخطوة 5) اختيار K وتدريب K-Means

K_range = range(2, 9)
silhouettes = []
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

best_k = list(K_range)[np.argmax(silhouettes)]
print(f'أفضل K: {best_k} (silhouette = {max(silhouettes):.4f})')

kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
dataset['Cluster'] = cluster_labels

print(f'\\nالعملاء لكل مجموعة:')
print(dataset['Cluster'].value_counts().sort_index())""",

    39: """## المثال 2 — الخلية 6: ملفات تعريف المجموعات

اعرض **المتوسط** لكل ميزة لكل مجموعة — تفسير شرائح العملاء.

**ما تفعله هذه الخلية:** يلخّص شكل كل مجموعة من منظور الأعمال.""",

    40: """# الخطوة 6) ملفات تعريف المجموعات — متوسط الميزات لكل مجموعة

profile = dataset.groupby('Cluster')[FEATURE_COLS].mean().round(2)
profile['Count'] = dataset.groupby('Cluster').size()

print('ملفات تعريف المجموعات (متوسط قيم الميزات):')
display(profile)

print('\\nتلميح للتفسير:')
print('  دخل مرتفع + إنفاق مرتفع → عملاء مميزون')
print('  دخل منخفض + إنفاق منخفض → شريحة اقتصادية')""",

    41: """## المثال 2 — الخلية 7: التصور

ارسم المجموعات باستخدام **PCA** (إسقاط ثنائي الأبعاد لـ 4 ميزات) واعرض **أحجام المجموعات**.

**ما تفعله هذه الخلية:** يصوّر المجموعات عالية الأبعاد في 2D.""",

    42: """# الخطوة 7) التصور — إسقاط PCA ثنائي الأبعاد + أحجام المجموعات

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)  # إسقاط 4D → 2D للرسم

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

palette = sns.color_palette('tab10', best_k)
for c in range(best_k):
    mask = cluster_labels == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[palette[c]], label=f'المجموعة {c}', alpha=0.7, s=50)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title(f'مجموعات K-Means (عرض PCA، K={best_k})')
axes[0].legend(fontsize=8)

dataset['Cluster'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_xlabel('المجموعة')
axes[1].set_ylabel('عدد العملاء')
axes[1].set_title('أحجام المجموعات')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print(f'التباين المفسَّر بـ PCA: {pca.explained_variance_ratio_.sum():.2%} من إجمالي التباين')""",

    43: """## المثال 2 — الخلية 8: التقييم

احسب **Inertia** و **Silhouette Score** للتجميع متعدد الميزات.

**ما تفعله هذه الخلية:** يقيّم جودة المجموعات في فضاء مقيّس رباعي الأبعاد.""",

    44: """# الخطوة 8) التقييم

sil_avg = silhouette_score(X_scaled, cluster_labels)

results = pd.DataFrame({
    'Metric': ['Inertia', 'Silhouette Score', 'Number of Clusters', 'Features'],
    'Value': [kmeans.inertia_, sil_avg, best_k, len(FEATURE_COLS)],
    'Description': [
        'مجموع مربعات المسافات داخل المجموعة',
        'جودة فصل المجموعات الإجمالية',
        'K المختار بـ Silhouette',
        'عدد ميزات الإدخال'
    ]
})

display(results.round(4))

print(f'\\nمعامل Silhouette للمثال 2 = {sil_avg:.4f}')""",

    45: """## لماذا يعمل K-Means لشرائح العملاء؟

| # | السبب | الشرح |
|---|--------|-------------|
| 1 | **ميزات متعددة** | الدخل والإنفاق والعمر والمشتريات تحدد أنماط حياة مميزة |
| 2 | **ملفات تعريف المجموعات** | المتوسط لكل مجموعة يعطي شرائح أعمال قابلة للتنفيذ |
| 3 | **تصور PCA** | يُسقِط مجموعات 4D إلى 2D للرسم |
| 4 | **المقياس ضروري** | Income (~60k) مقابل Num_Purchases (~5) — يجب التقييس |
| 5 | **ضبط Silhouette** | يختار K الذي يزيد فصل المجموعات |

> **حالة الاستخدام:** فرق التسويق تسمّي المجموعات (مثلاً "Premium"، "Budget") بناءً على جداول الملفات التعريفية.""",

    46: """## فهم مقاييس التجميع — المثال 2

| Silhouette | التفسير |
|------------|----------------|
| **> 0.7** | مجموعات قوية ومفصولة جيدًا |
| **0.5 – 0.7** | بنية معقولة |
| **0.25 – 0.5** | بنية ضعيفة لكن قابلة للاستخدام |
| **< 0.25** | تجميع ضعيف — جرّب K مختلفًا أو خوارزمية أخرى |

> **القيود:** K-Means يفترض مجموعات **كروية** — إن كانت الشرائح ممدودة، جرّب **DBSCAN** أو **GMM**.""",
}
