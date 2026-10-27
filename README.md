# 🩻 MedData Doctor - فحص صحة وجودة البيانات الطبية

Screening and health check-up tool for medical datasets before training machine learning models.

---

## 🚀 طريقة الرفع والنشر المباشر على Vercel (Deployment on Vercel)

المشروع الآن مهيأ بنسبة 100% ليعمل على **Vercel** بدون أي إعدادات معقدة وبشكل مجاني تماماً وبدون توقف عن طريق تقنية **Stlite (WebAssembly Streamlit in the browser)**.

### خطوات النشر على Vercel:
1. **رفع المحتوى إلى GitHub:**
   - قم بإنشاء مستودع جديد (New Repository) على حسابك في GitHub.
   - ارفع محتويات هذا المجلد (`hackathon`) بالكامل إلى المستودع.
2. **الربط مع Vercel:**
   - ادخل على حسابك في [Vercel](https://vercel.com).
   - اضغط على **Add New...** ثم اختر **Project**.
   - اختر المستودع الخاص بك من القائمة (Import).
   - اترك جميع الإعدادات الافتراضية كما هي (Framework Preset: **Other** / Output Directory: **Empty/Default**).
   - اضغط على **Deploy**.
3. **مبروك! 🎉**
   - سيعمل التطبيق فوراً وسيعطيك Vercel رابطاً مجانياً ومباشراً للعمل والتجربة.

---

## 🌐 بدائل أخرى للنشر (Alternative Deployment)

### نشر التطبيق على Streamlit Community Cloud:
المشروع متوافق تماماً مع Streamlit Cloud:
1. اربط حساب GitHub مع [share.streamlit.io](https://share.streamlit.io).
2. اختر المستودع، وضع مسار الملف الرئيسي: `app.py`.
3. اضغط **Deploy**.

---

## 💻 التشغيل المحلي (Run Locally)

### الطريقة الأولى: تشغيل Streamlit الأصلي
```bash
pip install -r requirements.txt
streamlit run app.py
```

### الطريقة الثانية: تشغيل كـ Static Web App
```bash
python -m http.server 8000
```
ثم افتح المتصفح على: `http://localhost:8000`

---

## 📁 هيكل المشروع (Project Files)

- `app.py`: الواجهة التفاعلية الرئيسية لتطبيق Streamlit.
- `audit.py`: محرك الفحص الذكي (فحص القيم غير المنطقية، كشف تسريب البيانات، فحص التوازن، وتدريب نماذج المقارنة).
- `rules.py`: القواعد والنطاقات الطبية المعيارية للفحوصات الحيوية.
- `demo.py`: مولد بيانات طبية افتراضية لاختبار المنصة فوراً بدون الحاجة لتحميل ملف.
- `index.html`: ملف التشغيل لـ Vercel والمبني بتقنية WebAssembly (Stlite).
- `vercel.json`: إعدادات وتوجيهات منصة Vercel لضمان سرعة التحميل وسلامة الملفات.
- `requirements.txt`: متطلبات ومكتبات بايثون.
