import streamlit as st
import joblib
import pandas as pd
# استدعاء الكلاس الخاص بك ليفهم الموديل كيفية معالجة البيانات
from preprocessor import CustomPreprocessor
# 1. إعدادات الصفحة
st.set_page_config(page_title="FlightVerdict", page_icon="✈️", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #87CEEB 0%, #F0F8FF 100%);
    }
    h1, h2, h3, p, label {
        color: #003366 !important;
    }
    .stButton>button {
        background-color: #0074D9;
        color: white;
        border-radius: 20px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
# 2. تحميل الموديل (تم تحميله مرة واحدة فقط لسرعة الأداء)
@st.cache_resource
def load_model():
    # تأكدي أن ملف model.joblib موجود في نفس المجلد
    return joblib.load('model.joblib')

model = load_model()

# 3. واجهة المستخدم (العناوين)
st.title("✈️ FlightVerdict")
st.markdown("### Predicting Passenger Satisfaction using Machine Learning")
st.write("Please enter the flight details and passenger ratings below:")

# 4. تقسيم المدخلات إلى أعمدة لجعل التصميم أجمل

# 4. تقسيم المدخلات إلى تبويبات لتنظيم الـ 14 خدمة
tab1, tab2, tab3 = st.tabs(["📋 Passenger Info", "🛋️ Comfort & Entertainment", "🛡️ Service & Logistics"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder=" ")
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"], index=None, placeholder=" ")
        type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"], index=None, placeholder=" ")
        arrival_delay = st.number_input("Arrival Delay (min)", 0, 1000, 0)
    with col2:
        flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"], index=None, placeholder=" ")
        age = st.number_input("Age", 1, 100, 25)
        flight_distance = st.number_input("Flight Distance (km)", 1, 10000, 1000)
        departure_delay = st.number_input("Departure Delay (min)", 0, 1000, 0)

with tab2:
    # التقييمات الخاصة بالراحة والرفاهية
    wifi_service = st.slider("Inflight wifi service", 0, 5)
    online_booking = st.slider("Ease of Online booking", 0, 5)
    food_drink = st.slider("Food and drink", 0, 5)
    seat_comfort = st.slider("Seat comfort", 0, 5)
    cleanliness = st.slider("Cleanliness", 0, 5)
    entertainment = st.slider("Inflight entertainment", 0, 5)
with tab3:
    # التقييمات الخاصة بالخدمات اللوجستية والمطار
    on_board = st.slider("On-board service", 0, 5)
    leg_room = st.slider("Leg room service", 0, 5)
    baggage = st.slider("Baggage handling", 0, 5)
    checkin = st.slider("Check-in service", 0, 5)
    inflight_serv = st.slider("Inflight service", 0, 5)
    online_boarding = st.slider("Online boarding", 0, 5)
    gate_loc = st.slider("Gate location", 0, 5)
    time_conv = st.slider("Departure/Arrival time convenient", 0, 5)
# 5. زر التنبؤ
if st.button("Analyze Satisfaction"):
    # تجهيز البيانات بالأسماء "الصغيرة" المتوقعة في الـ Preprocessor
    # ملاحظة: unnamed:_0 يجب أن يكون سمول كما طلب الموديل


    data = {
        'unnamed:_0': [0],
        'id': [0],
        'gender': [gender],
        'customer_type': [customer_type],
        'age': [age],
        'type_of_travel': [type_of_travel],
        'class': [flight_class],
        'flight_distance': [flight_distance],
        'inflight_wifi_service': [wifi_service],
        'departure/arrival_time_convenient': [time_conv],
        'ease_of_online_booking': [online_booking],
        'gate_location': [gate_loc],
        'food_and_drink': [food_drink],
        'online_boarding': [online_boarding],
        'seat_comfort': [seat_comfort],
        'inflight_entertainment': [entertainment],
        'on-board_service': [on_board],
        'leg_room_service': [leg_room],
        'baggage_handling': [baggage],
        'checkin_service': [checkin],
        'inflight_service': [inflight_serv],
        'cleanliness': [cleanliness],
        'departure_delay_in_minutes': [departure_delay],
        'arrival_delay_in_minutes': [arrival_delay]
    }
    input_df = pd.DataFrame(data)

    # تنفيذ التنبؤ باستخدام الـ Pipeline
    # الـ Pipeline سيمر عبر الـ CustomPreprocessor أولاً
    prediction = model.predict(input_df)[0]
    
    # حساب الاحتمالية (Confidence)
    proba = model.predict_proba(input_df)[0]
    
    # 6. عرض النتيجة
    #st.divider()
    
    # فحص النتيجة (سواء كانت 1 أو 'satisfied') حسب كيف تم تدريب الموديل
    #if str(prediction).lower() == 'satisfied' or prediction == 1:
       # confidence = proba[1] * 100
      #  st.success(f"### Result: SATISFIED (Confidence: {confidence:.2f}%) 😊")
      #  st.balloons()
   # else:
        # إذا كانت النتيجة محايدة أو غير راضية، نأخذ احتمال الكلاس الأول
     #   confidence = proba[0] * 100
       # st.error(f"### Result: NEUTRAL or DISSATISFIED (Confidence: {confidence:.2f}%) ☹️")
    # 6. عرض النتيجة (تصميم احترافي ومطور)
    st.divider()
    
    # فحص الحالة وتحديد الألوان
    is_satisfied = (str(prediction).lower() == 'satisfied' or prediction == 1)
    
    if is_satisfied:
        confidence = proba[1] * 100
        result_text = "SATISFIED"
        st.success(f"### Result: {result_text} ")
        st.balloons()
    else:
        confidence = proba[0] * 100
        result_text = "DISSATISFIED"
        st.error(f"### Result: {result_text} ")
        st.snow()
    # عرض تفاصيل التحليل في أعمدة جذابة
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(label="Status", value=result_text)
    
    with col_b:
        # اللون الأخضر للرضا والأحمر لعدم الرضا في المقياس
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%", 
                  delta=f"{'+' if is_satisfied else '-'} Analysis Strength")

    # إضافة شريط تقدم مرئي يوضح مدى قوة التوقع
    st.write("**Prediction Analysis Strength:**")
    st.progress(int(confidence))

    # إضافة لمسة تحليلية إضافية (اختياري)
    with st.expander("Show detailed probability breakdown"):
        st.write(f"Probability of being Satisfied: {proba[1]:.2%}")
        st.write(f"Probability of being Neutral/Dissatisfied: {proba[0]:.2%}")
    st.write("---")
    st.subheader("📊 Visual Analytics Breakdown")

    # إنشاء عمودين للرسوم البيانية
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.write("**Prediction Probability**")
        # Pie Chart لنسبة الثقة
        pie_data = pd.DataFrame({
            'Status': ['Satisfied', 'Neutral/Dissatisfied'],
            'Probability': [proba[1], proba[0]]
        })
        import plotly.express as px
        fig_pie = px.pie(pie_data, values='Probability', names='Status', 
                         color='Status',
                         color_discrete_map={'Satisfied':'#28a745', 'Neutral/Dissatisfied':'#dc3545'},
                         hole=0.4) # جعلها بشكل Donut لجعلها أجمل
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
        st.plotly_chart(fig_pie, use_container_width=True)

    with viz_col2:
        st.write("**Service Ratings Summary**")
        # Bar Chart للخدمات الخمس التي تم تقييمها
        ratings_data = pd.DataFrame({
            'Service': ['WiFi', 'Booking', 'Food', 'Seat', 'Cleanliness'],
            'Score': [wifi_service, online_booking, food_drink, seat_comfort, cleanliness]
        })
        fig_bar = px.bar(ratings_data, x='Service', y='Score', 
                         color='Score',
                         color_continuous_scale='RdYlGn', # تدرج من الأحمر للأخضر
                         range_y=[0, 5])
        fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)