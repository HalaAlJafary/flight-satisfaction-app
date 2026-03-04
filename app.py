import streamlit as st
import joblib
import pandas as pd
from preprocessor import CustomPreprocessor
import plotly.express as px

st.set_page_config(page_title="FlightVerdict", page_icon="✈️", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #87CEEB 0%, #F0F8FF 100%);}
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border-radius: 5px;
        color: #003366 !important;}
    div[data-baseweb="input"] {
        background-color: white !important;
        border-radius: 5px;}
    div[data-baseweb="input"] button {
        background-color: white !important;
        color: #003366 !important; 
        border: none !important;}
    div[data-baseweb="input"] input {
        background-color: white !important;
        color: #003366 !important;}
    h1, h2, h3, p, label, .stMarkdown {
        color: #003366 !important; }
    .stTabs [data-baseweb="tab-list"] button {
        color: #003366 !important;}
    .stButton>button {
        background-color: #0074D9;
        color: white !important;
        border-radius: 20px;
        width: 100%;}
    </style>
    """, unsafe_allow_html=True)
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')
model = load_model()


st.title("✈️ FlightVerdict")
st.markdown("### Predicting Passenger Satisfaction using Machine Learning")
st.markdown("### 📋 Passenger Info")
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

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🛋️ Comfort & Entertainment")
    wifi_service = st.select_slider("Inflight wifi service",options=["",0,1,2,3,4,5],value="")
    online_booking = st.select_slider("Ease of Online booking", options=["",0,1,2,3,4,5], value="")
    food_drink = st.select_slider("Food and drink", options=["",0,1,2,3,4,5], value="")
    seat_comfort = st.select_slider("Seat comfort", options=["",0,1,2,3,4,5], value="")
    cleanliness = st.select_slider("Cleanliness", options=["",0,1,2,3,4,5], value="")
    entertainment = st.select_slider("Inflight entertainment", options=["",0,1,2,3,4,5], value="")
with col2:
    st.markdown("### 🛡️ Service & Logistics")
    on_board = st.select_slider("On-board service", options=["",0,1,2,3,4,5], value="")
    leg_room = st.select_slider("Leg room service", options=["",0,1,2,3,4,5], value="")
    baggage = st.select_slider("Baggage handling", options=["",0,1,2,3,4,5], value="")
    checkin = st.select_slider("Check-in service", options=["",0,1,2,3,4,5], value="")
    inflight_serv = st.select_slider("Inflight service", options=["",0,1,2,3,4,5], value="")
    online_boarding = st.select_slider("Online boarding", options=["",0,1,2,3,4,5], value="")


sliders = [wifi_service, online_booking, food_drink, seat_comfort, cleanliness, 
           entertainment, on_board, leg_room, baggage, checkin, inflight_serv, online_boarding]

all_sliders_touched = all(s != "" for s in sliders)

is_ready = (gender is not None) and (customer_type is not None) and (type_of_travel is not None) and (flight_class is not None) and all_sliders_touched 

if st.button("Analyze Satisfaction", disabled=not is_ready):

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
        'departure/arrival_time_convenient': 5,
        'ease_of_online_booking': [online_booking],
        'gate_location': 5,
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
        'arrival_delay_in_minutes': [arrival_delay]}
    input_df = pd.DataFrame(data)
    prediction = model.predict(input_df)[0]    
    proba = model.predict_proba(input_df)[0]   
    st.divider()    
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
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(label="Status", value=result_text)
    
    with col_b:
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%", 
                  delta=f"{'+' if is_satisfied else '-'} Analysis Strength")

    st.write("**Prediction Analysis Strength:**")
    st.progress(int(confidence))

    with st.expander("Show detailed probability breakdown"):
        st.write(f"Probability of being Satisfied: {proba[1]:.2%}")
        st.write(f"Probability of being Neutral/Dissatisfied: {proba[0]:.2%}")
    st.write("---")
    st.subheader("📊 Visual Analytics Breakdown")
    pie_data = pd.DataFrame({'Status': ['Satisfied', 'Neutral/Dissatisfied'],'Probability': [proba[1], proba[0]]})
    fig_pie = px.pie(pie_data, values='Probability', names='Status',color='Status',color_discrete_map={'Satisfied':'#28a745','Neutral/Dissatisfied':'#dc3545'},hole=0.4)
    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)
    st.write("---")
    st.markdown("### 🎯 **Feature Importance Analysis **")

    static_importance_data = {
        'Feature': [
            'Total Delay', 'Departure/Arrival Time Convenient', 'Flight Distance', 
            'Age', 'Baggage Handling', 'Inflight Service', 'Checkin Service', 
            'On-board Service', 'Leg Room Service', 'Inflight Experience Score', 
            'Customer Type', 'Inflight Entertainment', 'Class', 'Type Of Travel', 
            'Digital Experience', 'Online Boarding'],
        'Importance': [
            0.005, 0.012, 0.015, 0.018, 0.019, 0.021, 0.022, 0.026, 
            0.035, 0.039, 0.044, 0.089, 0.126, 0.130, 0.147, 0.260]}

    df_static = pd.DataFrame(static_importance_data)

    df_static['Importance_Pct'] = df_static['Importance'] * 100

    fig = px.bar(
        df_static,
        x='Importance_Pct',
        y='Feature',
        orientation='h',
        color='Importance_Pct',
        color_continuous_scale='Turbo', 
        text='Importance_Pct',
        title="Impact of Each Factor on the Verdict (%)")

    fig.update_traces(
        texttemplate='%{text:.1f}%', 
        textposition='outside',
        cliponaxis=False)

    fig.update_layout(
        showlegend=False,
        height=700,
        margin=dict(l=20, r=70, t=50, b=20), 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#003366", size=13),
        xaxis_title="Influence Percentage (%)",
        yaxis_title=None,
        xaxis=dict(range=[0, 30], showgrid=False),
        coloraxis_showscale=False)

    st.plotly_chart(fig, use_container_width=True)
else:
    if not is_ready:
        st.info("Please fill in all fields")
