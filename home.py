import streamlit as st


st.set_page_config(
    page_title="Attendance App",
    layout="wide"
)

st.header('Attendance System using Face Recognition')


with st.spinner('Loading models and connecting to redis DB...'):
    import face_rec

st.success('Model loaded successsfully')
st.success('Redis db successfully connected')

