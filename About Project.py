import streamlit as st
from styles import *

streamlit_style()

st.markdown("<h1 style='text-align: center; color: #4f4f4f;'>About Project</h1>",
            unsafe_allow_html = True)

st.markdown("<h4 style='text-align: center; color: #4f4f4f;'>Sub-task : Create a page describing project</h4>",
            unsafe_allow_html = True)




















































# import yaml
# from styles import *

# streamlit_style()

# st.markdown("<h1 style='text-align: center; color: #468189;'>Welcome to MobileUurka App</h1>",
#                 unsafe_allow_html = True)

# def sidebar():
#     st.sidebar.title(f"Welcome User")
#     authenticator.logout("Logout", "sidebar")

# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

# name, authentication_status, username = authenticator.login("Login", "main")

# if authentication_status == False:
#     st.error("Username/password is incorrect")

# if authentication_status == None:
#     st.warning("Please enter your username and password")

# st.session_state['validated'] = authentication_status

# if authentication_status:

#     sidebar()
    
#     st.subheader(" ")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col2:
#         st.image('mobileuurka.jpeg')

#     st.subheader(" ")

#     st.markdown("<h3 style='text-align: center; color: #468189;'>Open the Sidebar to proceed</h3>",
#                 unsafe_allow_html = True)




