import streamlit as st
import streamlit.components.v1 as components


# ---- HIDE STREAMLIT STYLE ----

def streamlit_style():

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.set_page_config(initial_sidebar_state = "collapsed", layout = 'centered', page_icon = 'logo.png', page_title = 'Mobile Uurka')

    st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)

    st.markdown("""
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """, unsafe_allow_html=True)

    hvar = """
    
            <script>
            
                var elements = window.parents.document.querySelectorAll('.streamlit-expanderHeader')
                elements[0].style.color = 'rgba(83, 36, 118, 1)';
                elements[0].style.fontFamily = 'Didot';
                elements[0].style.fontStyle = 'x-large';
                elements[0].style.fontWeight = 'bold';

            </script>    
            """

    # st.markdown(""" 
    #             <style>
    #             div.stButton > button:first-child {
    #             background-color: #2a9d8f; 
    #             color:white; 
    #             font-size:14px; 
    #             height:3em; 
    #             text-align:center; 
    #             width:10em; 
    #             border-radius:40px 40px 40px 40px;ß
    #             }
    #             </style>
    #             """, unsafe_allow_html=True)