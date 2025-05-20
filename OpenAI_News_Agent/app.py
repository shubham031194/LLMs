import streamlit as st
from agents import NewsAgent

if 'src_list' not in st.session_state:
  st.session_state.src_list = []

if 'agent' not in st.session_state:
  st.session_state.agent = None

if 'selected_items' not in st.session_state:
  st.session_state.selected_items = []

st.title("NewsAgent")
input_string = st.text_input("Enter the keyword:")  
submitted = st.button("Search")
if submitted:
  st.session_state.agent = NewsAgent(input_string)
  src_list = st.session_state.agent.fetchNewsAlerts()
  st.session_state.src_list = list(set(src_list))

if st.session_state.src_list:
  select_opt = st.multiselect('Sorces: ',st.session_state.src_list)
  submitted_filter = st.button("Genrate Article")
  if submitted_filter:
    st.session_state.selected_items = select_opt
    st.session_state.agent.gatherNewsData(st.session_state.selected_items)
    final_srticle = st.session_state.agent.final_article_genration()
    st.write(final_srticle)
