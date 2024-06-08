import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import os
from keybert import KeyBERT




score_to_category = {
    1: 'Very Dissatisfied',
    2: 'Dissatisfied',
    3: 'Neutral',
    4: 'Satisfied',
    5: 'Very Satisfied'
}


def initialize_state():
    # Initialize session states with default values if not already present
    keys = ['previous_dashboard', 'selected_role', 'selected_function', 'selected_location', 'uploaded_file']
    defaults = [None, [], [], [], None]
    for key, default in zip(keys, defaults):
        if key not in st.session_state:
            st.session_state[key] = default


def reset_filters():
    st.session_state['selected_role'] = []
    st.session_state['selected_function'] = []
    st.session_state['selected_location'] = []


st.set_page_config(layout="wide")
initialize_state()


# Load and clean data
@st.cache_data(persist=True)
def load_data():
    # Load data and cache the DataFrame to avoid reloads on each user interaction
    url = 'https://github.com/001202ZHENG/V1_Chatbot_Streamlit/raw/main/data/Voice%20of%20Customer_Second%20data%20set.xlsx'
    data = pd.read_excel(url)
    return data


data = load_data()

# General Page Layout
st.markdown(
    '''
    <style>
        .main .block-container {
            padding-top: 0.25rem;
            padding-right: 0.25rem;
            padding-left: 0.25rem;
            padding-bottom: 0.25rem;
        }
        h1 {
            margin-top: 0rem;
            margin-bottom: 0rem;
        }
        h3 {
            margin-top: 0rem;
            margin-bottom: 0rem;
        }
    </style>
    ''',
    unsafe_allow_html=True
)


# Header Function
def render_header(title, subtitle=None):
    style = style = """
    <style>
        h1.header, h3.subheader {
            background-color: #336699; /* Steel blue background */
            color: white; /* White text color */
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 15px 0;
            height: auto
        }
        h1.header {
            margin-bottom: 0;
            font-size: 30px;
        }
        h3.subheader {
            font-size: 20px;
            font-weight: normal;
            margin-top: 0;
        }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    st.markdown(f'<h1 class="header">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<h3 class="subheader">{subtitle}</h3>', unsafe_allow_html=True)


# Sidebar for dashboard selection
dashboard = st.sidebar.radio("Select Dashboard", ('General Survey Results',
                                                  'Section 1: Employee Experience',
                                                  'Section 2: Recruiting & Onboarding',
                                                  'Section 3: Performance & Talent',
                                                  'Section 4: Learning',
                                                  'Section 5: Compensation',
                                                  'Section 6: Payroll',
                                                  'Section 7: Time Management',
                                                  'Section 8: User Experience'
                                                  ))

if dashboard != st.session_state['previous_dashboard']:
    reset_filters()  # Reset filters if dashboard changed
    st.session_state['previous_dashboard'] = dashboard


@st.cache_data
def get_unique_values(column):
    return data[column].unique()


roles = get_unique_values('What is your role at the company ?')
functions = get_unique_values('What function are you part of ?')
locations = get_unique_values('Where are you located ?')

st.sidebar.multiselect('Select Role', options=roles, default=st.session_state['selected_role'], key='selected_role')
st.sidebar.multiselect('Select Function', options=functions, default=st.session_state['selected_function'],
                       key='selected_function')
st.sidebar.multiselect('Select Location', options=locations, default=st.session_state['selected_location'],
                       key='selected_location')


def apply_filters(data, roles, functions, locations):
    filtered = data
    if roles:
        filtered = filtered[filtered['What is your role at the company ?'].isin(roles)]
    if functions:
        filtered = filtered[filtered['What function are you part of ?'].isin(functions)]
    if locations:
        filtered = filtered[filtered['Where are you located ?'].isin(locations)]
    return filtered


# Use the function with both a title and a subtitle
if dashboard == 'General Survey Results':
    render_header("General Survey Results")
elif dashboard == 'Section 1: Employee Experience':
    render_header("Employee Experience: General HR Services Evaluation")
elif dashboard == 'Section 2: Recruiting & Onboarding':
    render_header("Recruiting & Onboarding")
elif dashboard == 'Section 3: Performance & Talent':
    render_header("Performance & Talent")
elif dashboard == 'Section 4: Learning':
    render_header("Learning")
elif dashboard == 'Section 5: Compensation':
    render_header("Compensation")
elif dashboard == 'Section 6: Payroll':
    render_header("Payroll")
elif dashboard == 'Section 7: Time Management':
    render_header("Time Management")
elif dashboard == 'Section 8: User Experience':
    render_header("User Experience")

if dashboard == 'Section 1: Employee Experience':
    @st.cache(allow_output_mutation=True)
    def get_sentiment_analyzer():
        return pipeline("sentiment-analysis")
    
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
    

    sentiment_analyzer = get_sentiment_analyzer()
    
    kw_model = KeyBERT()

    st.title("Sentiment Analysis App")
    
    def get_transformer_sentiment(text):
        result = sentiment_analyzer(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']
    
    # Function to extract keywords using KeyBERT
    def extract_keywords(text):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=5)
        return ', '.join([word for word, _ in keywords])

    # Apply transformer-based sentiment analysis to each text in the DataFrame
    filtered_data['Communication_Sentiment_Score1'] = filtered_data.iloc[:, 14].apply(get_transformer_sentiment)

    # Identify top 5 positive and negative texts
    top_5_positive = filtered_data.nlargest(5, 'Communication_Sentiment_Score1')
    top_5_negative = filtered_data.nsmallest(5, 'Communication_Sentiment_Score1')

    # Extract keywords for top 5 positive and negative texts
    top_5_positive['Key Phrases'] = top_5_positive.iloc[:, 14].apply(extract_keywords)
    top_5_negative['Key Phrases'] = top_5_negative.iloc[:, 14].apply(extract_keywords)

    # Columns to display
    columns_to_display1 = ['From 1 to 5, how satisfied are you with the communication channels used to relay important HR information to employees?', 'Key Phrases']

    
    # Columns to display
    columns_to_display2 = ['From 1 to 5, how satisfied are you with the communication channels used to relay important HR information to employees?', 'Which reason(s) drive that score ?']

    # Display tables in Streamlit
    st.write("Top 5 Positive Key Phrases")
    st.table(top_5_positive[columns_to_display1])

    st.write("Top 5 Negative Key Phrases")
    st.table(top_5_negative[columns_to_display1])

    st.write("Top 5 Positive Responses")
    st.table(top_5_positive[columns_to_display2])

    st.write("Top 5 Negative Responses")
    st.table(top_5_negative[columns_to_display2])

    
    





@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])

