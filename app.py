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
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.util import ngrams




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

def prepare_summaries(data):
    continent_to_country_code = {
        'Asia': 'KAZ',
        'Oceania': 'AUS',
        'North America': 'CAN',
        'South America': 'BRA',
        'Europe': 'DEU',
        'Africa': 'TCD'
    }
    country_code_to_continent = {v: k for k, v in continent_to_country_code.items()}
    location_summary = pd.DataFrame(data['Where are you located ?'].value_counts()).reset_index()
    location_summary.columns = ['Continent', 'Count']
    location_summary['Country_Code'] = location_summary['Continent'].map(continent_to_country_code)
    location_summary['Label'] = location_summary['Continent'].apply(
        lambda x: f"{x}: {location_summary.loc[location_summary['Continent'] == x, 'Count'].iloc[0]}")

    role_summary = pd.DataFrame(data['What is your role at the company ?'].value_counts()).reset_index()
    role_summary.columns = ['Role', 'Count']
    function_summary = pd.DataFrame(data['What function are you part of ?'].value_counts()).reset_index()
    function_summary.columns = ['Function', 'Count']
    return location_summary, role_summary, function_summary

# MARIAS SCORE DISTRIBUTION FUNCTION
def score_distribution(data, column_index):
    # Extract the data series based on the column index
    data_series = data.iloc[:, column_index]

    # Calculate the percentage of each response
    value_counts = data_series.value_counts(normalize=True).sort_index() * 100

    # Ensure the value_counts includes all categories with zero counts for missing categories
    value_counts = value_counts.reindex(range(1, 6), fill_value=0)

    # Create the DataFrame

    # Calculate the median score
    raw_counts = data_series.value_counts().sort_index()
    scores = np.repeat(raw_counts.index, raw_counts.values)
    median_score = np.median(scores)

    return value_counts, median_score

satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']

def filter_by_satisfaction(data, satisfaction_level, column_index):
    if satisfaction_level != 'Select a satisfaction level':
        data = data[data.iloc[:, column_index] == satisfaction_options.index(satisfaction_level)]
    return data

############ SENTIMENT ANALYSIS FUNCTION STARTS ############
def generate_wordclouds(df, score_col_idx, reasons_col_idx, custom_stopwords):
    # Custom stopwords
    stopwords_set = set(STOPWORDS)
    stopwords_set.update(custom_stopwords)

    # Filter the DataFrame for scores 4 and 5
    df_high_scores = df[df.iloc[:, score_col_idx].isin([4, 5])]

    # Filter the DataFrame for scores 1, 2, and 3
    df_low_scores = df[df.iloc[:, score_col_idx].isin([1, 2, 3])]

    # Generate the text for word clouds
    text_high_scores = ' '.join(df_high_scores.iloc[:, reasons_col_idx].astype(str))
    text_low_scores = ' '.join(df_low_scores.iloc[:, reasons_col_idx].astype(str))

    # Generate the word clouds
    wordcloud_high_scores = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set, collocations=False).generate(text_high_scores)
    wordcloud_low_scores = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set, collocations=False).generate(text_low_scores)

    # Create columns for displaying the word clouds side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud for High Scores</h3>", unsafe_allow_html=True)
        fig_high_scores, ax_high_scores = plt.subplots(figsize=(10, 5))
        ax_high_scores.imshow(wordcloud_high_scores, interpolation='bilinear')
        ax_high_scores.axis('off')
        st.pyplot(fig_high_scores)

    with col2:
        st.markdown("<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud for Low Scores</h3>", unsafe_allow_html=True)
        fig_low_scores, ax_low_scores = plt.subplots(figsize=(10, 5))
        ax_low_scores.imshow(wordcloud_low_scores, interpolation='bilinear')
        ax_low_scores.axis('off')
        st.pyplot(fig_low_scores)

def get_transformer_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']
    
    # Function to extract keywords using KeyBERT
def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=5)
    return ', '.join([word for word, _ in keywords])


@st.cache(allow_output_mutation=True)
def get_sentiment_analyzer():
    return pipeline("sentiment-analysis")
# Usage
sentiment_analyzer = get_sentiment_analyzer()
kw_model = KeyBERT()

############ SENTIMENT ANALYSIS FUNCTION ENDS ############




if dashboard == 'Section 1: Employee Experience':
    # Apply filters to the data    
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
    
    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )

    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])
    
    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q11ValuesCount, q11MedianScore = score_distribution(filtered_data, 13)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q11ValuesCount.values})

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on HR Communication Channels</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q11MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="rating_hr_communication_channels_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_dropdown2 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown2')

        satisfaction_filtered_data2 = filter_by_satisfaction(filtered_data, satisfaction_dropdown2, 13)

        location_summary2, role_summary2, function_summary2 = prepare_summaries(satisfaction_filtered_data2)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role2 = px.bar(role_summary2, y='Role', x='Count', orientation='h')
        fig_role2.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role2.update_traces(marker_color='#336699', text=role_summary2['Count'], textposition='outside')
        fig_role2.update_yaxes(showticklabels=True, title='')
        fig_role2.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role2, use_container_width=True, key="roles_bar_chart2")

        fig_function2 = px.bar(function_summary2, y='Function', x='Count', orientation='h')
        fig_function2.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function2.update_traces(marker_color='#336699', text=function_summary2['Count'], textposition='outside')
        fig_function2.update_yaxes(showticklabels=True, title='')
        fig_function2.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function2, use_container_width=True, key="functions_bar_chart2")
    
    # Question 9: Which reason(s) drive that score ?
    st.title("Sentiment Analysis App")

    # Display the reasons for communication channel satisfaction
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">The Reasons for Ratings on Communication Channels</h1>', unsafe_allow_html=True)

    # Define custom stopwords for the word clouds
    communication_stopwords = ["communication", "channels", "HR", "information", "important", "informed", "stay", "communicated", "employees", "company", "help", "communicates", "need", "everyone", "makes"]

    # Run this code in a Streamlit app
    if __name__ == "__main__":
        st.markdown("<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Word Cloud Visualization</h1>", unsafe_allow_html=True)
        generate_wordclouds(filtered_data, 13, 14, communication_stopwords)

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

if dashboard == 'Section 3: Performance & Talent':
    # Apply filters to the data
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
    
    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    ### Question19: From 1 to 5, how satisfied are you with the company's performance evaluation and feedback process ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q19ValuesCount, q19MedianScore = score_distribution(filtered_data, 26)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q19ValuesCount.values})

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Company's Performance Evaluation and Feedback Process</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q19MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown1 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown1')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 26)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")
    
    
    ### Question20: Which reason(s) drive that score ?
    st.title("Sentiment Analysis App")

    # Display the reasons for performance evaluation and feedback process satisfaction
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">The Reasons for Ratings on Performance Evaluation and Feedback Process</h1>', unsafe_allow_html=True)

    # Define custom stopwords for the word clouds
    performance_stopwords = ["performance", "evaluation", "feedback", "process", "talent", "employees", "company", "help", "need", "everyone", "makes"]

    # Run this code in a Streamlit app
    if __name__ == "__main__":
        st.markdown("<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Word Cloud Visualization</h1>", unsafe_allow_html=True)
        generate_wordclouds(filtered_data, 26, 27, performance_stopwords)
    
    # Apply transformer-based sentiment analysis to each text in the DataFrame
    filtered_data['Performance_Sentiment_Score1'] = filtered_data.iloc[:, 27].apply(get_transformer_sentiment)

    # Identify top 5 positive and negative texts
    top_5_positive = filtered_data.nlargest(5, 'Performance_Sentiment_Score1')
    top_5_negative = filtered_data.nsmallest(5, 'Performance_Sentiment_Score1')

    # Extract keywords for top 5 positive and negative texts
    top_5_positive['Key Phrases'] = top_5_positive.iloc[:, 27].apply(extract_keywords)
    top_5_negative['Key Phrases'] = top_5_negative.iloc[:, 27].apply(extract_keywords)

    # Columns to display
    columns_to_display1 = ['From 1 to 5, how satisfied are you with the company\'s performance evaluation and feedback process ?', 'Key Phrases']

    # Columns to display
    columns_to_display2 = ['From 1 to 5, how satisfied are you with the company\'s performance evaluation and feedback process ?', 'Which reason(s) drive that score ?2']

    # Display tables in Streamlit
    st.write("Top 5 Positive Key Phrases")
    st.table(top_5_positive[columns_to_display1])

    st.write("Top 5 Negative Key Phrases")
    st.table(top_5_negative[columns_to_display1])

    st.write("Top 5 Positive Responses")
    st.table(top_5_positive[columns_to_display2])

    st.write("Top 5 Negative Responses")
    st.table(top_5_negative[columns_to_display2])

    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    Reasons that drive scores: 1 - Very Uncomfortable / 2 - Uncomfortable / 3 - Hesitant 
    </h2>
    """,
    unsafe_allow_html=True
    )
    
    q29_data = pd.DataFrame({'negative_reasons': filtered_data.iloc[:, 29]})
    q29_data = q29_data.explode('negative_reasons')
    q29_data.dropna(inplace=True)

    # Count the occurrences of each negative reason
    negative_reason_recruiting_counts = q29_data['negative_reasons'].value_counts().reset_index()
    negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']

    # Calculate percentage
    negative_reason_recruiting_counts['percentage'] = negative_reason_recruiting_counts['count'] / len(
        filtered_data) * 100

    # Create a horizontal bar chart with Plotly
    fig1 = px.bar(negative_reason_recruiting_counts, y='negative_reasons', x='percentage', text='count',
                  color='negative_reasons', color_discrete_sequence=['#FFA500'], orientation='h')

    # Customize the tooltip
    fig1.update_traces(hovertemplate='<b>Reason:</b> %{y}<br><b>Count:</b> %{text}')

    # Show the chart
    st.plotly_chart(fig1, use_container_width=False)

if dashboard == 'Section 4: Learning':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
        
    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    
    ### Question24: From 1 to 5, how satisfied are you with your current learning management system ?
    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q24ValuesCount, q24MedianScore = score_distribution(filtered_data, 31)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q24ValuesCount.values})

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Current Learning Management System</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q24MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown1 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown1')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 31)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Column 35: What could be improved or what kind of format is missing today ?
    st.title("Sentiment Analysis App")

    # Display the improvement/missing format for learning management system
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">the improvement/missing format for learning management system</h1>', unsafe_allow_html=True)

    #Extract key phrases from the text
    learning_stopwords = ["this","about", "of", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "help", "need", "everyone", "makes", "improved", "improvement", "missing", "format", "today", "no", "and","should","more", "training"]

    improvement_and_missing = filtered_data.iloc[:, 35]
    improvement_and_missing = improvement_and_missing.dropna()

    nltk.download('punkt', quiet=True)

    def extract_keyphrases(text):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 10), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=3)
        return ', '.join([word for word, _ in keywords])

    #extract keywords from the text
    improvement_and_missing_keywords = improvement_and_missing.apply(extract_keyphrases)


    # Function to extract bigrams from text
    def extract_bigrams(text):
        tokens = nltk.word_tokenize(text)
        bigrams = list(ngrams(tokens, 2))
        return [' '.join(bigram) for bigram in bigrams]

    # Concatenate all text data
    all_text = ' '.join(improvement_and_missing_keywords.astype(str))

    # Generate bigrams
    bigrams = extract_bigrams(all_text)

    # Count the frequency of each bigram
    bigram_freq = Counter(bigrams)

    # Generate the word cloud
    phrase_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=learning_stopwords).generate_from_frequencies(bigram_freq)

    # Display the word cloud using Streamlit
    st.title('Phrase Cloud')
    st.image(phrase_cloud.to_array(), use_column_width=True)

    st.write(improvement_and_missing_keywords)


    # Function to split and list phrases
    def list_phrases(dataframe):
        phrases = []
        for row in dataframe:
            if pd.notna(row):
                phrases.extend([phrase.strip() for phrase in row.split(',')])
        return phrases

    # List phrases in the DataFrame using the column index (0 in this case)
    phrases = list_phrases(improvement_and_missing_keywords)

    # Convert to DataFrame and sort by phrase length
    phrases_df = pd.DataFrame(phrases, columns=['Key Reasons']).sort_values(by='Key Reasons', key=lambda x: x.str.len())
    
    phrases_df = phrases_df[phrases_df['Key Reasons'].str.strip() != '']

    # Display the phrase counts without index
    st.write(phrases_df.reset_index(drop=True))

    

    
    #check the phrases
    st.table(improvement_and_missing_keywords.head(20))

    #list every keyphrase as a single one and count the frequency
    improvement_and_missing_keywords = improvement_and_missing_keywords.str.strip()
    #check the phrases
    st.table(improvement_and_missing_keywords.head(20))
    improvement_and_missing_keywords = improvement_and_missing_keywords[improvement_and_missing_keywords != '']
    #check the phrases
    st.table(improvement_and_missing_keywords.head(20))
    improvement_and_missing_keywords = improvement_and_missing_keywords.value_counts()
    #check the phrases
    st.table(improvement_and_missing_keywords.head(20))

    #display the frequency of the keywords
    st.write("Top 5 Keywords")
    st.table(improvement_and_missing_keywords.value_counts().head(15))
    



    #generate the word_cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=learning_stopwords, collocations=False).generate(' '.join(improvement_and_missing_keywords))

    #count frequency of each keyphrase
    improvement_and_missing_keywords = improvement_and_missing_keywords.str.strip()
    improvement_and_missing_keywords = improvement_and_missing_keywords[improvement_and_missing_keywords != '']
    improvement_and_missing_keywords = improvement_and_missing_keywords.value_counts()

    #display the frequency of the keywords
    st.write("Top 5 Keywords")
    st.table(improvement_and_missing_keywords.value_counts().head(15))








        






