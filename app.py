import base64
import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
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
from nltk.util import ngrams as nltk_ngrams
import plotly.graph_objects as go


nltk.download('punkt', quiet=True)

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

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file")
st.session_state['uploaded_file'] = uploaded_file

# Load and clean data
@st.cache_data(persist=True)
def load_data():
    if st.session_state['uploaded_file'] is not None:
        data = pd.read_excel(st.session_state['uploaded_file'])
    else:
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
dashboard = st.sidebar.radio("Select Sentiment/Reason Analysis Dashboard", (
                                                  'Section 1: Employee Experience',
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

st.sidebar.markdown('For complete survey dashboard, please check out the [link](https://v1chatbotapp-2syadnkccp6nyevahkmkbm.streamlit.app)')


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
if dashboard == 'Section 1: Employee Experience':
    render_header("Employee Experience: General HR Services Evaluation")
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
    role_summary = role_summary.sort_values('Count', ascending=True)
    function_summary = pd.DataFrame(data['What function are you part of ?'].value_counts()).reset_index()
    function_summary.columns = ['Function', 'Count']
    function_summary = function_summary.sort_values('Count', ascending=True)
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

comfort_options = ['Select a comfort level', 'Very Uncomfortable', 'Uncomfortable', 'Hesitant',
                           'Comfortable', 'Very Comfortable']

def filter_by_comfort(data, comfort_level, column_index):
    if comfort_level != 'Select a comfort level':
        data = data[data.iloc[:, column_index] == comfort_options.index(comfort_level)]
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
    wordcloud_high_scores = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set,
                                      collocations=False).generate(text_high_scores)
    wordcloud_low_scores = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set,
                                     collocations=False).generate(text_low_scores)

    # Create columns for displaying the word clouds side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud for Low Scores</h3>",
            unsafe_allow_html=True)
        fig_low_scores, ax_low_scores = plt.subplots(figsize=(10, 5))
        ax_low_scores.imshow(wordcloud_low_scores, interpolation='bilinear')
        ax_low_scores.axis('off')
        st.pyplot(fig_low_scores)

    with col2:
        st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud for High Scores</h3>",
            unsafe_allow_html=True)
        fig_high_scores, ax_high_scores = plt.subplots(figsize=(10, 5))
        ax_high_scores.imshow(wordcloud_high_scores, interpolation='bilinear')
        ax_high_scores.axis('off')
        st.pyplot(fig_high_scores)

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b")

@st.cache_resource
def load_star_rating_model():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

@st.cache_resource
def load_keyword_extractor():
    return KeyBERT()

# Functions for prediction
def get_transformer_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

def predict_star_rating(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    star_rating = torch.argmax(probabilities, dim=1).item() + 1
    return star_rating

def extract_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=5)
    return ', '.join([word for word, _ in keywords])

# Load models
sentiment_analyzer = load_sentiment_analyzer()
tokenizer, model = load_star_rating_model()
kw_model = load_keyword_extractor()


############ SENTIMENT ANALYSIS FUNCTION ENDS ############



############ SECTION 1 STARTS ############
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


    # Question 4: What HR processes do you interact with the most in your day-to-day work ?
    q4_data = pd.DataFrame({
        'ID': filtered_data['ID'],
        'HR_Process': filtered_data['What HR processes do you interact with the most in your day-to-day work ?']
    })
    # Remove the last semicolon from each HR_Process value
    q4_data['HR_Process'] = q4_data['HR_Process'].str.rstrip(';')
    # Splitting the HR_Process values into separate lists of processes
    q4_data['HR_Process'] = q4_data['HR_Process'].str.split(';')
    # Explode the lists into separate rows while maintaining the corresponding ID
    q4_processed = q4_data.explode('HR_Process')
    # Reset index to maintain the original ID
    q4_processed.reset_index(drop=True, inplace=True)
    q4_count = q4_processed.groupby('HR_Process').size().reset_index(name='Count')

    # Question 5: In what areas do you think HR could improve its capabilities to enhance how they deliver services and support you ?
    q5_data = pd.DataFrame({
        'ID': filtered_data['ID'],
        'Improve_Area': filtered_data[
            'In what areas do you think HR could improve its capabilities to enhance how they deliver services and support you ?']
    })
    # Remove the last semicolon from each value
    q5_data['Improve_Area'] = q5_data['Improve_Area'].str.rstrip(';')
    # Splitting the values into separate lists of processes
    q5_data['Improve_Area'] = q5_data['Improve_Area'].str.split(';')
    # Explode the lists into separate rows while maintaining the corresponding ID
    q5_processed = q5_data.explode('Improve_Area')
    # Reset index to maintain the original ID
    q5_processed.reset_index(drop=True, inplace=True)
    q5_count = q5_processed.groupby('Improve_Area').size().reset_index(name='Count')

    # Question 4 and 5 combined
    # Merge the two dataset on function
    # Merge datasets by matching HR_Process and Improve_Area
    q4_q5_count = pd.merge(q4_count, q5_count, left_on='HR_Process', right_on='Improve_Area', how='outer')
    # Drop unnecessary columns
    q4_q5_count.drop(['Improve_Area'], axis=1, inplace=True)
    q4_q5_count.rename(
        columns={'HR_Process': 'HR Function', 'Count_x': 'HR_Process_Interacted', 'Count_y': 'Improvement_Areas'},
        inplace=True)
    q4_q5_count.sort_values('HR_Process_Interacted', ascending=False, inplace=True)
    # Separate 'None' row from the DataFrame
    none_row = q4_q5_count[q4_q5_count['HR Function'] == 'None']
    q4_q5_count = q4_q5_count[q4_q5_count['HR Function'] != 'None']

    # Sort 'HR_Process_Interacted' in descending order
    q4_q5_count.sort_values(by='HR_Process_Interacted', ascending=True, inplace=True)

    # Append 'None' row at the end
    q4_q5_count = pd.concat([none_row, q4_q5_count])
    # Reshape data into tidy format
    df_tidy = q4_q5_count.melt(id_vars='HR Function', var_name='Type', value_name='Count')

    # Question 7: How do you access HR Information ?
    q7_data = pd.DataFrame({'device': filtered_data["How do you access HR Information ?"]})
    q7_data['device'] = q7_data['device'].str.rstrip(';').str.split(';')
    q7_data = q7_data.explode('device')
    q7_data.dropna(inplace=True)
    # Count the occurrences of each device
    device_counts = q7_data['device'].value_counts().reset_index()
    device_counts.columns = ['device', 'count']
    # Calculate percentage
    device_counts['percentage'] = device_counts['count'] /len(q7_data) * 100

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
        q6ValuesCount, q6MedianScore = score_distribution(filtered_data, 11)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q6ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Overall Rating on HR Services and Support</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q6MedianScore:.1f}</div>"
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

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 11)

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

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q11ValuesCount, q11MedianScore = score_distribution(filtered_data, 13)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q11ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)        

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

    # Define colors for each device
    colors = {'Computer': '#440154', 'Mobile': '#5ec962', 'Tablet': '#3b528b'}

    # Set up space for two visualizations
    fig_q4_ratio = 0.65
    fig_q7_ratio = 1 - fig_q4_ratio
    q4_col, q7_col = st.columns([fig_q4_ratio, fig_q7_ratio])

    with q4_col:
        # Plot for HR Processes in the first column
        fig_q4 = go.Figure(data=[
            go.Bar(
                name='Improvement Areas',
                y=df_tidy[df_tidy['Type'] == 'Improvement_Areas']['HR Function'],
                # make it horizontal bar chart to show texts completely
                x=df_tidy[df_tidy['Type'] == 'Improvement_Areas']['Count'],
                # make it horizontal bar chart to show texts completely
                marker_color='#3b528b',
                orientation='h'  # make it horizontal bar chart to show texts completely
            ),
            go.Bar(
                name='Employee Interaction',
                y=df_tidy[df_tidy['Type'] == 'HR_Process_Interacted']['HR Function'],
                # make it horizontal bar chart to show texts completely
                x=df_tidy[df_tidy['Type'] == 'HR_Process_Interacted']['Count'],
                # make it horizontal bar chart to show texts completely
                marker_color='#5ec962',
                orientation='h'  # make it horizontal bar chart to show texts completely
            )
        ])
        fig_q4.update_layout(
            title='HR Processes: Employee Interaction vs Improvement Areas',
            title_font=dict(size=17, family="Arial", color='#333333'),
            xaxis_title='Number of Respondents',
            yaxis_title='HR Process',
            barmode='group',
            annotations=[
                dict(
                    xref='paper', yref='paper', x=0, y=1.1,
                    xanchor='left', yanchor='top',
                    text="<i>Each respondent is able to select more than one HR process</i>",
                    font=dict(family='Arial', size=12, color='#707070'),
                    showarrow=False)
            ],
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="top"
            ),
            margin=dict(l=22, r=20, t=70, b=70)
        )
        st.plotly_chart(fig_q4, use_container_width=True)

    # Plot for Device Usage in the second column
    with q7_col:
        fig_q7 = px.bar(device_counts, x='percentage', y='device', text='percentage', orientation='h', color='device',
                        color_discrete_map=colors)
        fig_q7.update_layout(
            title='Devices Used to Access HR Information',
            title_font=dict(size=17, family="Arial", color='#333333'),
            xaxis={'visible': False, 'showticklabels': False},
            yaxis_title=None,
            showlegend=False
        )
        fig_q7.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        st.plotly_chart(fig_q7, use_container_width=True)

    
    # Question 9: Which reason(s) drive that score ?

    # Display the reasons for communication channel satisfaction
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">The Reasons for Ratings on Communication Channels</h1>', unsafe_allow_html=True)

    # Define custom stopwords for the word clouds
    communication_stopwords = ["communication", "channels", "HR", "information", "important", "informed", "stay", "communicated", "employees", "company", "help", "communicates", "need", "everyone", "makes"]

    # Run this code in a Streamlit app
    if __name__ == "__main__":
        st.markdown("<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Word Cloud Visualization</h1>", unsafe_allow_html=True)
        generate_wordclouds(filtered_data, 13, 14, communication_stopwords)

    #################stay only in this dashboard start##################
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
    show_positive_key_phrases = st.checkbox("Show top 5 positive key phrases")
    if show_positive_key_phrases:
        st.table(top_5_positive[columns_to_display1])

    show_negative_key_phrases = st.checkbox("Show top 5 negative key phrases")
    if show_negative_key_phrases:
        st.table(top_5_negative[columns_to_display1])
    
    show_positive_responses = st.checkbox("Show Top 5 positive responses")
    if show_positive_responses:
        st.table(top_5_positive[columns_to_display2])
    
    show_negative_responses = st.checkbox("Show Top 5 negative responses")
    if show_negative_responses:
        st.table(top_5_negative[columns_to_display2])
    #################stay only in this dashboard end##################
    
############ SECTION 1 ENDS ############


############ SECTION 3 STARTS ############
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
        
        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)


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

    # Display the reasons for performance evaluation and feedback process satisfaction
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">The Reasons for Ratings on Performance Evaluation and Feedback Process</h1>', unsafe_allow_html=True)

    # Define custom stopwords for the word clouds
    performance_stopwords = ["performance", "evaluation", "feedback", "process", "talent", "employees", "company", "help", "need", "everyone", "makes"]

    # Run this code in a Streamlit app
    if __name__ == "__main__":
        st.markdown("<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Word Cloud Visualization</h1>", unsafe_allow_html=True)
        generate_wordclouds(filtered_data, 26, 27, performance_stopwords)

    #################stay only in this dashboard start##################
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
    show_positive_key_phrases = st.checkbox("Show top 5 positive key phrases")
    if show_positive_key_phrases:
        st.table(top_5_positive[columns_to_display1])
    
    show_negative_key_phrases = st.checkbox("Show top 5 negative key phrases")
    if show_negative_key_phrases:
        st.table(top_5_negative[columns_to_display1])
    
    show_positive_responses = st.checkbox("Show top 5 positive responses")
    if show_positive_responses:
        st.table(top_5_positive[columns_to_display2])

    show_negative_responses = st.checkbox("Show Top 5 negative responses")
    if show_negative_responses:
        st.table(top_5_negative[columns_to_display2])
    
    #################stay only in this dashboard end##################

    ### Question21: From 1 to 5, how comfortable do you feel discussing your career goals and development with your manager? 
    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Uncomfortable', 'Uncomfortable', 'Hesitant', 'Comfortable', 'Very Comfortable']
        q21ValuesCount, q21MedianScore = score_distribution(filtered_data, 28)

        ratings_df = pd.DataFrame({'Comfort Level': categories, 'Percentage': q21ValuesCount.values})
        
        #Define the order of the categories
        comfort_order = ['Very Comfortable', 'Comfortable', 'Hesitant', 'Uncomfortable', 'Very Uncomfortable']

        # Convert 'Comfort Level' to a categorical variable with the specified order
        ratings_df['Comfort Level'] = pd.Categorical(ratings_df['Comfort Level'], categories=comfort_order, ordered=True)

        # Sort the DataFrame by 'Comfort Level'
        ratings_df.sort_values('Comfort Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Comfort Level in Discussing Career Goals         and Development with Manager</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median comfort score is                   {q21MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Comfort Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Comfort Level', color_discrete_map={
                'Very Uncomfortable': '#440154',  # Dark purple
                'Uncomfortable': '#3b528b',  # Dark blue
                'Hesitant': '#21918c',  # Cyan
                'Comfortable': '#5ec962',  # Light green
                'Very Comfortable': '#fde725'  # Bright yellow
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
        comfort_options = ['Select a comfort level', 'Very Uncomfortable', 'Uncomfortable', 'Hesitant',
                           'Comfortable', 'Very Comfortable']
        comfort_dropdown1 = st.selectbox('', comfort_options,
                                         key='comfort_dropdown1')

        comfort_filtered_data1 = filter_by_comfort(filtered_data, comfort_dropdown1, 28)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(comfort_filtered_data1)
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

    fig1 = px.bar(negative_reason_recruiting_counts, y='negative_reasons', x='percentage', text='percentage',
                  color='negative_reasons', color_discrete_sequence=['#3b528b'], orientation='h')

    fig1.update_traces(hovertemplate='<b>Reason:</b> %{y}<br><b>Count:</b> %{text}')

    # Set the y-axis title
    fig1.update_yaxes(title_text='Reasons for Discomfort')

    # Remove the legend
    fig1.update_layout(showlegend=False)

    # Show the chart
    st.plotly_chart(fig1, use_container_width=False)

############ SECTION 3 ENDS ############

############ SECTION 4 STARTS ############
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

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)


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

    # Display the improvement/missing format for learning management system
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">The Improvement/Missing Format for Learning Management System</h1>', unsafe_allow_html=True)

    #Extract key phrases from the text
    learning_stopwords = ["this","about", "of", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "help", "need", "everyone", "makes", "improved", "improvement", "missing", "format", "today", "no", "and","should","more", "training"]

    improvement_and_missing = filtered_data.iloc[:, 35]
    improvement_and_missing = improvement_and_missing.dropna()

    #generate text for simple word cloud
    improvement_and_missing_text = ' '.join(improvement_and_missing.astype(str))

    #generate word cloud
    improvement_and_missing_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=learning_stopwords).generate(improvement_and_missing_text)

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(improvement_and_missing_cloud.to_array(), use_column_width=True)

    #################stay only in this dashboard start##################
    def extract_keyphrases(text):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 10), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=3)
        return ', '.join([word for word, _ in keywords])

    #extract keywords from the text
    improvement_and_missing_keywords = improvement_and_missing.apply(extract_keyphrases)


    # Function to extract bigrams from text
    def extract_bigrams(text):
        tokens = nltk.word_tokenize(text)
        bigrams = list(nltk_ngrams(tokens, 2))
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
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Phrase Cloud</h3>",
        unsafe_allow_html=True)
    st.image(phrase_cloud.to_array(), use_column_width=True)

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
    phrases_df = pd.DataFrame(phrases, columns=['Improvement/Missing']).sort_values(by='Improvement/Missing', key=lambda x: x.str.len())
    
    phrases_df = phrases_df[phrases_df['Improvement/Missing'].str.strip() != '']

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display complete improvement/missing dataFrame mentioned'):
        # Convert DataFrame to HTML and display it
        html = phrases_df.to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = phrases_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="improvement_missing.csv">Download improvement_missing csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
    #################stay only in this dashboard end##################

############ SECTION 4 ENDS ############

############ SECTION 5 STARTS ############
if dashboard == 'Section 5: Compensation':
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
    
    ### Qustion29: Do you participate in the Compensation Campaign ?
    q29_data_available_count = (filtered_data.iloc[:, 36] == 'Yes').sum()
    q29_data_available_pct = q29_data_available_count / len(filtered_data) * 100
   
    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    Compensation Campaign Participation
    </h2>
    """,
    unsafe_allow_html=True
    )

    st.write(
        f"{q29_data_available_pct:.2f}% of the respondents, {q29_data_available_count} employee(s), participated in the   compensation campaign.")
    
    ### Qustion30: Do you think that the data available in the Compensation form enables you to make a fair decision regarding a promotion, a bonus or a raise ? (e.g : compa-ratio, variation between years, historical data on salary and bonus, …) 
    q30_data_available_count = (filtered_data.iloc[:, 37] == 'Yes').sum()
    q30_data_available_pct = q30_data_available_count / q29_data_available_count * 100
    
    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    Data availability in the Compensation Form
    </h2>
    """,
    unsafe_allow_html=True
    )
    
    st.write(
        f"Among the people who participate the Compensation Campaign, {q30_data_available_pct:.2f}% of the respondents, {q30_data_available_count} employee(s), think that the data available in the Compensation form enables him/her to make a fair decision regarding a promotion, a bonus or a raise.")
    
    ### Question33: How would you rate the overall satisfaction regarding the compensation campaign ?
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
        q33ValuesCount, q33MedianScore = score_distribution(filtered_data, 40)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q33ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)


        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Compensation Campaign</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q33MedianScore:.1f}</div>"
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

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 40)

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
    

    ### Qustion31: What data is missing according to you ?


    # Display the Data Missing for Compensation
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">Data Missing for Compensation</h1>', unsafe_allow_html=True)

    #stopwords for data missing for compensation
    compensation_stopwords = ["compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    data_missing = filtered_data.iloc[:, 38]
    data_missing = data_missing.dropna()

    #generate all text for word cloud for data missing
    data_missing_text = ' '.join(data_missing.astype(str))
    
    #generate simple word cloud for data missing
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=compensation_stopwords).generate(data_missing_text)

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(wordcloud.to_array(), use_column_width=True)
    
    #################stay only in this dashboard start##################
    # Function to extract keyphrases from the text
    def extract_keyphrases(text):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 6), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=2)
        return ', '.join([word for word, _ in keywords])

    #extract keywords from the text
    data_missing_keywords = data_missing.apply(extract_keyphrases)


    # Function to extract bigrams from text
    def extract_bigrams(text):
        tokens = nltk.word_tokenize(text)
        bigrams = list(nltk_ngrams(tokens, 2))
        return [' '.join(bigram) for bigram in bigrams]

    # Concatenate all text data
    all_text = ' '.join(data_missing_keywords.astype(str))

    # Generate bigrams
    bigrams = extract_bigrams(all_text)

    # Count the frequency of each bigram
    bigram_freq = Counter(bigrams)

    # Generate the word cloud
    phrase_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=compensation_stopwords).generate_from_frequencies(bigram_freq)

    # Display the word cloud using Streamlit
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Phrase Cloud</h3>",
        unsafe_allow_html=True)
    
    st.image(phrase_cloud.to_array(), use_column_width=True)

    # Function to split and list phrases
    def list_phrases(dataframe):
        phrases = []
        for row in dataframe:
            if pd.notna(row):
                phrases.extend([phrase.strip() for phrase in row.split(',')])
        return phrases

    # List phrases in the DataFrame using the column index (0 in this case)
    phrases = list_phrases(data_missing_keywords)

    # Convert to DataFrame and sort by phrase length
    phrases_df = pd.DataFrame(phrases, columns=['Data Missing']).sort_values(by='Data Missing', key=lambda x: x.str.len())
    phrases_df = phrases_df[phrases_df['Data Missing'].str.strip() != '']

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display data missing'):
        # Convert DataFrame to HTML and display it
        html = phrases_df.to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = phrases_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data_missing.csv">Download data_missing csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
    #################stay only in this dashboard end##################

############ SECTION 5 ENDS ############

############ SECTION 6 STARTS ############
if dashboard == 'Section 6: Payroll':
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
    
    
    ### Question41: Are you part of the payroll team ?
    q41_data_available_count = (filtered_data.iloc[:, 48] == 'Yes').sum()
    q41_data_available_pct = q41_data_available_count / len(filtered_data) * 100
   
    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    Payroll Team
    </h2>
    """,
    unsafe_allow_html=True
    )

    st.write(
        f"{q41_data_available_pct:.2f}% of the respondents, {q41_data_available_count} employee(s), are part of the payroll team.")
    
    
    ### Question42: How satisfied are you with your current payroll system ?
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
        q42ValuesCount, q42MedianScore = score_distribution(filtered_data, 49)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q42ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)


        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Current Payroll System</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q42MedianScore:.1f}</div>"
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
        satisfaction_dropdown38 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown38')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown38, 49)

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

    ### Question53: What specific features of the current system do you like/that made you choose it ?

    # Display the specific features of the current system that you like/that made you choose it
    st.markdown('<h2 style="font-size:17px;font-family:Arial;color:#333333;">Specific Features of the Current System that People like/that made People choose it</h2>', unsafe_allow_html=True)

    #stopwords for specific features of the current system that you like/that made you choose it
    features_stopwords = ["payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    specific_features = filtered_data.iloc[:, 53]

    #generate wordcloud since the repsonses are too few
    word_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=features_stopwords).generate(' '.join(specific_features.dropna().astype(str)))

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    
    st.image(word_cloud.to_array(), use_column_width=True)
    
    #Generate more complex wordcloud if there are more repsonses
    specific_features = specific_features.dropna()

    #################stay only in this dashboard start##################
    def extract_keyphrases(text):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 4), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=2)
        return ', '.join([word for word, _ in keywords])

    #extract keywords from the text
    specific_features_keywords = specific_features.apply(extract_keyphrases)


    # Function to extract bigrams from text
    def extract_bigrams(text):
        tokens = nltk.word_tokenize(text)
        bigrams = list(nltk_ngrams(tokens, 2))
        return [' '.join(bigram) for bigram in bigrams]

    # Concatenate all text data
    all_text = ' '.join(specific_features_keywords.astype(str))

    # Generate bigrams
    bigrams = extract_bigrams(all_text)

    # Count the frequency of each bigram
    bigram_freq = Counter(bigrams)

    # Generate the word cloud
    phrase_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=features_stopwords).generate_from_frequencies(bigram_freq)

    # Display the word cloud using Streamlit
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Phrase Cloud</h3>",
        unsafe_allow_html=True)
    st.image(phrase_cloud.to_array(), use_column_width=True)


    # Function to split and list phrases
    def list_phrases(dataframe):
        phrases = []
        for row in dataframe:
            if pd.notna(row):
                phrases.extend([phrase.strip() for phrase in row.split(',')])
        return phrases

    # List phrases in the DataFrame using the column index (0 in this case)
    phrases = list_phrases(specific_features_keywords)

    # Convert to DataFrame and sort by phrase length
    phrases_df = pd.DataFrame(phrases, columns=['Key Features']).sort_values(by='Key Features', key=lambda x: x.str.len())
    phrases_df = phrases_df[phrases_df['Key Features'].str.strip() != '']

    #################stay only in this dashboard end##################
    
    st.markdown(
        "<h3 style='text-align: left; font-size: 20px; font-weight: normal;'>All specific features of the current system that people like/that made people choose it</h3>",
        unsafe_allow_html=True)
    st.write(specific_features)

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display complete specific features of the current system that people like/that made people choose it'):
        # Convert DataFrame to HTML and display it
        html = specific_features.to_frame().to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = filtered_data.iloc[:, 53].to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="specific_features.csv">Download specific_features CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

############ SECTION 6 ENDS ############

############ SECTION 7 STARTS ############
if dashboard == "Section 7: Time Management":
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
    
    
    ### Question52: Are you part of the Time Management Team ?
    q52_data_available_count = (filtered_data.iloc[:, 59] == 'Yes').sum()
    q52_data_available_pct = q52_data_available_count / len(filtered_data) * 100
   
    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    Time Management Team
    </h2>
    """,
    unsafe_allow_html=True
    )

    st.write(
        f"{q52_data_available_pct:.2f}% of the respondents, {q52_data_available_count} employee(s), are part of the time management team.")
    
    
    ### Question53: Do you currently have a time management system ?
    q53_data_available_count = (filtered_data.iloc[:, 60] == 'Yes').sum()
    q53_data_available_pct = q53_data_available_count / q52_data_available_count * 100
   
    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    Time Management System
    </h2>
    """,
    unsafe_allow_html=True
    )

    st.write(
        f"Among the people who are part of the time management team, {q53_data_available_pct:.2f}% of the respondents,  {q53_data_available_count} employee(s),  answer that they currently have a time management system.")
    
    
    ### Question54: How satisfied are you with your current time management system ?
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
        q54ValuesCount, q54MedianScore = score_distribution(filtered_data, 61)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q54ValuesCount.values})
        
        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)


        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Current Time Management System</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q54MedianScore:.1f}</div>"
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
        satisfaction_dropdown38 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown38')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown38, 61)

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

    #Column 66: According to you, what functionalities are missing from your current system ?
    # Display the functionalities missing from the current system
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">Functionalities missing from the current system</h1>', unsafe_allow_html=True)

    #stopwords for functionalities missing from the current system
    functionalities_stopwords = ["functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    functionalities_missing = filtered_data.iloc[:, 66]

    #generate wordcloud since the repsonses are too few
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(functionalities_missing.dropna().astype(str)))

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(word_cloud.to_array(), use_column_width=True)
    
    #Generate more complex wordcloud if there are more repsonses
    #drop missing values first
    functionalities_missing = functionalities_missing.dropna()

    #################stay only in this dashboard start##################
    def extract_keyphrases(text):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 4), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=1)
        return ', '.join([word for word, _ in keywords])

    #extract keywords from the text
    functionalities_missing_keywords = functionalities_missing.apply(extract_keyphrases)


    # Function to extract bigrams from text
    def extract_bigrams(text):
        tokens = nltk.word_tokenize(text)
        bigrams = list(nltk_ngrams(tokens, 2))
        return [' '.join(bigram) for bigram in bigrams]

    # Concatenate all text data
    all_text = ' '.join(functionalities_missing_keywords.astype(str))

    # Generate bigrams
    bigrams = extract_bigrams(all_text)

    # Count the frequency of each bigram
    bigram_freq = Counter(bigrams)

    # Generate the word cloud
    phrase_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=functionalities_stopwords).generate_from_frequencies(bigram_freq)

    # Display the word cloud using Streamlit
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Phrase Cloud</h3>",
        unsafe_allow_html=True)
    st.image(phrase_cloud.to_array(), use_column_width=True)

    # Function to split and list phrases
    def list_phrases(dataframe):
        phrases = []
        for row in dataframe:
            if pd.notna(row):
                phrases.extend([phrase.strip() for phrase in row.split(',')])
        return phrases

    # List phrases in the DataFrame using the column index (0 in this case)
    phrases = list_phrases(functionalities_missing_keywords)

    # Convert to DataFrame and sort by phrase length
    phrases_df = pd.DataFrame(phrases, columns=['functionalities missing']).sort_values(by='functionalities missing', key=lambda x: x.str.len())
    phrases_df = phrases_df[phrases_df['functionalities missing'].str.strip() != '']

    #################stay only in this dashboard end##################

    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>All the functionalities missing from the current system:</h3>",
        unsafe_allow_html=True)
    st.write(functionalities_missing)

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display complete missing functionalities'):
        # Convert DataFrame to HTML and display it
        html = functionalities_missing.to_frame().to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = filtered_data.iloc[:, 66].to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="functionalities_missing.csv">Download functionalities_missing csv file</a>'
    st.markdown(href, unsafe_allow_html=True)

############ SECTION 7 ENDS ############

############ SECTION 8 STARTS ############
if dashboard == "Section 8: User Experience":
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
    
    
    ### Question62: In the context of your job, what are the most valuable activities your current HRIS enable you to do?
    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    In the context of the job, what are the most valuable activities the current HRIS enable them to do?
    </h2>
    """,
    unsafe_allow_html=True
    )


    # Display the most valuable activities in the current HRIS
    st.markdown(
            "<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Most Valuable Activities in the Current HRIS</h1>",
            unsafe_allow_html=True)

    #stopwords for most valuable activities in the current HRIS
    HRIS_stopwords = ["I","and" ," and", "and ", " ;", "; ", " to", " for", "for ", "for", "to ", "my", "activities", "HRIS", "valuable", "system", "HR", "current", "functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    valuable_activities = filtered_data.iloc[:, 69]

    #drop missing values first
    valuable_activities = valuable_activities.dropna()

    #generate wordcloud since the repsonses are too few
    word_cloud_valuable = WordCloud(width=800, height=400, background_color='white', stopwords=HRIS_stopwords).generate(' '.join(valuable_activities.dropna().astype(str)))


    
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    
    st.image(word_cloud_valuable.to_array(), use_column_width=True)
    
    #################stay only in this dashboard start##################
    def extract_keyphrases(text):
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(2, 4), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=1)
        return ', '.join([word for word, _ in keywords])

    #extract keywords from the text
    valuable_activities_keywords = valuable_activities.apply(extract_keyphrases)


    # Function to extract bigrams from text
    def extract_bigrams(text):
        tokens = nltk.word_tokenize(text)
        bigrams = list(nltk_ngrams(tokens, 2))
        return [' '.join(bigram) for bigram in bigrams]

    # Concatenate all text data
    valuable_text = ' '.join(valuable_activities_keywords.astype(str))

    # Generate bigrams
    bigrams_valuable = extract_bigrams(valuable_text)

    # Count the frequency of each bigram
    bigram_freq_valuable = Counter(bigrams_valuable)

    # Generate the Phrase cloud
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Phrase Cloud</h3>",
            unsafe_allow_html=True)
    
    phrase_cloud_valuable = WordCloud(width=800, height=400, background_color='white', stopwords=HRIS_stopwords).generate_from_frequencies(bigram_freq_valuable)

    st.image(phrase_cloud_valuable.to_array(), use_column_width=True)

    
    # Function to split and list phrases
    def list_phrases(dataframe):
        phrases = []
        for row in dataframe:
            if pd.notna(row):
                phrases.extend([phrase.strip() for phrase in row.split(',')])
        return phrases

    # List phrases in the DataFrame using the column index (0 in this case)
    phrases_valuable = list_phrases(valuable_activities_keywords)

    # Convert to DataFrame and sort by phrase length
    phrases_valuable_df = pd.DataFrame(phrases_valuable, columns=['Key Activities']).sort_values(by='Key Activities', key=lambda x: x.str.len())
    phrases_valuable_df = phrases_valuable_df[phrases_valuable_df['Key Activities'].str.strip() != '']

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display all key phrases for most valuable activities in the current HRIS'):
        # Convert DataFrame to HTML and display it
        html = phrases_valuable_df.to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv_valuable = phrases_valuable_df.to_csv(index=False)
    b64_valauable = base64.b64encode(csv_valuable.encode()).decode()  # some strings <-> bytes conversions necessary here
    href_valuable = f'<a href="data:file/csv;base64,{b64_valauable}" download="most_valuable_activities.csv">Download most_valuable_activities csv file</a>'
    st.markdown(href_valuable, unsafe_allow_html=True)
    #################stay only in this dashboard end##################

    ### Question63: In the context of your job, what do your current HRIS fail to address?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        In the context of the job, what does the current HRIS fail to address?
        </h2>
        """,
        unsafe_allow_html=True
    )

    #stopwords for most functions are missing in the current HRIS
    HRIS_stopwords2 = ["I", "my", "activities", "fail", "address", "missing", "HRIS", "valuable", "system", "HR", "current", "functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]

    functions_missing = filtered_data.iloc[:, 70]

    #generate wordcloud since the repsonses are too few
    word_cloud_functions = WordCloud(width=800, height=400, background_color='white', stopwords=HRIS_stopwords2).generate(' '.join(functions_missing.dropna().astype(str)))

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(word_cloud_functions.to_array(), use_column_width=True)

    #Generate more complex wordcloud if there are more repsonses

    #drop missing values first
    functions_missing = functions_missing.dropna()

    #################stay only in this dashboard start##################
    #extract keywords from the text
    functions_missing_keywords = functions_missing.apply(extract_keyphrases)

    # Concatenate all text data
    functions_text = ' '.join(functions_missing_keywords.astype(str))

    # Generate bigrams
    bigrams_functions = extract_bigrams(functions_text)

    # Count the frequency of each bigram
    bigram_freq_functions = Counter(bigrams_functions)

    # Generate the word cloud
    phrase_cloud_functions = WordCloud(width=800, height=400, background_color='white', stopwords=HRIS_stopwords2).generate_from_frequencies(bigram_freq_functions)

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Phrase Cloud</h3>",
            unsafe_allow_html=True)
    st.image(phrase_cloud_functions.to_array(), use_column_width=True)

    # Function to split and list phrases
    phrases_functions = list_phrases(functions_missing_keywords)

    # Convert to DataFrame and sort by phrase length
    phrases_functions_df = pd.DataFrame(phrases_functions, columns=['Issues to Be Resovled']).sort_values(by='Issues to Be Resovled', key=lambda x: x.str.len())
    phrases_functions_df = phrases_functions_df[phrases_functions_df['Issues to Be Resovled'].str.strip() != '']

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display all key phrases for issues in the current HRIS'):
        # Convert DataFrame to HTML and display it
        html = phrases_functions_df.to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)
    
    # Convert DataFrame to CSV and generate download link
    csv_functions = phrases_functions_df.to_csv(index=False)
    b64_functions = base64.b64encode(csv_functions.encode()).decode()  # some strings <-> bytes conversions necessary here
    href_functions = f'<a href="data:file/csv;base64,{b64_functions}" download="issues_in_HRIS.csv">Download issues_in_HRIS csv file</a>'
    st.markdown(href_functions, unsafe_allow_html=True)
    #################stay only in this dashboard end##################

    #Column 72: In 3 words, how would you describe your experience with the current HRIS?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        In 3 words, how would people describe the current user-experience with the HRIS ?
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Descriptions about User Experience with the Current HRIS</h1>", unsafe_allow_html=True)
    #get the data
    overall_experience = filtered_data.iloc[:, 72]

    #set the stopwords
    Overall_stopwords = [",", ";", " ;", "; ", "not very", "not", "very", " ; ", "I", "my", "activities", "fail", "address", "missing", "HRIS", "valuable", "system", "HR", "current", "functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]

    # Function to extract n-grams from text
    def extract_ngrams(x, n):
        ngrams = []
        phrases = x.split(', ')
        for phrase in phrases:
            words = phrase.split(' ')
            ngrams.extend([' '.join(ng) for ng in nltk_ngrams(words, n)])
        return ngrams

    #drop missing values first
    overall_experience = overall_experience.dropna()

    # Concatenate all text data
    overall_text = ' '.join(overall_experience.astype(str))

    # Generate unigrams, bigrams, and trigrams
    unigrams_overall = extract_ngrams(overall_text, 1)
    bigrams_overall = extract_ngrams(overall_text, 2)
    trigrams_overall = extract_ngrams(overall_text, 3)

    # Count the frequency of each n-gram
    unigram_freq_overall = Counter(unigrams_overall)
    bigram_freq_overall = Counter(bigrams_overall)
    trigram_freq_overall = Counter(trigrams_overall)

    # Combine the frequencies
    combined_freq_overall = unigram_freq_overall + bigram_freq_overall + trigram_freq_overall

    # Generate the word cloud
    phrase_cloud_overall = WordCloud(width=800, height=400, background_color='white', stopwords = Overall_stopwords).generate_from_frequencies(combined_freq_overall)

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(phrase_cloud_overall.to_array(), use_column_width=True)

    #################stay only in this dashboard start##################
    #sentiment analysis for overall experience with the current HRIS

    # Apply the function to the 72nd column
    filtered_data['star_rating'] = filtered_data.iloc[:, 72].apply(lambda text: predict_star_rating(text, tokenizer, model))
    
    #count the number of 1-5 star rating predictions
    sentiment_count = filtered_data['star_rating'].value_counts()

    # Display the sentiment count   
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
        c73ValuesCount, c73MedianScore = score_distribution(filtered_data, 73)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': c73ValuesCount.values})
        
        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)



        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Star Rating Prediction on User Experience</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {c73MedianScore:.1f}</div>"
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
        st.plotly_chart(fig, use_container_width=True, key="rating_on_user_experience_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_dropdown2 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown2')

        satisfaction_filtered_data2 = filter_by_satisfaction(filtered_data, satisfaction_dropdown2, 73)

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
    
    #################stay only in this dashboard end##################

    #check if the user wants to see the data
    if st.checkbox('Display complete description data in 3 words'):
        # Convert DataFrame to HTML and display it
        html = filtered_data.iloc[:, 72].to_frame().to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = filtered_data.iloc[:, 72].to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="user_experience.csv">Download user_experience csv file</a>'
    st.markdown(href, unsafe_allow_html=True)















