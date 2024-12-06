import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from scipy.stats import ttest_ind
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import re
import seaborn as sns


# Download necessary NLTK data
import nltk
from nltk.data import find
try:
    # 檢查 'punkt' 是否已經存在
    find('tokenizers/punkt')
except LookupError:
    # 如果不存在則下載
    nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords')

# Load Data
nyt_data = pd.read_csv("NT.csv")  
wapo_data = pd.read_csv("WT.csv")   
wsj_data = pd.read_csv("WS.csv")   

data = pd.concat([
    nyt_data.assign(source="nyt"),
    wapo_data.assign(source="wapo"),
    wsj_data.assign(source="wsj")
])

# Define stopwords
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\W+', ' ', text.lower())  # Remove non-word characters and lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing to create processed_text
data['processed_text'] = data['text'].apply(preprocess_text)

##################################################################
#### Title #######################################################
##################################################################
st.title("Sentiment Analysis of Major US Newspapers on COVID-19 During Election Period")
st.caption("This page was created by Seizu HSU Yi-Ju")
#container = st.container(border=False)
#container.caption("This page was created by Seizu HSU Yi-Ju")

##################################################################
#### Intro #######################################################
##################################################################
st.header("Introduction")
st.write("""
In recent years, the political environment has become increasingly polarized, with stark 
                  divisions in ideology and values shaping public discourse. This polarization 
                  extends to the media landscape, where unsystematic and unverified 
                  information—whether from news outlets or social media platforms like 
                  Twitter—proliferates, contributing to widespread confusion and mistrust. 
                  In such a chaotic information environment, the framing and tone of news 
                  articles gain heightened significance, reflecting the values and mindsets 
                  of both the journalists and the publications they represent.
         
The description and tone of news articles are particularly crucial in times of crisis, such 
                  as during the COVID-19 pandemic. The emotional tone of coverage can influence 
                  public perception and response to governmental management of the crisis. In 
                  the United States, three of the most influential newspapers are The New York 
                  Times, The Washington Post, and The Wall Street Journal. Generally, it is 
                  perceived that The New York Times and The Washington Post lean towards the 
                  left, while The Wall Street Journal leans towards the right. This perception 
                  provides a foundation for examining whether there are significant differences 
                  in how these newspapers report on COVID-19 management during a politically 
                  charged period like an election.
         
Our research aims to investigate the heterogeneity in the emotional tone—positive, negative, 
                  or neutral—of COVID-19 coverage between these left-leaning and right-leaning 
                  newspapers. By focusing on the period from October 1st to 31st, 2020, we seek 
                  to understand if the political orientation of these publications results in 
                  differing emotional portrayals of the pandemic. This analysis will not only 
                  shed light on media bias but also contribute to broader discussions on the 
                  role of journalism in shaping public opinion during critical moments in history.
""")

##################################################################
#### Data Collection #############################################
##################################################################
st.divider()
st.header("Data Collection")
st.write("""
For this study, data was collected from the Protest Database, focusing specifically on articles 
         published during the critical time frame of October 1st to 31st, 2020. This period is 
         identified as the peak of the election campaign, characterized by heightened media 
         activity and public interest.

The scope of data collection was limited to articles publicized within the United 
         States, excluding any content from foreign offices. This geographic focus ensures 
         that the analysis is relevant to the domestic audience and reflects the media landscape 
         within the U.S.

To identify relevant articles, we employed a keyword filter that included terms most frequently 
         associated with the COVID-19 pandemic. The keywords used were "pandemics," "coronaviruses
         ," and "covid-19," which are ranked as the top three COVID-related terms. These keywords
         were applied to filter articles that specifically discussed the pandemic and its 
         management.

Under these conditions, we collected all available articles, including their full texts, from 
         three major newspapers: The New York Times, The Washington Post, and The Wall Street 
         Journal. These publications were chosen due to their significant influence and perceived 
         political leanings—The New York Times and The Washington Post as left-leaning, and The 
         Wall Street Journal as right-leaning. The comprehensive collection of articles from these 
         sources provides a robust dataset for analyzing the emotional tone in their coverage of 
         COVID-19 during the specified period. And then, the text of each article is tokenized, 
         splitting the into individual words, preparing for the further analysis.
""")

##################################################################
#### Methodology #################################################
##################################################################
st.divider()
st.header("Hypothesis and Methodology")
st.write("""
Since we want to investigate the difference of emotional tone among left-leaning and right-leaning 
         newspaper, we need to seperate three publication into two group first.
""")

container01 = st.container(border=True)
container01.write("""
**Left-leaning group: The New Yorks Times, The Washinton Post**
                  
**Right-leaning group: The Wall Street Journal**
""")

st.write("And, our hypothesis is:")

container02 = st.container(border=True)
container02.markdown("""
**Null Hypothesis ($ H_0 $)**: There is no significant difference in sentiment scores between 
                  left-leaning and right-leaning media publications. $ \mu_{left} = \mu_{right} $

**Alternative Hypothesis ($ H_1 $)**: There is a significant difference in sentiment scores 
                  between left-leaning and right-leaning media publications. $ \mu_{left} \\neq \mu_{right} $
""")

st.write("""
To test our hypothesis, we conduct following steps. First, we conduct the frequency analysis, compare the those words with highest frequency of each 
         publication, to see whether it has the difference or not intuitively. And then, we use the 
         sentiment analytical tools **nltk.sentiment** to evaluate the sentiment score of the text. 
         Compare the statistics of different publication, and do the t-test as well to check our 
         hypothesis.

For here, there are 2 lexicon be applied to the sentiment analysis:
""")      

container03 = st.container(border=True)
container03.write("""
**(1) VADER**

The VADER lexicon assesses the sentiment of text by categorizing words into positive, neutral, 
                  or negative categories and assigning sentiment intensity scores to each. Using 
                  this method, we calculate a compound sentiment score for each article. The 
                  compound score is a normalized metric ranging from -1 (most negative) to +1 
                  (most positive) and is used for further analysis.

**(2) AFINN**
                  
"Afinn" lexicon will contains words with sentiment scores ranging from -5 to 5. We calculate 
                  the summation value of each article as the sentiment score in this case. 
                  Prepare for the further analysis.
""")
# All bing related variables are refered to the VADER lexicon.

st.write("""
After catching the sentiment score, we plan to conduct the t-test to determine if there is a 
         significant difference in sentiment scores between left-leaning and right-leaning 
         publications for both Bing and AFINN lexicons. We expect the results of the t-test 
         can provide statistical evidence regarding the differences in emotional tone across 
         the different political leanings of the newspapers.
""")

##################################################################
#### Frequency ###################################################
##################################################################
st.divider()
st.header("Frequency Analysis")
st.write("""
Frequency analysis can give us an initial understanding of the textual data collected for this 
         study, we begin by examining the most frequently occurring words in the articles. This 
         preliminary analysis sometimes can helps to identify common themes and topics discussed 
         across different newspapers. Visualizing these frequent words through bar plots and word 
         clouds provides a clear and immediate impression of the textual content.

Hence, please see the following figure first.
""")

# Define a function to generate word cloud and top words
def get_top_words(text, n=20):
    from collections import Counter
    words = text.split()
    counter = Counter(words)
    return counter.most_common(n)

# Function to generate word cloud and top words side-by-side
# Function to generate word cloud and top words side-by-side with captions
# Function to generate word cloud and top words side-by-side with centered captions
def generate_figures(source, title):
    # Word Cloud
    text = ' '.join(data[data['source'] == source]['processed_text'])
    
    # Set seed for reproducibility and use an orange-red color scheme
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap='RdBu',  # Use orange-red color scheme
        random_state=42
    ).generate(text)

    # Top Words
    top_words = get_top_words(text)

    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)

    # Display Word Cloud in the first column
    with col1:
        fig_wc, ax_wc = plt.subplots(figsize=(8, 6))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
        st.markdown(
            f"<p style='text-align: center; font-size: 14px;'>Word Cloud - {title}</p>",
            unsafe_allow_html=True
        )  # Centered caption below the figure

    # Display Top Words in the second column
    with col2:
        top_words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
        fig_tw, ax_tw = plt.subplots(figsize=(8, 6))
        if source in ("nyt", "wapo"):
            color = 'darkslateblue'
        elif source == "wsj":
            color = 'firebrick'
        ax_tw.barh(
            top_words_df["Word"], 
            top_words_df["Frequency"], 
            color=mcolors.CSS4_COLORS[color]  # Use a specific orange-red color
        )
        ax_tw.set_xlabel("Frequency")
        ax_tw.set_ylabel("Word")
        ax_tw.invert_yaxis()  # Invert y-axis to show highest frequency at the top
        st.pyplot(fig_tw)
        st.markdown(
            f"<p style='text-align: center; font-size: 14px;'>Top 20 Words - {title}</p>",
            unsafe_allow_html=True
        )  # Centered caption below the figure

# Add a dropdown menu for publication selection
#st.header("Frequency Analysis")

# Define publications
publications = {
    "The New York Times": "nyt",
    "The Washington Post": "wapo",
    "The Wall Street Journal": "wsj"
}

# Dropdown for selection
selected_publication_name = st.selectbox(
    "Please choose a publication to:",
    options=list(publications.keys()),
    index=list(publications.keys()).index("The New York Times")  # Default to "The New York Times"
)

# Map the selected publication name to its corresponding source key
selected_publication = publications[selected_publication_name]

# Dynamically show content based on the selected publication
generate_figures(selected_publication, selected_publication_name)

st.write("""
Based on the frequency analysis results, we can find out that for the top 20 words of each 
         publication did not have huge difference. That is, unfortunately, if we only rely on 
         the descriptive statistics, there is no takeaway. Then, we move on to the sentiment 
         analysis.
""")

#####################################################################
##### Sentiment Analysis ############################################
#####################################################################
st.divider()
st.header("Sentiment Analysis")
st.write("""
By the two lexicon, the sentiment score of each text has been calculated, the distribution are 
         as the figure.
""")

#### Sentiment Analysis ##############################################
from nltk.sentiment import SentimentIntensityAnalyzer
from afinn import Afinn

# Download NLTK resources if not already done
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzers
sia = SentimentIntensityAnalyzer()  # VADER sentiment analyzer
afinn = Afinn()  # AFINN sentiment analyzer

# Function to calculate sentiment scores
def calculate_sentiment_scores(text):
    if not isinstance(text, str) or not text.strip():
        return 0, 0
    # Bing-like sentiment (VADER from NLTK)
    sia_scores = sia.polarity_scores(text)
    # AFINN sentiment
    afinn_score = afinn.score(text)
    return sia_scores['compound'], afinn_score

# Calculate sentiment scores and add to DataFrame
data['bing_score'], data['afinn_score'] = zip(*data['processed_text'].apply(calculate_sentiment_scores))

#### Sentiment Distributions ####
#st.subheader(":gray[Sentiment Distributions]")

def plot_sentiment_distribution(data, score_column, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define custom colors with transparency
    custom_palette = {
        'nyt': '#8c3537',   # Dark red 
        'wapo': '#f79779',  # Light orange 
        'wsj': '#526ca1'    # Navy blue
    }

    # Create the boxplot with a single color and transparency for each group
    for source, color in custom_palette.items():
        sns.boxplot(
            x='source', 
            y=score_column, 
            data=data[data['source'] == source], 
            palette=custom_palette,  
            width=0.6,        # Box width
            linewidth=1.5,    # Box line width
            ax=ax,
        )

    # Set titles and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Publication", fontsize=12)
    ax.set_ylabel("Sentiment Score", fontsize=12)
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    plot_sentiment_distribution(data, 'bing_score', "VADER Sentiment Distribution")

with col4:
    plot_sentiment_distribution(data, 'afinn_score', "AFINN Sentiment Distribution")

st.write("""
According to the figure, it shows:
         
1. The sentiment scores for left-leaning media are skewed more positively, with a higher median value.
2. Right-leaning media show a distribution leaning toward neutral or negative values, with a narrower range of data.
3. The distinct differences in distribution and further support the following T-test results, confirming significant sentiment differences between the two groups.

""")


# Plot Bing Sentiment Distribution
#plot_sentiment_distribution(data, 'bing_score', "Bing Sentiment Distribution")

# Plot AFINN Sentiment Distribution
#plot_sentiment_distribution(data, 'afinn_score', "AFINN Sentiment Distribution")


#sentiment_summary = data.groupby('source').agg({
#    'bing_score': ['mean', 'std'],
#    'afinn_score': ['mean', 'std']
#}).reset_index()
st.subheader("Summary of Sentiment Score")
#st.write(sentiment_summary)
summary = pd.read_csv("sentiment_summary.csv",
                      index_col=1)
summary = summary.drop(columns=["Unnamed: 0"])
st.write(summary)

#####################################################################
##### T-Test ########################################################
#####################################################################
#### T-Test Results ####
st.divider()
st.header("T-Test")
left_leaning = data[data['source'].isin(['nyt', 'wapo'])]
right_leaning = data[data['source'] == 'wsj']

# Perform T-tests
bing_ttest = ttest_ind(left_leaning['bing_score'], right_leaning['bing_score'])
afinn_ttest = ttest_ind(left_leaning['afinn_score'], right_leaning['afinn_score'])

container04 = st.container(border=True)
container04.write("**(1)VADER**")
container04.write(f"T-statistic: ** :red[{bing_ttest.statistic:.2f}]**, P-value: ** :red[{bing_ttest.pvalue:.4e}]**")
container04.write("")
container04.write("**(2)AFINN**")
container04.write(f"T-statistic: **{afinn_ttest.statistic:.2f}**, P-value: **{afinn_ttest.pvalue:.4e}**")

#st.write("_VADER Sentiment T-Test Results_")
#st.write(f"T-statistic: {bing_ttest.statistic:.2f}, P-value: {bing_ttest.pvalue:.4e}")

#st.write("_AFINN Sentiment T-Test Results_")
#st.write(f"T-statistic: {afinn_ttest.statistic:.2f}, P-value: {afinn_ttest.pvalue:.4e}")

# t-test conclusion
st.write("""
Both tests suggest **rejecting the null hypothesi**, meaning the data shows **significant 
         sentiment differences between right-leaning and left-leaning media**. The results 
         align with the alternative hypothesis, indicating the sentiment scores are not 
         equivalent across these two groups.
         """)
st.write("""
The negative T-statistic suggests the direction of the difference: left-leaning media 
         likely has higher sentiment scores (more positive) than right-leaning media 
         in this context.
         """)

#####################################################################
##### Conclusion ####################################################
#####################################################################
st.divider()
st.header("Conclusion")
st.write("""
The results of this updated analysis provide evidence that there are significant 
         differences in sentiment scores between left-leaning and right-leaning 
         media publications in their coverage of COVID-19-related issues during 
         the election period. Using both the VADER and AFINN sentiment analysis 
         lexicons, the T-test results indicate that the null hypothesis—that 
         there is no significant difference in sentiment scores—can be rejected 
         in favor of the alternative hypothesis.

The results strongly suggest that left-leaning media publications (The New York 
         Times and The Washington Post) exhibit significantly more positive 
         sentiment compared to the right-leaning publication (The Wall Street Journal).

The negative T-statistic values across both lexicons further indicate the direction of 
         the sentiment difference, with left-leaning media showing higher average sentiment scores. This implies a distinct heterogeneity in the emotional tone of coverage between these groups.
""")

st.subheader("_Possible Implications_")
st.write("""
The significant results may reflect inherent differences in editorial policies, framing, 
         and narrative focus influenced by the political orientation of these publications. 
         Left-leaning newspapers might employ a more optimistic or reassuring tone in their 
         pandemic reporting, whereas right-leaning newspapers might lean towards a more 
         critical or skeptical tone.

""")

st.subheader("_Limitation and Future Research_")
st.write("""
While this analysis successfully identifies significant sentiment differences, certain 
         limitations must be acknowledged:

1. The study is restricted to three major newspapers, which may not fully represent 
         the broader media landscape in the U.S.
2. The scope is limited to a single month (October 2020), potentially missing temporal 
         variations in media sentiment.
3. Although VADER and AFINN are robust tools, sentiment analysis is inherently constrained 
         by lexicon coverage and the nuances of language.

Future research could address these limitations by expanding the dataset to include a more diverse range of publications, extending the timeframe of analysis, and applying more sophisticated sentiment and natural language processing (NLP) techniques. Such approaches could provide deeper insights into the interaction between media sentiment, political leanings, and public discourse on significant events.

""")

#####################################################################
##### Note ##########################################################
#####################################################################
st.divider()
st.header(" :gray[Note]")
url = "https://drive.google.com/file/d/15XS-yBPqVYrYCVXi-sn9VMVbVEzk8Aqn/view?usp=share_link"
st.write(":gray[check out this [link](%s)]" % url)
st.write("""
:gray[Here is the same analysis done by R before, however, it showed the different insight than the page,
         which still need more study to figure out.]
""")
st.caption("Last update: 2024-12-06")