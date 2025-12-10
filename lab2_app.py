import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

@st.cache_data
def load_data():
    """Load and cache the wine data"""
    try:
        df = pd.read_csv("lab2/wine_data_lab2.csv", skiprows=1)
        return df
    except FileNotFoundError:
        st.error("✗ Eroare: Fișierul 'lab2/wine_data_lab2.csv' nu a fost găsit")
        st.stop()
    except Exception as e:
        st.error(f"✗ Eroare la încărcare: {e}")
        st.stop()

@st.cache_data
def clean_data(df):
    """Clean the wine data"""
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate()
    df = df.drop_duplicates()
    
    df['category'] = df["category"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].replace({"rose": "rosé", "roze": "rosé"})
    
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace({
            'NaN': np.nan, 'None': np.nan, 'null': np.nan,
            "n/a": np.nan, '': np.nan
        })
    
    df['performance'] = df['points'] / df['price']
    
    return df

@st.cache_data
def process_text_data(df):
    """Process text descriptions and extract top words"""
    all_words = []
    for text in df["description"].astype(str):
        cleaned_word = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        all_words.extend(cleaned_word.split())
    
    stop_words = ['and', 'the', 'a', 'of', 'with', 'this', 'is', 'in',
                  'to', 'it', 'on', 'that', 'but', 'its', 'from',
                  'are', 'has', 'for', 'by', 'an', 'as', 'at',
                  'very', 'some', 'more', 'not', 'while', 'well',
                  'now.', 'or', 'be', 'all', 'into', 'bit', 'just',
                  'like', 'up', 'also', 'made', 'out']
    
    filtered_words = [w for w in all_words if w not in stop_words]
    word_frequency = Counter(filtered_words)
    top_words = word_frequency.most_common(10)
    
    return filtered_words, top_words

def search_wines(df, search_text):
    """Search wines by description text"""
    if not search_text:
        return df
    search_terms = search_text.lower().split()
    mask = df['description'].str.lower().str.contains('|'.join(search_terms), na=False)
    return df[mask]

st.set_page_config(page_title="Wine Data Analysis", layout="wide")
st.title("🍷 Wine Data Analysis Dashboard")

df = load_data()
df = clean_data(df)
filtered_words, top_words = process_text_data(df)

st.sidebar.header("Filtre")

categories = ['Toate'] + sorted(df['category'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Categoria vinului", categories)

countries = ['Toate'] + sorted(df['country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Țara", countries)

min_price, max_price = float(df['price'].min()), float(df['price'].max())
price_range = st.sidebar.slider(
    "Interval de preț",
    min_price, max_price,
    (min_price, max_price)
)

min_points, max_points = float(df['points'].min()), float(df['points'].max())
points_range = st.sidebar.slider(
    "Interval de puncte",
    min_points, max_points,
    (min_points, max_points)
)

min_perf, max_perf = float(df['performance'].min()), float(df['performance'].max())
performance_range = st.sidebar.slider(
    "Raport preț/calitate (performance)",
    min_perf, max_perf,
    (min_perf, max_perf)
)

search_text = st.sidebar.text_input("Caută în descriere", "")

filtered_df = df.copy()
if selected_category != 'Toate':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]
if selected_country != 'Toate':
    filtered_df = filtered_df[filtered_df['country'] == selected_country]

filtered_df = filtered_df[
    (filtered_df['price'] >= price_range[0]) &
    (filtered_df['price'] <= price_range[1]) &
    (filtered_df['points'] >= points_range[0]) &
    (filtered_df['points'] <= points_range[1]) &
    (filtered_df['performance'] >= performance_range[0]) &
    (filtered_df['performance'] <= performance_range[1])
]

if search_text:
    filtered_df = search_wines(filtered_df, search_text)

st.sidebar.metric("Vinuri găsite", len(filtered_df))

tab1, tab2, tab3, tab4 = st.tabs(["📊 Vizualizări", "🔍 Date Filtrate", "☁️ Word Cloud", "📈 Analize"])

with tab1:
    st.header("Vizualizări")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuția punctelor")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.hist(filtered_df['points'], bins=20, color="skyblue", edgecolor="black")
        ax1.set_xlabel("Puncte")
        ax1.set_ylabel("Număr de vinuri")
        ax1.set_title("Distribuția punctelor")
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Preț mediu pe țară (Top 15)")
        df_country = filtered_df.groupby("country")["price"].mean().sort_values(ascending=False).head(15)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(range(len(df_country)), df_country.values, color='coral')
        ax2.set_xticks(range(len(df_country)))
        ax2.set_xticklabels(df_country.index, rotation=45, ha='right')
        ax2.set_xlabel("Țara")
        ax2.set_ylabel("Preț mediu")
        ax2.set_title("Preț mediu pe țară")
        plt.tight_layout()
        st.pyplot(fig2)
    
    st.subheader("Distribuția categoriilor pe regiuni (Top 20 regiuni)")
    df_group = filtered_df.groupby(["region_1", "category"]).size().reset_index(name="count")
    top_regions = df_group.groupby("region_1")["count"].sum().nlargest(20).index
    df_top = df_group[df_group["region_1"].isin(top_regions)]
    
    if len(df_top) > 0:
        pivot = df_top.pivot_table(index="region_1", columns="category", values="count", fill_value=0)
        
        category_colors = {
            "dessert": "#FF61D0", "fortified": "#4B0082",
            "orange": "#FF9900", "port/sherry": "#F10303FF",
            "red": "#8B0000", "sparkling": "#FBE04AFF",
            "rosé": "#FF0080", "white": "#F5DEB3",
        }
        
        colors = [category_colors.get(c, '#888888') for c in pivot.columns]
        
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        pivot.plot(kind="bar", stacked=True, ax=ax3, color=colors)
        ax3.set_title("Distribuția categoriilor pe regiuni")
        ax3.set_xlabel("Regiune")
        ax3.set_ylabel("Număr de vinuri")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("Nu există date pentru această combinație de filtre.")

with tab2:
    st.header("Date Filtrate")
    st.dataframe(
        filtered_df[['title', 'country', 'category', 'variety', 'points', 'price', 'performance', 'description']].head(100),
        use_container_width=True
    )
    
    st.download_button(
        label="Download CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_wines.csv',
        mime='text/csv',
    )

with tab3:
    st.header("Word Cloud din Descrieri")
    
    if len(filtered_df) > 0:
        desc_words = []
        for text in filtered_df["description"].astype(str):
            cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            desc_words.extend(cleaned.split())
        
        stop_words = ['and', 'the', 'a', 'of', 'with', 'this', 'is', 'in',
                      'to', 'it', 'on', 'that', 'but', 'its', 'from',
                      'are', 'has', 'for', 'by', 'an', 'as', 'at',
                      'very', 'some', 'more', 'not', 'while', 'well',
                      'or', 'be', 'all', 'into', 'bit', 'just',
                      'like', 'up', 'also', 'made', 'out']
        
        filtered_desc_words = [w for w in desc_words if w not in stop_words and len(w) > 2]
        
        if len(filtered_desc_words) > 0:
            text = ' '.join(filtered_desc_words)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            ax4.imshow(wordcloud, interpolation='bilinear')
            ax4.axis('off')
            st.pyplot(fig4)
            
            word_freq = Counter(filtered_desc_words)
            st.subheader("Top 20 cuvinte")
            top_20 = word_freq.most_common(20)
            st.table(pd.DataFrame(top_20, columns=['Cuvânt', 'Frecvență']))
        else:
            st.info("Nu există suficiente date pentru a genera word cloud.")
    else:
        st.info("Nu există vinuri filtrate.")

with tab4:
    st.header("Analize și Corelații")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistici Descriptive")
        st.dataframe(filtered_df[['points', 'price', 'performance', 'alcohol']].describe())
    
    with col2:
        st.subheader("Corelații")
        corr_matrix = filtered_df[['points', 'price', 'performance', 'alcohol']].corr()
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax5)
        ax5.set_title("Matricea de corelație")
        st.pyplot(fig5)
    
    st.subheader("Top 10 vinuri după performance")
    top_performance = filtered_df.nlargest(10, 'performance')[
        ['title', 'country', 'variety', 'points', 'price', 'performance']
    ]
    st.dataframe(top_performance, use_container_width=True)
    
    st.subheader("Scatter Plot: Preț vs Puncte")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    scatter = ax6.scatter(
        filtered_df['price'],
        filtered_df['points'],
        c=filtered_df['performance'],
        cmap='viridis',
        alpha=0.6
    )
    ax6.set_xlabel("Preț")
    ax6.set_ylabel("Puncte")
    ax6.set_title("Relația dintre Preț și Puncte")
    plt.colorbar(scatter, label='Performance', ax=ax6)
    st.pyplot(fig6)