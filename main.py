import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'

Path('graphs').mkdir(parents=True, exist_ok=True)

def load_data():
    """Handling the import of data"""
    try:
        df = pd.read_csv("wine_data_lab2.csv", skiprows=1)
        return df
    except FileNotFoundError:
        print("✗ Eroare: Fișierul 'lab2/wine_data_lab2.csv' nu a fost găsit")
        print("  Asigură-te că fișierul există în directorul corect")
        exit()
    except Exception as e:
        print(f"✗ Eroare la încărcare: {e}")
        exit()

def clean_data(df):
    """Cleaning imported data"""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("There are", missing.sum(), "missing values")
        print("Replacing them with the mean of neighbouring values")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate()
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print("Deleting", duplicates, "duplicated data")
        df = df.drop_duplicates()
    
    df['category'] = df["category"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].replace({
        "rose": "rosé",
        "roze": "rosé"
    })
    
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace({
            'NaN': np.nan,
            'None': np.nan,
            'null': np.nan,
            "n/a": np.nan,
            '': np.nan
        })
        
        if df[col].astype(str).str.len().max() < 30:
            print(df[col].value_counts(dropna=True))
    
    return df

def transforming_data(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    df['performance'] = df['points'] / df['price']
    
    print(df[numeric_cols].describe())
    
    all_words = []
    for text in df["description"].astype(str):
        cleaned_word = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        all_words.extend(cleaned_word.split())
    
    stop_words = ['and', 'the', 'a', 'of', 'with', 'this', 'is', 'in',
                  'to', 'it', 'on', 'that', 'but', 'its', 'from',
                  'are', 'has', 'for', 'by', 'an', 'as', 'at', 
                  'very', 'some', 'more', 'not', 'while',
                  'well', 'now.', 'or', 'be', 'all', 
                  'into','bit', 'just', 'like', 'up', 
                  'also', 'made', 'out',
                  ]
    filtered_words = [w for w in all_words if w not in stop_words]
    word_frequency = Counter(filtered_words)
    top_words = word_frequency.most_common(10)
    print(top_words)
    
    print(df['description'].astype(str).str.len().mean())
    
    text = ' '.join(filtered_words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("graphs/wine_wordcloud.png")
    plt.show()
    
    word_cols = [f"word_{w[0]}" for w in top_words]
    
    for w_text, w_count in top_words:
        df[f"word_{w_text}"] = df["description"].str.lower().str.contains(rf"\b{w_text}\b", regex=True).astype(int)
    
    correlation_price = df[word_cols + ["price"]].corr()["price"]
    correlation_points = df[word_cols + ["points"]].corr()["points"]
    
    print("\nCorelație cu prețul:")
    print(correlation_price.drop("price"))
    
    print("\nCorelație cu ratingul (points):")
    print(correlation_points.drop("points"))
    
    top_varieties_list = df['variety'].value_counts().head(20).index.tolist()
    df_top_varieties = df[df['variety'].isin(top_varieties_list)]
    df_dummies = pd.get_dummies(df_top_varieties, columns=['variety'], dtype=int)
    variety_dummy_cols = [col for col in df_dummies.columns if col.startswith('variety_')]
    correlation_varieties_matrix = df_dummies[word_cols + variety_dummy_cols].corr()
    
    print("\nAnaliza de corelare a cuvintelor cu soiurile (doar cuvintele vs soiurile):")
    print(correlation_varieties_matrix.loc[word_cols, variety_dummy_cols])
    
    print(df[['points', "price"]].corr()["price"])
    print(df[['points', "alcohol"]].corr()["points"])
    
    regions = df['region_1'].astype(str)
    top_regions = Counter(regions).most_common(10)
    
    region_cols = [f'region_{r[0]}' for r in top_regions]
    for r_region, r_count in top_regions:
        df[f'region_{r_region}'] = df['region_1'].str.lower().str.contains(rf"\b{r_region}\b", regex=True).astype(int)
    
    corelations_regions_price = df[region_cols + ['price']].corr()['price']
    corelations_regions_points = df[region_cols + ['points']].corr()['points']
    
    print(corelations_regions_price.drop('price'))
    print(corelations_regions_points.drop('points'))
    
    plt.hist(df['points'], bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Amount of wines")
    plt.ylabel("Points Awarded")
    plt.title("Histogram of points awarded to wines")
    plt.savefig("graphs/awards_points_histogram.png")
    plt.show()
    
    df_grup_tara_pret = df.groupby("country")["price"].mean().reset_index()
    df_grup_tara_pret = df_grup_tara_pret.sort_values("price", ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(df_grup_tara_pret["country"], df_grup_tara_pret["price"])
    plt.xticks(rotation=90)
    plt.xlabel("Tara")
    plt.ylabel("Pret mediu")
    plt.title("Pretul mediu al vinului per tara")
    plt.tight_layout()
    plt.savefig("graphs/average_wine_price.png")
    plt.show()
    
    df_group = df.groupby(["region_1", "category"]).size().reset_index(name="count")
    top_regions = df_group.groupby("region_1")["count"].sum().nlargest(20).index
    df_top = df_group[df_group["region_1"].isin(top_regions)]
    
    pivot = df_top.pivot_table(index="region_1", columns="category", values="count", fill_value=0)
    
    category_colors = {
        "dessert": "#FF61D0",
        "fortified": "#4B0082",
        "orange": "#FF9900",
        "port/sherry": "#F10303FF",
        "red": "#8B0000",
        "sparkling": "#FBE04AFF",
        "rosé": "#FF0080",
        "white": "#F5DEB3",
    }
    
    colors = [category_colors[c] for c in pivot.columns]
    
    pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 7),
        color=colors
    )
    
    plt.title("Distribuția vinurilor după categorii și regiuni")
    plt.xlabel("Regiune")
    plt.ylabel("Număr de vinuri")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("graphs/wines_per_country.png")
    plt.show()
    
    return df

df = load_data()
df = clean_data(df)
df = transforming_data(df)