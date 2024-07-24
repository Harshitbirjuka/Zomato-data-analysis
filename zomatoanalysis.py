import numpy as np 
import pandas as pd
import os
import seaborn as sns
print(os.listdir("D:\zomato"))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
from tqdm import tqdm
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models import word2vec
import nltk
# Load the CSV file
'''file_path = 'D:\zomato\zomato.csv'  # Update with your file path if necessary
data = pd.read_csv(file_path)

# Function to clean text data
def clean_text(text):
    if isinstance(text, str):
        # Fix common encoding issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Correct spacing issues
        text = re.sub(r'\s+', ' ', text)
        # Correct grammatical errors and spacing (basic cleaning)
        text = text.replace(' .', '.').replace(' ,', ',')
        # Strip leading and trailing whitespace
        text = text.strip()
    return text

# Apply the clean_text function to relevant columns
text_columns = ['address', 'name', 'location', 'rest_type', 'dish_liked', 'cuisines', 'reviews_list']
for col in text_columns:
    data[col] = data[col].apply(clean_text)

# Handle missing values (example: fill missing values with 'Unknown')
data.fillna('Unknown', inplace=True)

# Function to clean reviews list
def clean_reviews(reviews):
    if isinstance(reviews, str):
        # Convert string representation of list to actual list
        reviews = eval(reviews)
        cleaned_reviews = []
        for rating, review in reviews:
            cleaned_reviews.append((clean_text(rating), clean_text(review)))
        return cleaned_reviews
    return reviews

data['reviews_list'] = data['reviews_list'].apply(clean_reviews)

# Save the cleaned data back to a CSV file
cleaned_file_path = 'D:\zomato\zomatocleaned.csv'
data.to_csv(cleaned_file_path, index=False)

print("Data cleaning complete. Cleaned data saved to:", cleaned_file_path)'''
#df = pd.read_csv(r'D:\zomato\zomatocleaned.csv')
'''
print(df.head())
print(df.info())



# Plotting
plt.figure(figsize=(10,7))
chains = df['name'].value_counts()[:20]
sns.barplot(x=chains, y=chains.index, palette='deep')
plt.title("Most Famous Restaurants Chains in Bangalore")
plt.xlabel("Number of Outlets")
plt.ylabel("Restaurant Chain")
plt.show()

'''

'''
# Assuming the location column in your CSV is named 'location'
location_counts = df['location'].value_counts()

# Calculate percentages
total_restaurants = location_counts.sum()
location_percentages = location_counts / total_restaurants * 100

# Group locations based on percentage ranges
grouped_locations = {
    'Less than 1%': location_percentages[location_percentages < 1].sum(),
    'Between 1% and 1.5%': location_percentages[(location_percentages >= 1) & (location_percentages < 1.5)].sum()
}

# Collect locations with more than 1.5%
individual_locations = location_percentages[location_percentages >= 1.5]

# Add individual locations to grouped_locations
for location, percentage in individual_locations.items():
    grouped_locations[location] = percentage

# Create a new DataFrame for grouped data
grouped_data = pd.Series(grouped_locations)

# Plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Restaurants by Location')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


#Average Rating and Rating distribution
rating = df['rate'].dropna().apply(lambda x: float(x.split('/')[0]) if (len(x) > 3 and x != 'Unknown') else np.nan).dropna()
print(rating.mean())

# Plotting a distribution plot (histogram)
plt.figure(figsize=(6, 5))
sns.distplot(rating, bins=20, kde=False, color='skyblue', hist_kws={'edgecolor': 'black'})
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# Filter out rows where 'rate' or 'approx_cost(for two people)' is 'Unknown' or 'NEW'
filtered_df = df[(df['rate'].str.contains('^\d+(\.\d+)?/5$', na=False)) & (df['approx_cost(for two people)'] != 'Unknown')]

# Further clean the data by dropping any remaining NaN values
cost_dist = filtered_df[['rate', 'approx_cost(for two people)', 'online_order']].dropna()

# Convert 'rate' to float
cost_dist['rate'] = cost_dist['rate'].apply(lambda x: float(x.split('/')[0]))

# Convert 'approx_cost(for two people)' to integer
cost_dist['approx_cost(for two people)'] = cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',', '')))

# Plotting
plt.figure(figsize=(10, 7))
sns.scatterplot(x="rate", y='approx_cost(for two people)', hue='online_order', data=cost_dist)
plt.title('Scatter Plot of Ratings vs Approximate Cost for Two People')
plt.xlabel('Rating')
plt.ylabel('Approx Cost for Two People')
plt.show()

plt.figure(figsize=(6,6))
sns.distplot(cost_dist['approx_cost(for two people)'])
plt.show()

#Relation between Rating and online/offline orders
#Restaurants acception online orders have higher rating
votes_yes=df[df['online_order']=="Yes"]['votes']
trace0=go.Box(y=votes_yes,name="accepting online orders",
              marker = dict(
        color = 'rgb(214, 12, 140)',
    ))

votes_no=df[df['online_order']=="No"]['votes']
trace1=go.Box(y=votes_no,name="Not accepting online orders",
              marker = dict(
        color = 'rgb(0, 128, 128)',
    ))

layout = go.Layout(
    title = "Box Plots of votes",width=800,height=500
)

data=[trace0,trace1]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


cost_dist=filtered_df[['rate','approx_cost(for two people)','location','name','rest_type']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]))
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
def return_budget(location,rest):
    budget=cost_dist[(cost_dist['approx_cost(for two people)']<=400) & (cost_dist['location']==location) & 
                     (cost_dist['rate']>4) & (cost_dist['rest_type']==rest)]
    return(budget['name'].unique())
return_budget('BTM',"Quick Bites")
'''



import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import folium
from folium.plugins import HeatMap

# Load data
df = pd.read_csv(r'D:\zomato\zomatocleaned.csv')

# Prepare unique locations DataFrame
locations = pd.DataFrame({"Name": df['location'].unique()})
locations['Name'] = locations['Name'].apply(lambda x: "Bangalore " + str(x))

# Initialize geolocator
geolocator = Nominatim(user_agent="app")

# Geocode locations
lat_lon = []
for location in locations['Name']:
    loc = geolocator.geocode(location)
    if loc is None:
        lat_lon.append(None)
    else:    
        geo = (loc.latitude, loc.longitude)
        lat_lon.append(geo)

# Add geolocation data to DataFrame
locations['geo_loc'] = lat_lon
locations.to_csv('locations.csv', index=False)

# Clean up location names
locations['Name'] = locations['Name'].apply(lambda x: x.replace("Bangalore ", ""))

print(locations.head())

# Count restaurant locations and merge with geolocations
Rest_locations = pd.DataFrame(df['location'].value_counts().reset_index())
Rest_locations.columns = ['Name', 'count']
Rest_locations = Rest_locations.merge(locations, on='Name', how="left").dropna()

# Generate base map
def generateBaseMap(default_location=[12.97, 77.59], default_zoom_start=12):
    return folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

# Extract latitude and longitude
Rest_locations['lat'] = Rest_locations['geo_loc'].apply(lambda x: x[0])
Rest_locations['lon'] = Rest_locations['geo_loc'].apply(lambda x: x[1])

# Create heatmap for all locations
basemap = generateBaseMap()
HeatMap(Rest_locations[['lat', 'lon', 'count']].values.tolist(), zoom=20, radius=15).add_to(basemap)
basemap.save("all_locations_heatmap.html")

# Function to produce data for a specific cuisine
def produce_data(col, name):
    data = pd.DataFrame(df[df[col] == name].groupby(['location'], as_index=False)['url'].agg('count'))
    data.columns = ['Name', 'count']
    print(data.head())
    data = data.merge(locations, on="Name", how='left').dropna()
    data['lat'] = data['geo_loc'].apply(lambda x: x[0])
    data['lon'] = data['geo_loc'].apply(lambda x: x[1])
    return data.drop(['geo_loc'], axis=1)

# Generate heatmap for North Indian cuisine
North_India = produce_data('cuisines', 'North Indian')
basemap = generateBaseMap()
HeatMap(North_India[['lat', 'lon', 'count']].values.tolist(), zoom=20, radius=15).add_to(basemap)
basemap.save("north_indian_heatmap.html")

 
'''
food=produce_data('cuisines','South Indian')
basemap=generateBaseMap()
HeatMap(food[['lan','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)
basemap

def produce_chains(name):
    data_chain = pd.DataFrame(df[df["name"] == name]['location'].value_counts().reset_index())
    data_chain.columns = ['Name', 'count']
    data_chain = data_chain.merge(locations, on="Name", how="left").dropna()
    
    # Handle cases where geo_loc is missing or None
    data_chain['geo_loc'] = data_chain['geo_loc'].apply(lambda x: x if isinstance(x, tuple) else (np.nan, np.nan))
    
    data_chain['lan'], data_chain['lon'] = zip(*data_chain['geo_loc'].values)
    return data_chain[['Name', 'count', 'lan', 'lon']]

df_1=df.groupby(['rest_type','name']).agg('count')
datas=df_1.sort_values(['url'],ascending=False).groupby(['rest_type'],
                as_index=False).apply(lambda x : x.sort_values(by="url",ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})

mapbox_access_token="pk.eyJ1Ijoic2hhaHVsZXMiLCJhIjoiY2p4ZTE5NGloMDc2YjNyczBhcDBnZnA5aCJ9.psBECQ2nub0o25PgHcU88w"

def produce_trace(data_chain,name):
        data_chain['text']=data_chain['Name']+'<br>'+data_chain['count'].astype(str)
        trace =  go.Scattermapbox(
           
                lat=data_chain['lan'],
                lon=data_chain['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=data_chain['count']*4
                ),
                text=data_chain['text'],name=name
            )
        
        return trace



cafe=datas[datas['rest_type']=='Cafe']
cafe
data=[]  
for row in cafe['name']:
    data_chain=produce_chains(row) 
    trace_0=produce_trace(data_chain,row)
    data.append(trace_0)



layout = go.Layout(title="Cafe Restaurant chains locations around Banglore",
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,style="streets",
        center=dict(
            lat=12.96,
            lon=77.59
        ),
        pitch=0,
        zoom=10
    ),
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='Montreal Mapbox')


all_ratings = []

for name,ratings in tqdm(zip(df['name'],df['reviews_list'])):
    ratings = eval(ratings)
    for score, doc in ratings:
        if score:
            score = score.strip("Rated").strip()
            doc = doc.strip('RATED').strip()
            score = float(score)
            all_ratings.append([name,score, doc])
            rating_df=pd.DataFrame(all_ratings,columns=['name','rating','review'])
rating_df['review']=rating_df['review'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]',"",x))
rating_df.to_csv("Ratings.csv")
rating_df.head()
rest=df['name'].value_counts()[:9].index
def produce_wordcloud(rest):
    
    plt.figure(figsize=(20,30))
    for i,r in enumerate(rest):
        plt.subplot(3,3,i+1)
        corpus=rating_df[rating_df['name']==r]['review'].values.tolist()
        corpus=' '.join(x  for x in corpus)
        wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1500, height=1500).generate(corpus)
        plt.imshow(wordcloud)
        plt.title(r)
        plt.axis("off")
        

        
        
produce_wordcloud(rest)
plt.figure(figsize=(7,6))
rating=rating_df['rating'].value_counts()
sns.barplot(x=rating.index,y=rating)
plt.xlabel("Ratings")
plt.ylabel('count')
rating_df['sent']=rating_df['rating'].apply(lambda x: 1 if int(x)>2.5 else 0)
stops=stopwords.words('english')
lem=WordNetLemmatizer()
corpus=' '.join(lem.lemmatize(x) for x in rating_df[rating_df['sent']==1]['review'][:3000] if x not in stops)
tokens=word_tokenize(corpus)

vect=TfidfVectorizer()
vect_fit=vect.fit(tokens)
id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vectorized_data=vect_fit.transform(tokens)
gensim_corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)
ldamodel = gensim.models.ldamodel.LdaModel(gensim_corpus,id2word=id_map,num_topics=5,random_state=34,passes=25)
counter=Counter(corpus)
out=[]
topics=ldamodel.show_topics(formatted=False)
for i,topic in topics:
    for word,weight in topic:
        out.append([word,i,weight,counter[word]])

dataframe = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        


# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(8,6), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.3, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=dataframe.loc[dataframe.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    #ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=8)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(dataframe.loc[dataframe.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')



'''