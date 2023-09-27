from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from geopy.geocoders import Nominatim

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index4.html')

@app.route('/results')
def results():
    return render_template('index3.html')

@app.route('/earthquake_results', methods=['POST'])
def earthquake_results():
    # Load earthquake dataset (assuming it's in CSV format)
    df = pd.read_csv('cleaned_dataset_10000.csv')

    place = request.form['place']

    # Load the shapefile of India
    india = gpd.read_file('IND_adm/IND_adm1.shp')

    # Filter the shapefile for the specified region
    place_file = india[india['NAME_1'] == place]

    # Extract the latitude and longitude coordinates of the earthquakes
    coords = df[['latitude', 'longitude']].values

    # Filter the earthquake data to include only those points within the specified region
    in_place = gpd.points_from_xy(df['longitude'], df['latitude'])
    filterd_df = gpd.GeoDataFrame(df, geometry=in_place)
    filterd_df = gpd.sjoin(filterd_df, place_file, op='within')
    results = filterd_df[["time","latitude","longitude","depth","magnitude","location"]]

    # Perform clustering on the filtered earthquake data
    cluster_data = np.array(filterd_df[['latitude','longitude','magnitude']])
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    clusters = dbscan.fit_predict(cluster_data)

    filterd_df['Clusters'] = clusters

    # Create a folium map centered on the specified region
    geolocator = Nominatim(user_agent='my_app_name')
    location = geolocator.geocode(place, timeout=None)
    m = folium.Map(location=[location.latitude, location.longitude], zoom_start=6, width="100%", height="100%")

    # Add the boundaries of the specified region to the folium map
    folium.GeoJson(place_file).add_to(m)

    # Add the earthquake points to the folium map
    fg = folium.FeatureGroup(name='Earthquake')
    for index, row in filterd_df.iterrows():
        color = 'Green'  # default color for magnitude < 5.0
        if row['Clusters'] == -1:
            magnitude = row['magnitude']
            if magnitude >= 8.0:
                color = 'red'
            elif magnitude >= 5.0:
                color = 'orange'
            fg.add_child(folium.CircleMarker(location=[row['latitude'], row['longitude']], 
                                              popup="Magnitude = "+ str(row['magnitude'])+"\n"+row['location'], radius=3, color=color, fill=True))
    m.add_child(fg)

    # Add layer control to the map
    folium.LayerControl().add_to(m)

    # # Save the folium map to an HTML file
    # m.save('templates/earthquake_results.html')

    # # Render the earthquake results page
    # return render_template('earthquake_results.html', results=results.to_html())

    #Render the map into right section of html file
    return render_template('index3.html',my_map = m._repr_html_())

if __name__ == "__main__":
    app.run(debug=True)
