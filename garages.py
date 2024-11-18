import folium
import json

# Read the JSON file
with open('data/garages.json') as f:
    data = json.load(f)

# Create a base map centered on UK
m = folium.Map(location=[54.5, -3.5], zoom_start=6)

# Add markers for each garage
for garage in data['garageMarkers']:
    # Skip any garages without lat/lng
    if 'lat' not in garage or 'lng' not in garage:
        continue
        
    # Create popup content
    popup_content = f"""
        <b>{garage['companyName']}</b><br>
        Rating: {garage['averageRating']}/5<br>
        Jobs Done: {garage['jobsDoneCount']}
    """
    
    # Add marker
    folium.Marker(
        [float(garage['lat']), float(garage['lng'])],
        popup=popup_content,
        icon=folium.Icon(color='red' if not garage['bookingEnabled'] else 'green')
    ).add_to(m)

# Save the map
m.save('garages_map.html')