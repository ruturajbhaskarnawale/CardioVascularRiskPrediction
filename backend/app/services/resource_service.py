
import pandas as pd
import math
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "data", "india_health_facilities.csv")

MAJOR_CITIES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Nashik": {"lat": 20.0112, "lon": 73.7909},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433}
}

class ResourceService:
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points on Earth using the Haversine formula.
        Returns distance in kilometers.
        """
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def find_resources(self, city_name, resource_type, max_distance_km=10):
        """
        Loads Indian healthcare data from a local CSV and finds nearby resources.
        """
        try:
            if not os.path.exists(DATASET_PATH):
                print(f"Dataset not found at {DATASET_PATH}")
                return []

            df = pd.read_csv(DATASET_PATH)

            user_location = MAJOR_CITIES.get(city_name)
            if not user_location:
                return []

            user_lat, user_lon = user_location['lat'], user_location['lon']

            # --- Filter for Maharashtra First ---
            df_maharashtra = df[df['State Name'].str.strip().str.lower() == 'maharashtra'].copy()

            # --- Data Cleaning ---
            required_cols = ['Latitude', 'Longitude', 'Facility Name']
            if not all(col in df_maharashtra.columns for col in required_cols):
                return []

            df_maharashtra = df_maharashtra.dropna(subset=['Latitude', 'Longitude'])
            df_maharashtra['Latitude'] = pd.to_numeric(df_maharashtra['Latitude'], errors='coerce')
            df_maharashtra['Longitude'] = pd.to_numeric(df_maharashtra['Longitude'], errors='coerce')

            # --- Calculate Distances ---
            # Applying row-wise is slow but fine for this dataset size
            distances = []
            for index, row in df_maharashtra.iterrows():
                dist = self.haversine_distance(user_lat, user_lon, row['Latitude'], row['Longitude'])
                distances.append(dist)
                
            df_maharashtra['distance_km'] = distances

            # --- Filter by Distance ---
            nearby_df = df_maharashtra[df_maharashtra['distance_km'] <= max_distance_km].copy()
            nearby_df['distance_km'] = nearby_df['distance_km'].round(2)
            
            # Sort and convert to list of dicts
            return nearby_df.sort_values('distance_km').to_dict(orient='records')

        except Exception as e:
            print(f"Error finding resources: {e}")
            return []

resource_service = ResourceService()
