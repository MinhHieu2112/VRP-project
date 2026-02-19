import folium
import random
import requests
import polyline

class Visualizer:
    def __init__(self, df_locations, osrm_url="http://localhost:5001"):
        self.df = df_locations.set_index('id')
        self.osrm_url = osrm_url
        self.depot_coords = (self.df.loc[0, 'lat'], self.df.loc[0, 'lon'])

    def _get_route(self, p1, p2):
        url = f"{self.osrm_url}/route/v1/driving/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=polyline"
        try:
            r = requests.get(url, timeout=1).json()
            return polyline.decode(r['routes'][0]['geometry'])
        except:
            return [p1, p2]

    def draw(self, routes_dict, output_path):
        m = folium.Map(location=self.depot_coords, zoom_start=13)
        folium.Marker(self.depot_coords, popup="DEPOT", icon=folium.Icon(color='red')).add_to(m)

        for v_id, route in routes_dict.items():
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            fg = folium.FeatureGroup(name=f"Xe {v_id}")
            path_coords = []

            for i in range(len(route) - 1):
                p1 = (self.df.loc[route[i], 'lat'], self.df.loc[route[i], 'lon'])
                p2 = (self.df.loc[route[i+1], 'lat'], self.df.loc[route[i+1], 'lon'])
                
                segment = self._get_route(p1, p2)
                path_coords.extend(segment)
                
                if route[i+1] != 0:
                    folium.CircleMarker(p2, radius=3, color=color, fill=True).add_to(fg)

            folium.PolyLine(path_coords, color=color, weight=3).add_to(fg)
            fg.add_to(m)

        folium.LayerControl().add_to(m)
        m.save(output_path)