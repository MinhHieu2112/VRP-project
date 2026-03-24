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

    def load_routes_from_txt(self, file_path):
        """
        Đọc file alns_result.txt và chuyển thành dictionary routes
        """
        routes_dict = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            start_parsing = False

            for line in lines:
                if "Lộ trình chi tiết:" in line:
                    start_parsing = True
                    continue

                if start_parsing and "Route #" in line:
                    parts = line.split(':')
                    route_id = int(parts[0].replace('Route #', '').strip())

                    nodes = [int(n) for n in parts[1].strip().split()]
                    full_route = [0] + nodes + [0]

                    routes_dict[route_id] = full_route

            return routes_dict

        except Exception as e:
            print(f"Lỗi khi đọc file kết quả: {e}")
            return None