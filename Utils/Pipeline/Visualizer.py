import folium
import random
import requests
import polyline

class Visualizer:
    def __init__(self, df_locations, osrm_url="http://localhost:5001", use_osrm=True):
        self.df = df_locations.set_index('id')
        self.osrm_url = osrm_url
        self.depot_coords = (self.df.loc[0, 'lat'], self.df.loc[0, 'lon'])
        self.use_osrm = use_osrm and self._check_osrm_health()

    def _check_osrm_health(self):
        if not self.osrm_url:
            print("[VISUALIZER] OSRM URL empty, dùng line thẳng.")
            return False

        test_url = f"{self.osrm_url}/route/v1/driving/{self.depot_coords[1]},{self.depot_coords[0]};{self.depot_coords[1]},{self.depot_coords[0]}?overview=false"
        try:
            r = requests.get(test_url, timeout=2)
            if r.status_code == 200 and 'routes' in r.json():
                return True
        except Exception as e:
            print(f"[VISUALIZER] OSRM health check failed ({e}), fallback to straight lines.")
        return False

    def _get_route(self, p1, p2):
        if not self.use_osrm:
            return [p1, p2]

        url = f"{self.osrm_url}/route/v1/driving/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=polyline"
        try:
            r = requests.get(url, timeout=2)
            data = r.json()
            if r.status_code == 200 and data.get('routes'):
                return polyline.decode(data['routes'][0]['geometry'])
        except Exception as e:
            print(f"[VISUALIZER] OSRM route request failed ({e}); dùng line thẳng.")
        return [p1, p2]

    def draw(self, routes_dict, output_path):
        from folium.plugins import Search
        
        m = folium.Map(location=self.depot_coords, zoom_start=13)
        folium.Marker(
            self.depot_coords, 
            popup="<b>DEPOT (Kho hàng)</b>", 
            tooltip="Kho hàng",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

        geojson_features = []

        for v_id, route in routes_dict.items():
            # Tạo màu sáng dễ nhìn (tránh màu quá tối)
            color = "#{:06x}".format(random.randint(0x222222, 0xDDDDDD))
            fg = folium.FeatureGroup(name=f"Xe {v_id} (Màu {color})")
            path_coords = []

            for i in range(len(route) - 1):
                p1 = (self.df.loc[route[i], 'lat'], self.df.loc[route[i], 'lon'])
                p2 = (self.df.loc[route[i+1], 'lat'], self.df.loc[route[i+1], 'lon'])
                
                segment = self._get_route(p1, p2)
                path_coords.extend(segment)
                
                if route[i+1] != 0:
                    node_id = route[i+1]
                    # Thêm chú thích cho điểm khách hàng
                    folium.CircleMarker(
                        p2, 
                        radius=5, 
                        color=color, 
                        fill=True,
                        fill_opacity=0.9,
                        tooltip=f"Điểm {node_id} (Xe {v_id})",
                        popup=f"<b>Khách hàng:</b> {node_id}<br><b>Phục vụ bởi:</b> Xe {v_id}"
                    ).add_to(fg)
                    
                    # Thu thập dữ liệu để làm thanh tìm kiếm
                    geojson_features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [p2[1], p2[0]] # [lon, lat]
                        },
                        "properties": {
                            "name": f"Điểm {node_id}",
                            "vehicle": f"Xe {v_id}"
                        }
                    })

            # Thêm tooltip cho đường đi
            folium.PolyLine(
                path_coords, 
                color=color, 
                weight=4, 
                opacity=0.8,
                tooltip=f"Lộ trình của Xe {v_id}"
            ).add_to(fg)
            fg.add_to(m)

        # Chèn công cụ tìm kiếm
        if geojson_features:
            search_layer = folium.GeoJson(
                {"type": "FeatureCollection", "features": geojson_features},
                name="Công cụ tìm kiếm",
                show=False, 
                marker=folium.Circle(radius=0, opacity=0, fillOpacity=0) # Ẩn marker giả
            ).add_to(m)

            Search(
                layer=search_layer,
                geom_type='Point',
                placeholder='Nhập Điểm (vd: Điểm 15)...',
                collapsed=False,
                search_label='name'
            ).add_to(m)

        folium.LayerControl().add_to(m)
        m.save(output_path)