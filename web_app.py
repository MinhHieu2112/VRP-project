import os
import json
import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory
from main import ALGORITHMS

app = Flask(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_algorithm_config_path(algo_id, algo):
    """Return the config file that belongs to the selected algorithm."""
    config_filename = 'config_tabu.json' if algo_id == 6 else 'config.json'
    return os.path.join(os.path.dirname(algo['script']), config_filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    algos_list = []
    # Thứ tự từ nhanh tới chậm: OR-Tools(2) > PyVRP(1) > ALNS(4) > Tabu(6) > SA(5) > ACO(3)
    speed_order = [2, 1, 4, 6, 5, 3]
    
    for k in speed_order:
        if k in ALGORITHMS:
            v = ALGORITHMS[k]
            if "MILP" not in v["name"]:  # Bỏ qua MILP
                algos_list.append({"id": k, "name": v["name"]})
                
    # Nếu có thuật toán khác ngoài danh sách sắp xếp thì nhét xuống cuối
    for k, v in ALGORITHMS.items():
        if k not in speed_order and "MILP" not in v["name"]:
            algos_list.append({"id": k, "name": v["name"]})
            
    return jsonify(algos_list)

@app.route('/api/run', methods=['POST'])
def run_algorithm():
    data = request.json
    algo_id = int(data.get('algorithm_id'))
    num_points = int(data.get('num_points', 50))

    if num_points <= 0:
        return jsonify({"error": "Số điểm phải lớn hơn 0"}), 400
    
    if algo_id not in ALGORITHMS:
        return jsonify({"error": "Thuật toán không tồn tại"}), 400
        
    algo = ALGORITHMS[algo_id]
    cwd = algo['cwd']
    
    # Cập nhật số điểm trong config.json của thuật toán (nếu có)
    config_path = get_algorithm_config_path(algo_id, algo)
    if not os.path.exists(config_path):
        return jsonify({"error": f"Không tìm thấy config: {config_path}"}), 500

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        if 'common_model_parameters' not in config_data:
            config_data['common_model_parameters'] = {}
        config_data['common_model_parameters']['num_points'] = num_points

        # Đã gỡ bỏ Fast Mode theo yêu cầu để thuật toán dò tìm kết quả tối ưu nhất.
        # Thuật toán sẽ dùng thông số gốc trong file config.json của từng thư mục.

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
    except (OSError, json.JSONDecodeError) as e:
        return jsonify({"error": f"Không thể cập nhật config: {e}"}), 500
            
    # Chạy thuật toán
    print(f"Đang chạy {algo['name']} với {num_points} điểm...")
    try:
        import sys
        import subprocess
        # Dùng sys.executable để lấy đúng python của môi trường hiện tại
        # Thiết lập PYTHONIOENCODING=utf-8 để tránh lỗi in text unicode trên console Windows
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        result = subprocess.run(
            [sys.executable, algo['script']],
            cwd=cwd, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            env=env
        )
        if result.returncode != 0:
             return jsonify({"error": "Lỗi khi chạy thuật toán", "details": result.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    # Đọc kết quả từ file JSON
    algo_folder_map = {
        1: "py_vrp",
        2: "or_tools",
        3: "aco",
        4: "alns",
        5: "sa",
        6: "tabu"
    }
    
    subfolder = algo_folder_map.get(algo_id)
    if not subfolder:
        return jsonify({"error": "Không xác định được thư mục kết quả"}), 500
        
    result_dir = os.path.join(PROJECT_ROOT, "Results", subfolder)
    json_path = os.path.join(result_dir, "latest_result.json")
    
    if not os.path.exists(json_path):
         return jsonify({"error": "Không tìm thấy file kết quả sau khi chạy"}), 500
         
    with open(json_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
        
    # Trả thêm map url và stdout log
    result_data['map_url'] = f"/api/map/{subfolder}/route_map.html"
    result_data['log'] = result.stdout
    
    return jsonify(result_data)

@app.route('/api/map/<path:subfolder>/<filename>')
def serve_map(subfolder, filename):
    result_dir = os.path.join(PROJECT_ROOT, "Results", subfolder)
    return send_from_directory(result_dir, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
