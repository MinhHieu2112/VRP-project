# VRP Project

Dự án giải quyết Vehicle Routing Problem (VRP) sử dụng các thuật toán khác nhau.

## Cài đặt

1. Clone repository:
   ```bash
   git clone <repository-url>
   cd VRP-project
   ```

2. Tạo môi trường ảo Python:
   ```bash
   python -m venv venv
   ```

3. Kích hoạt môi trường ảo:
   - Trên Windows:
     ```cmd
     venv\Scripts\activate
     ```
   - Trên macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Cài đặt dependencies:
   ```bash
   pip install -r Algorithms/ALNS/requirements.txt
   ```

5. (Tùy chọn) Build PyVRP local (nếu cần phiên bản mới):
   - Cài đặt build tools:
     ```bash
     pip install meson ninja
     ```
   - Build PyVRP:
     ```bash
     cd Algorithms/PyVRP
     meson setup builddir
     meson compile -C builddir
     pip install -e .
     cd ../..
     ```

## Chạy

```bash
python main.py
```

## Cấu trúc dự án

- `Algorithms/`: Các thuật toán giải VRP (ALNS, ORTools, PyVRP)
- `Data/`: Dữ liệu đầu vào
- `Utils/`: Utilities chung
- `Results/`: Kết quả đầu ra
- `Test/`: Test scripts