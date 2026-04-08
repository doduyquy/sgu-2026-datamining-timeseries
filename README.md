# SGU 2026 Data mining about Time series

> Link source: https://github.com/Apress/time-series-algorithm-recipes/

> Link code chapter oke: https://github.com/Apress/time-series-algorithm-recipes/blob/main/Chapter%201.ipynb

## Tóm tắt seminar

Trong seminar này, chúng tôi tìm hiểu về dữ liệu time series và các cách thức trực quan hóa dữ liệu, phân tích dữ liệu và dùng các model thống kê để dự đoán dữ liệu.

Ngoài ra, chúng tôi còn lấy dữ liệu Bitcoin gần nhất, lấy các model trong sách để so sánh với các model Học máy hiện. Kết quả cho ra model **XGBoost** cho thấy kết quả khả quan nhất.


## Cấu trúc project
```
sgu-2026-datamining-time-series/
│
├── data
│   ├── btc.csv
│   ├── tractor_salesSales.csv
│   └── ...                
│
├── time-series/                    
│   │
│   ├── Modules/                 # save model output
│   │   ├── checkpoint.pkl       
│   │   └── ....py               
│   │
│   ├── notebook.ipynb          # some file notebook when experiment
├── Predict_Realtime_BTC_prices.png  # predict output 
└── ...    
```

- Lệnh chạy app: 
`
cd d:\DataMining\sgu-2026-datamining-timeseries\time-series
d:\DataMining\.venv\Scripts\python.exe -m streamlit run app.py`