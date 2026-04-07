Xử lý Dữ liệu và Biến Mục Tiêu (Target)
Nguồn dữ liệu: Mô hình gộp dữ liệu từ hai file CSV lại, loại bỏ các dòng trùng lặp và sắp xếp theo thời gian, thu được tổng cộng 13.261 dòng dữ liệu giá Bitcoin theo khung giờ.

Chuyển đổi Tính dừng (Stationarity): Thay vì dự báo giá tuyệt đối bằng USD (do tính không dừng của giá), mô hình chuyển đổi toàn bộ chuỗi dữ liệu sang dạng Log Return (tỷ suất sinh lời logarit) bằng công thức np.log(df['price'] / df['price'].shift(1)).

Biến mục tiêu (Target): Cột mục tiêu cần dự báo là Log Return của 1 giờ tiếp theo (target_log_return = log_return.shift(-1)). Trước khi đưa vào huấn luyện, cột giá trị thực (price) và cột mục tiêu được loại bỏ khỏi ma trận đặc trưng (drop_cols) nhằm tránh rò rỉ dữ liệu (Data Leakage).

2. Kỹ Thuật Đặc Trưng (Feature Engineering)

Độ trễ lợi nhuận (Lags): Lấy lịch sử biến động qua các mốc 1h, 2h, 3h, 6h, 12h, 24h, 48h và 168h (1 tuần).

Rolling Stats: Tính toán đường trung bình động (MA) và độ lệch chuẩn (Std) của lợi nhuận trong các cửa sổ quá khứ 5, 10, 20 và 50 giờ.

Động lượng (Momentum): Chênh lệch giữa lợi nhuận hiện hành và đường trung bình động 10 giờ (momentum_10 = log_return - return_ma_10).

MACD, RSI: Mô hình tính RSI (chu kỳ 14) và MACD (chênh lệch giữa EMA 12 và EMA 26), nhưng áp dụng công thức lên Log Return chứ không phải trên đường giá gốc.

Đặc trưng Thời gian: Bổ sung các tính năng mang tính chu kỳ như giờ trong ngày (hour), ngày trong tuần (day_of_week), và ngày trong tháng (day_of_month).

3. Cấu trúc Mô hình & Tìm kiếm Siêu tham số
Hàm mục tiêu: Mô hình XGBoost Regressor được cấu hình với hàm mục tiêu là reg:squarederror để tối thiểu hóa sai số bình phương.

Cửa sổ trượt (Sliding Window): Thay vì chia tỷ lệ Train/Test tĩnh, mô hình chỉ học trên 1.500 giờ gần nhất để liên tục bám sát xu hướng thị trường mới nhất.

Tìm kiếm tham số (Manual Tuning Loop): Một vòng lặp 36 tổ hợp được chạy để tìm ra bộ tham số tối ưu nhất tránh quá khớp (overfitting). Kết quả tốt nhất là: {'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.75, 'colsample_bytree': 0.5}.

Mô hình áp dụng early_stopping_rounds=50 trong quá trình huấn luyện chính thức, nghĩa là quá trình học sẽ tự ngắt nếu sai số tập validation không giảm sau 50 vòng lặp.

4. Kiểm Chứng  (Walk-Forward Validation)
Mô hình được đưa vào một chu trình "Walk-Forward Pipeline", trong đó nó chạy dự báo trên 1.963 giờ dữ liệu tương lai và tự động cập nhật lại trọng số (Retrain) định kỳ (mỗi tuần / 168 giờ). Kết quả ghi nhận được:

Chỉ số Tín hiệu trực tiếp (Direct Return Metrics): RMSE là 0.005692, MAE là 0.003853. Độ chính xác hướng đi (Hit Rate) - tức khả năng đoán đúng nến tiếp theo là xanh hay đỏ - đạt mức 49.52%.

Chỉ số về Giá (Price Metrics): Khi giải mã từ lợi nhuận về lại mệnh giá USD, sai số trung bình (MAE) rơi vào mức $278.43 và RMSE là $409.29.

5. Khâu Lưu Trữ (Exporting)
Cuối cùng, trọng số mô hình XGBoost được lưu lại dưới 2 định dạng (.pkl và .json). Đặc biệt, code xuất 500 dòng dữ liệu cuối cùng của chuỗi thành file live_buffer_data.csv. File này sẽ đóng vai trò làm "mỏ neo" cung cấp các thông số trễ (như MA 50, Lag 168) khi ứng dụng web (app.py) khởi động để dự báo dòng dữ liệu mới.