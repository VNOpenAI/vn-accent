## Giới thiệu

Mã nguồn ngày được xây dựng dựa trên bộ mã nguồn tại [đây](https://github.com/vudaoanhtuan/vietnamese-tone-prediction), với mục đích thêm dấu tiếng Việt cho báo cáo y tế.

**Các thay đổi đã/sẽ thực hiện:**

- [x] Refactor code + Thêm model LSTM + Linear.
- [x] Sử dụng dữ liệu từ wiki + báo y tế.
- [x] Thêm metric + code test.
- [x] Thêm code tiền/hậu xử lý cho chuỗi đầu vào.
- [ ] Thử nghiệm mô hình đoán dấu thay vì đoán từ.

## Model

Repo này triển khai các mô hình dựa trên kiến trúc LSTM và Transformer. Trong đó với Transformer, do cần dự đoán một từ có dấu tương ứng với một từ không có dấu nên chỉ cần dùng Encoder là đủ.  

## Data

### Nguồn data

- [Wikipedia](https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2). File dữ liệu tải sẵn tại bài viết này: <https://phamdinhkhanh.github.io/2020/05/28/TransformerThemDauTV.html>.
- Dữ liệu tin tức và bài viết y tế tự crawl.
### Kết quả
- Dữ liệu tổng hợp được tại [đây](https://drive.google.com/drive/folders/1Ik_oK5_AeU60LZ2cx3nOAycM-HG2BsGp?fbclid=IwAR3x-rbGZRDLaC_tTccJvF2H2S2zsAlZxQ_1RwRS4iQXZdGQD5qKAYMtT7Q)
### Một số bước tiền xử lí với 2 tập này:

- Với tập wiki thì định dạng của nó là xml nên cần dùng tool [wikiextractor](https://github.com/attardi/wikiextractor) để lấy nội dung text của các bài viết. Sau đó tách các câu dựa vào các dấu chấm câu như `.!;:`
- Với tập các bài báo cũng tách các câu như vậy
- Với mỗi câu, loại bỏ số, các dấu chấm câu và đưa về chữ thường
- Loại bỏ các câu có dưới 10 từ và lớn hơn 200 từ rồi ghi ra file text, mỗi câu một dòng được một file khoảng 5300000 dòng
- Dùng scipt để tạo file không có dấu từ file trên và chia thành 2 tập training và validation (tỉ lệ 85-15)

## Tokenizer

Ở đây chúng ta không dùng tokenizer tạo từ các tập data trên mà dùng word list từ [vietnamese-wordlist](https://github.com/duyetdev/vietnamese-wordlist) để tạo bộ tokenizer bằng cách split các từ trong word list và chọn lại các từ đơn. 
Kết quả được khoảng 9000 từ có dấu và 3000 từ không có dấu.

## Demo

- Transformer Envolved:

```
sh run_transformer_evolved.sh
```

## Train

- Transformer Envolved:

```
sh train_transformer_evolved.sh
```

## Test

- Transformer Envolved:

```
sh test_transformer_evolved.sh
```

## Tham khảo

Mã nguồn được xây dựng trên mã nguồn của tác giả Vũ Đào Anh Tuấn tại [đây](https://github.com/vudaoanhtuan/vietnamese-tone-prediction).
