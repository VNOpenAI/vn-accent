## Giới thiệu

Đây là một bản fork của model thêm dấu tiếng Việt tại [đây](https://github.com/vudaoanhtuan/vietnamese-tone-prediction) với mục đích thêm dấu tiếng Việt cho báo cáo y tế. Các thay đổi đã thực hiện:

- [x] Refactor code + Thêm model LSTM + Linear.
- [x] Sử dụng dữ liệu từ wiki + báo y tế.
- [ ] Thêm metric + code test.
- [ ] Thêm code tiền/hậu xử lý cho chuỗi đầu vào.
- [ ] Thay đổi tokenizer.

## Model

Có thể coi đây là bài toán sequence prediction hoặc machine translation đều được.

- Với machine translation có thể dùng mô hình Encoder-Decoder ví dụ như Transformer.
- Với sequence prediction mình sẽ trình bày bên dưới 

Do cần dự đoán một từ có dấu tương ứng với một từ không có dấu nên chỉ cần dùng Encoder là đủ.  
Cách đơn giản nhất là dùng 1 tầng LSTM+Linear là được một baseline khá dễ implement.  
Hoặc có thể dùng Encoder theo kiến trúc của Transformer và thêm một tầng Linear + softmax để predict. 
Repo này implement kiến trúc của [Evolved Transformer](https://arxiv.org/abs/1901.11117). 

## Data

### Nguồn data

- [Wikipedia](https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2). File dữ liệu tải sẵn tại bài viết này: <https://phamdinhkhanh.github.io/2020/05/28/TransformerThemDauTV.html>.
- Dữ liệu tin tức và bài viết y tế tự crawl.

### Một số bước tiền xử lí với 2 tập này:

- với tập wiki thì định dạng của nó là xml nên cần dùng tool [wikiextractor](https://github.com/attardi/wikiextractor) để lấy nội dung text của các bài viết. Sau đó tách các câu dựa vào các dấu chấm câu như `.!;:`
- với tập các bài báo cũng tách các câu như vậy
- với mỗi câu, loại bỏ số, các dấu chấm câu và đưa về chữ thường
- sau đó loại bỏ các câu có dưới 10 từ và lớn hơn 200 từ rồi ghi ra file text, mỗi câu một dòng được một file khoảng 5300000 dòng
- sau đó dùng scipt để tạo file không có dấu từ file trên và chia thành 2 tập training và validation (tỉ lệ 85-15)

## Tokenizer

Ở đây chúng ta không dùng tokenizer tạo từ các tập data trên mà dùng word list từ [vietnamese-wordlist](https://github.com/duyetdev/vietnamese-wordlist) để tạo bộ tokenizer bằng cách split các từ trong word list và chọn lại các từ đơn. 
Kết quả được khoảng 9000 từ có dấu và 3000 từ không có dấu.

## Training

- Transformer Envolved:

```
sh train_transformer_evolved.sh
```

- LSTM + Linear:

```
sh train_transformer_evolved.sh
```