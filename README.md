## Giới thiệu
Đây là một model sử dụng cho cuộc thi [Thêm dấu tiếng Việt](https://www.aivivn.com/contests/3) của AIVIVN.  
Public test score trên scoreboard là 0.97522 (accuracy)

## Model
Có thể coi đây là bài toán sequence prediction hoặc machine translation đều được  
- Với machine translation có thể dùng mô hình Encoder-Decoder ví dụ như Transformer. Mình đã thử với cách này, sử dụng Transformer và train trên dataset của cuộc thi thì được public score khoảng 0.95.
- Với sequence prediction mình sẽ trình bày bên dưới 

Do cần dự đoán một từ có dấu tương ứng với một từ không có dấu nên chỉ cần dùng Encoder là đủ.  
Cách đơn giản nhất là dùng 1 tầng LSTM+Linear là được một baseline khá dễ implement.  
Hoặc có thể dùng Encoder theo kiến trúc của Transformer và thêm một tầng Linear + softmax để predict. 
Trong cuộc thi này, mình dùng Encoder theo kiến trúc của [Evolved Transformer](https://arxiv.org/abs/1901.11117). 

## Data
Data mình lấy từ 2 nguồn chính là:
- [Wikipedia](https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2)
- [Một số bài báo được một bạn crawl trên github](https://github.com/hoanganhpham1006/Vietnamese_Language_Model/blob/master/Train_Full.zip)

Một số bước tiền xử lí với 2 tập này:
- với tập wiki thì định dạng của nó là xml nên cần dùng tool [wikiextractor](https://github.com/attardi/wikiextractor) để lấy nội dung text của các bài viết. Sau đó tách các câu dựa vào các dấu chấm câu như `.!;:`
- với tập các bài báo cũng tách các câu như vậy
- với mỗi câu, loại bỏ số, các dấu chấm câu và đưa về chữ thường
- sau đó loại bỏ các câu có dưới 10 từ và lớn hơn 200 từ rồi ghi ra file text, mỗi câu một dòng được một file khoảng 5300000 dòng
- sau đó dùng scipt để tạo file không có dấu từ file trên và chia thành 2 tập training và validation (tỉ lệ 85-15)

## Tokenizer
Ở đây mình không dùng tokenizer tạo từ các tập data trên mà mình dùng word list từ [vietnamese-wordlist](https://github.com/duyetdev/vietnamese-wordlist) để tạo bộ tokenizer bằng cách split các từ trong word list và chọn lại các từ đơn. 
Kết quả được khoảng 9000 từ có dấu và 3000 từ không có dấu 

## Training
Mình train `base_model` trên 1 GPU 2080Ti, thời gian train 1 epoch khoảng 5h, và train tới epoch 12 thì được điểm như trên scoreboard.  
Với `big_model`, thời gian train là 15h cho 1 epoch, do hết thời gian nên mình không thử model này nữa. 