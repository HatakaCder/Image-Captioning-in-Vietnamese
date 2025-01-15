# Image-Captioning-in-Vietnamese
Thành viên nhóm:
- 21520553 - Trần Hoài An
- 21522443 - Lường Đại Phát
- 21522792 - Phạm Quốc Việt (Trưởng nhóm)

# Hướng dẫn thực thi code:
**Link model đã huấn luyện**: https://drive.google.com/drive/folders/1ZB_WPfBjxlTfo2-ygxMFU_oIIz5rrTWm?usp=drive_link

**Model**: 
- CNN+LSTM theo file model_cnn_lstm.py
- ClipCap: theo file ipynb tương ứng
- BLIP :

```
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
```

- BLIP-2:
```
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

quant_config = BitsAndBytesConfig(load_in_8bit=True)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", quantization_config=quant_config)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
```
**Xử lý dữ liệu và train**: truy cập các file ipynb tương ứng với các mô hình.

**Test**: dữ liệu theo dạng df với 2 cột image_path và caption, đối với BLIP phải qua bước thêm dấu cho các câu bởi Generative AI (Gemini, GPT), đánh giá qua test.py
- BLIP:
```
# Load model đã huấn luyện
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./img_captioning_model/blip_flickr_vn.pth", map_location=device))
model = model.to(device)

for index, record in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing Captions"):
    img = test_dataset[index]['image']
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    caption = ' '.join(generated_caption.replace('_', '').split())
    df_test.loc[index, 'predict'] = caption
```
- BLIP-2:
```
#Load mô hình đã huấn luyện
checkpoint_path = "..." # đường dẫn mô hình
model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:0"))

for index, record in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing Captions"):
    img = test_dataset[index]['image']
    inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=25)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    caption = ' '.join(generated_caption.replace('_', ' ').split())
    df_test.loc[index, 'predict'] = caption
```
**Demo**:
+ Truy cập vào link sau: https://colab.research.google.com/drive/1R9suAi3P7iOm9mUkbM5QHfcExoS5a1tu
+ Thay đổi NGROK_KEY thành ngrok_key của bạn (dashboard.ngrok.com)
+ Thay đổi model_path thành thư mục lưu các mô hình trong drive của bạn (tạo lối tắt)
+ Chỉnh api gemini ở cell cuối genai.configure(api_key=API_KEY).
+ Run all và chọn link ngrok ở cell cuối của code để sử dụng app demo.

Link dataset: https://www.kaggle.com/datasets/vitngquang/uit-viic-v1-0-vietnamese-image-captioning
