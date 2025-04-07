from Library import *
from Load_data import *
from Img_generator import *
from Model import *
from Train import *

model = load_model('/kaggle/working/save_model.h5')

class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Load ảnh và resize về kích thước đúng
img = load_img('/kaggle/input/helpme/double_wall_corrugated_1000x664.jpg', target_size=(224, 224))  # Resize ảnh về (224, 224)
img_array = img_to_array(img)  # Chuyển đổi ảnh thành mảng numpy
img_array = img_array / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
img_array = img_array.reshape(1, 224, 224, 3)  # Thêm batch dimension
img.show()
# Dự đoán với mô hình
predictions = model.predict(img_array)
# Lấy chỉ số của lớp có xác suất cao nhất
predicted_class_index = np.argmax(predictions)
predicted_label = class_labels[predicted_class_index]

plt.imshow(img)
plt.axis('on')
plt.show()

print("Predictions:", predictions)
print("Predicted Class Index:", predicted_class_index)
print("Predicted Label:", predicted_label)