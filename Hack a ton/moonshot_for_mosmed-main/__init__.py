import pydicom

import torch

from moonshot.model import moonshot
from moonshot.image_processor import ImageProcessor


# model - это объект класса timm.models.vision_transformer.VisionTransformer c подгруженными предобученными весами.
# Его можно использовать как вставной модуль в моделях для классификации, сегментации или других задач.
model = moonshot(model_filepath='D:/moonshot.pt')  # требуется передать путь до загруженного файла `moonshot.pt`

# image_processor - это объект, который отвечает за препроцессинг изображений,
# который необходим перед тем как подавать их в модель
image_processor = ImageProcessor()

dicom = pydicom.dcmread('/path/to/image.dcm')  # подгружаем DICOM с диска
image = image_processor.preprocess_dicom(dicom)  # DICOM -> PIL
image = image_processor(image)  # PIL -> torch.Tensor, тензор размера (1, 3, 518, 518)

# инференс модели
model.eval()
with torch.inference_mode():
    features, intermediates = model.forward_intermediates(image, return_prefix_tokens=True, output_fmt='NLC')

last_cls_token = features[:, 0]  # классификационный токен, тензор размера (1, 1024)
last_reg_tokens = features[:, 1:5]  # 4 токена-регистра, тензор размера (1, 4, 1024)
last_patch_tokens = features[:, 5:]  # (518 // 14) ** 2 = 1369 патч-токенов, тензор размера (1, 1369, 1024)

# для некоторых задач может быть полезно брать токены не с последнего, а с предпоследнего слоя
hidden_cls_token = intermediates[-2][1][:, 0]  # тензор размера (1, 1024)
hidden_reg_tokens = intermediates[-2][1][:, 1:]  # тензор размера (1, 4, 1024)
hidden_patch_tokens = intermediates[-2][0]  # тензор размера (1, 1369, 1024)