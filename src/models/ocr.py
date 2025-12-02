import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2

from .segmentation import ResNet18UNet
from .recognition import TextRecognition
from ..config import CHAR2IDX, IDX2CHAR, VOCAB_SIZE
from ..utils.mask_processing import get_bboxes_from_mask, enhance_mask
from ..utils.sorting import sort_table
from ..utils.visualization import visualize, visualize_attention_maps


class OCR(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_decoder_layers=3, max_length=30):
        super(OCR, self).__init__()
        self.segmentation_model = ResNet18UNet()
        self.image_to_text_model = TextRecognition(vocab_size, d_model, nhead, num_decoder_layers, max_length)
        self.segmentation_model.load_state_dict(torch.load("detect_model_resnet18_epoch_30.pth"))
        self.image_to_text_model.load_state_dict(torch.load("best_model_res50_36.pth"))
        self.segmentation_model.eval()
        self.image_to_text_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmentation_model.to(self.device)
        self.image_to_text_model.to(self.device)
        self.word_transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  
        self.max_length = max_length
        self.sos_idx = CHAR2IDX['<SOS>']
        self.eos_idx = CHAR2IDX['<EOS>']
        self.unk_idx = CHAR2IDX['<UNK>']
        self.pad_idx = CHAR2IDX['<PAD>']
        self.vocab_size = vocab_size
        self.final_max = 1024
    
    def get_bboxes(self, image, enhance_threshold=0.9, threshold=0.9):
        width, height = image.size
        max_size = max(width, height)
        ratio = self.final_max / max_size
        print(f"Image size: {width}x{height}")
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = image.resize((new_width, new_height), Image.BILINEAR)
        img = self.image_transform(img).unsqueeze(0).to(self.device)
        
        # Use no_grad context to disable gradient computation
        with torch.no_grad():
            pred_mask = self.segmentation_model(img)[0, 0].cpu().numpy()
        pred_mask = cv2.resize(pred_mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # enhanced_mask = enhance_mask(pred_mask, min_threshold=enhance_threshold)
        bboxes = get_bboxes_from_mask(pred_mask, threshold=threshold)
        import matplotlib.pyplot as plt
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
        plt.show()
        
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        return bboxes, new_width, new_height, ratio
    
    def crop_and_recognize(self, image, bboxes, new_width, new_height, ratio) -> str:
        image = image.resize((new_width, new_height), Image.BILINEAR)
        crops = []
        #Sort bboxes by x and y coordinates
        bboxes = sort_table(bboxes)
        for bbox in bboxes:
            x, y, w, h = bbox
            crop = image.crop((x, y, x + w, y + h))
            crop = self.word_transform(crop).unsqueeze(0).to(self.device)
            crops.append(crop)
        crops = torch.cat(crops, dim=0)
        list_preds = []
        for crop in crops:
            pred_seq, pred_text, attns_map = self.image_to_text_model.inference(crop, sos_idx=self.sos_idx, eos_idx=self.eos_idx, return_attn=True)
            visualize_attention_maps(self.image_to_text_model, crop, pred_text, attns_map, tgt_seq=pred_seq)
            list_preds.append(pred_text)
        return ' '.join(list_preds)
    
    def forward(self, image=None, image_path=False, enhance_threshold=0.95, threshold=0.9):
        bboxes = None
        if image_path:
            image = Image.open(image_path).convert('RGB')
        bboxes, new_width, new_height, ratio = self.get_bboxes(image, enhance_threshold, threshold)
        pred_text = self.crop_and_recognize(image, bboxes, new_width, new_height, ratio)
        return pred_text, bboxes, (new_width, new_height)
