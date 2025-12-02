# Optical Character Recognition (OCR) with Transformer

A state-of-the-art OCR system that combines ResNet-based text segmentation with Transformer-based text recognition to accurately detect and recognize text from images.

## Overview

This project implements an end-to-end OCR pipeline consisting of two main components:

1. **Text Segmentation**: ResNet18 U-Net architecture for detecting text regions
2. **Text Recognition**: ResNet50 encoder with Transformer decoder for character-level text recognition

## Key Features

- Hybrid CNN-Transformer architecture for improved accuracy
- Attention visualization for interpretability
- Support for multilingual text (Vietnamese and other languages)
- Bounding box extraction and text sorting
- Pre-trained model support

## Project Structure

```
src/
├── config.py                    # Configuration and vocabulary management
├── main.py                      # Application entry point
├── layers/
│   ├── positional_encoding.py   # Positional encoding layer
│   └── transformer_decoder.py   # Transformer decoder layer
├── models/
│   ├── segmentation.py          # ResNet18 U-Net segmentation model
│   ├── recognition.py           # Text recognition model
│   └── ocr.py                   # Complete OCR pipeline
└── utils/
    ├── mask_processing.py       # Mask processing utilities
    ├── sorting.py               # Bounding box sorting
    └── visualization.py         # Visualization utilities
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ocr-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models and place them in the project root:
   - `detect_model_resnet18_epoch_30.pth` (text segmentation)
   - `best_model_res50_36.pth` (text recognition)

## Usage

### Basic Usage

```python
from src.models.ocr import OCR
from src.config import VOCAB_SIZE
from PIL import Image

# Initialize OCR model
ocr_model = OCR(vocab_size=VOCAB_SIZE)

# Process image
image_path = "path/to/image.jpg"
pred_text, bboxes, dimensions = ocr_model(
    image_path=image_path,
    enhance_threshold=0.94,
    threshold=0.85
)

print(f"Recognized Text: {pred_text}")
print(f"Bounding Boxes: {bboxes}")
```

### Running the Application

```bash
python src/main.py
```

## Model Architecture

### Text Segmentation (ResNet18 U-Net)

- **Encoder**: ResNet18 with 5 encoder blocks (64, 64, 128, 256, 512 channels)
- **Decoder**: Symmetric decoder with skip connections and upsampling layers
- **Output**: Binary mask indicating text regions

### Text Recognition (ResNet50 + Transformer)

- **CNN Encoder**: ResNet50 (ImageNet pre-trained) without final layers
- **Transformer Decoder**: 
  - Multi-head self-attention
  - Cross-attention to CNN features
  - Feed-forward layers with GELU activation
  - Positional encoding for sequence positions
  - Layer scaling for improved training stability

## Configuration

Edit `src/config.py` to customize:

- Vocabulary management
- Character-to-index mappings
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`

## Performance

The model achieves high accuracy on multi-language OCR tasks with:
- Efficient text detection via semantic segmentation
- Character-level accuracy through attention-based recognition
- Automatic text reading order correction

## Troubleshooting

### Model Loading Issues

Ensure model weights are in the correct path and format. Update paths in `src/models/ocr.py` if needed.

### Memory Issues

- Reduce batch size in inference
- Adjust `max_length` parameter in `TextRecognition` class
- Use CPU mode: Set `self.device = torch.device("cpu")`

### Poor Recognition Results

- Adjust `threshold` parameter (0.7-0.95) for text detection sensitivity
- Pre-process images: resize, enhance contrast, correct rotation
- Verify vocabulary matches training data

## Requirements

See `requirements.txt` for complete dependencies:
- torch >= 1.9.0
- torchvision >= 0.10.0
- opencv-python >= 4.5.0
- pandas >= 1.1.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- Pillow >= 8.0.0

## License

[Specify your license here]

## Citation

If you use this project in your research, please cite:

```bibtex
@project{ocr_transformer,
  title={OCR with Transformer Architecture},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or support, please reach out to [your contact information]