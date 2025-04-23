import torch
import yaml
import os
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus, FPN, Unet

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def create_model(config):
    """원본 학습 코드와 완벽 호환되는 모델 생성"""
    model_name = config['MODEL']['NAME']
    arch = model_name.split('(')[0].lower()
    encoder = f"efficientnet-{model_name.split('(')[1][:-1]}"
    in_channels = config['DATA']['INP_CHANNEL']
    
    # 아키텍처 선택
    if arch == 'unet++':
        model = UnetPlusPlus(encoder_name=encoder, 
                           classes=20, 
                           activation=None,
                           in_channels=in_channels)
    elif arch == 'unet':
        model = Unet(encoder_name=encoder, 
                    classes=20, 
                    activation=None,
                    in_channels=in_channels)
    elif arch == 'fpn':
        model = FPN(encoder_name=encoder, 
                   classes=20, 
                   activation=None,
                   in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

def inference_single_image(image_path, model, transform, device):
    # 그레이스케일 변환 및 정사각형 리사이즈
    image = Image.open(image_path).convert('L')
    
    # 1. 종횡비 유지하며 최대 512로 리사이즈
    image.thumbnail((512, 512), Image.LANCZOS)
    
    # 2. 원본 크기 저장 (추가된 부분)
    original_size = image.size
    
    # 3. 512x512 정사각형으로 패딩 추가
    new_image = Image.new('L', (512, 512), 0)
    new_image.paste(image, ((512 - image.width) // 2, 
                           (512 - image.height) // 2))
    
    tensor = transform(new_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
    
    masks = (torch.sigmoid(output) > 0.5).cpu().numpy().astype(np.uint8) * 255
    return masks, original_size  # 정상적으로 정의됨

def save_masks(masks, original_size, output_dir, img_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 모든 마스크 통합 (20개 채널 중 최대값 추출)
    combined = np.max(masks, axis=1).squeeze()  # shape: (512, 512)
    
    # 2. 원본 크기로 리사이즈 (NEAREST 보간법 사용)
    mask_img = Image.fromarray(combined)
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    
    # 3. 파일 저장
    output_path = os.path.join(output_dir, f"{img_name}_combined.png")
    mask_img.save(output_path)
    print(f'✅ 통합 마스크 저장 완료: {output_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--val_dir', default='data/val/img')
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 초기화
    config = load_config(args.config)
    model = create_model(config)
    
    # 가중치 로드 (원본 학습 코드와 동일한 방식)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 이미지 전처리 (원본 학습 코드 기준으로 수정)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 추론 실행
    for img_file in os.listdir(args.val_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(args.val_dir, img_file)
        masks, orig_size = inference_single_image(img_path, model, transform, device)
        save_masks(masks, orig_size, args.output_dir, os.path.splitext(img_file)[0])

if __name__ == '__main__':
    main()
