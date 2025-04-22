import cv2
import numpy as np
import torch
from skimage.io import imsave
import matplotlib.pyplot as plt
from model import build_model
from data_loader import create_test_loader

def fourier_transform_compatible_visualization(image, mask, output_path):
    """푸리에 변환 호환 마스크 생성"""
    # 원본 이미지 전처리
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 마스크 이진화 (20개 클래스별 채널 분리)
    binary_mask = np.zeros((mask.shape[0], mask.shape[1]))
    for i in range(mask.shape[2]):
        binary_mask = np.where(mask[:,:,i] > 0.5, i+1, binary_mask)
    
    # 시각화 결과 저장 (원본 + 마스크)
    np.savez_compressed(
        output_path,
        image=gray_image,
        mask=binary_mask.astype(np.uint8)
    )
    
    # 시각화용 오버레이 이미지 생성
    overlay = cv2.addWeighted(
        cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB), 0.7,
        plt.get_cmap('tab20')(binary_mask)[:,:,:3], 0.3, 0
    )
    imsave(f"{output_path}_overlay.png", overlay)

# 추론 파이프라인 수정
def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            outputs = model(images.to(device))
            pred_masks = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(images.size(0)):
                original_image = images[i].permute(1,2,0).numpy()
                save_path = f"results/test_{batch_idx}_{i}"
                fourier_transform_compatible_visualization(
                    (original_image * 255).astype(np.uint8),
                    pred_masks[i].transpose(1,2,0),
                    save_path
                )

if __name__ == "__main__":
    # 사전 훈련된 모델 로드
    model = build_model('multi_unet++_b0')  # config와 동일한 아키텍처
    checkpoint = torch.load('saved_models/best_model.pth')
    model.load_state_dict(checkpoint['state_dict'])
    
    # 테스트 데이터 로더 생성
    test_loader = create_test_loader(
        data_dir='data/test/img',
        mask_path='data/test/Vindr_RibCXR_test_mask.json'
    )
    
    # 추론 실행
    test_model(model, test_loader, device='cuda')
