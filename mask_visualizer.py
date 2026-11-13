import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from datetime import datetime

def analyze_gt_mask(mask_path):
    """
    GT 마스크 이미지에서 Tooth와 Root의 픽셀 수를 계산합니다.
    
    Args:
        mask_path (str): 마스크 이미지 파일 경로
    
    Returns:
        dict: 분석 결과
    """
    # 이미지 로드
    mask_image = cv2.imread(mask_path)
    
    if mask_image is None:
        return {
            'filename': os.path.basename(mask_path),
            'error': 'Failed to load image'
        }
    
    # 고유한 색상 값들 찾기
    unique_colors = np.unique(mask_image.reshape(-1, 3), axis=0)
    
    # 각 색상별 픽셀 수 계산
    color_counts = {}
    for color in unique_colors:
        if not np.array_equal(color, [0, 0, 0]):  # 검은색 제외
            mask = cv2.inRange(mask_image, color, color)
            count = np.sum(mask > 0)
            if count > 0:
                color_counts[tuple(color)] = count
    
    # 가장 많은 픽셀을 가진 두 색상 찾기
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_colors) >= 2:
        # 첫 번째 색상 (가장 많은 픽셀) - Tooth로 가정
        tooth_color, tooth_pixels = sorted_colors[0]
        # 두 번째 색상 (두 번째로 많은 픽셀) - Root로 가정
        root_color, root_pixels = sorted_colors[1]
    elif len(sorted_colors) == 1:
        # 하나의 색상만 있는 경우
        tooth_color, tooth_pixels = sorted_colors[0]
        root_pixels = 0
    else:
        # 색상이 없는 경우
        tooth_pixels = 0
        root_pixels = 0
    
    # 비율 계산
    if tooth_pixels > 0:
        ratio = (root_pixels / tooth_pixels) * 100
    else:
        ratio = 0
    
    return {
        'filename': os.path.basename(mask_path),
        'tooth_pixels': int(tooth_pixels),
        'root_pixels': int(root_pixels),
        'ratio': round(ratio, 2),
        'error': None,
        'color_counts': color_counts,
        'unique_colors': unique_colors.tolist()
    }

def visualize_mask(mask_path, save_path=None):
    """
    마스크 이미지를 시각화합니다.
    
    Args:
        mask_path (str): 마스크 이미지 파일 경로
        save_path (str): 저장할 경로 (선택사항)
    """
    # 이미지 로드
    mask_image = cv2.imread(mask_path)
    
    if mask_image is None:
        print(f"오류: 이미지를 로드할 수 없습니다: {mask_path}")
        return
    
    # BGR에서 RGB로 변환
    mask_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    
    # 고유한 색상 값들 찾기
    unique_colors = np.unique(mask_image.reshape(-1, 3), axis=0)
    
    # 검은색 제외한 색상들
    non_black_colors = [color for color in unique_colors if not np.array_equal(color, [0, 0, 0])]
    
    # 시각화를 위한 색상 매핑 (작은 값들을 더 큰 값으로 확대)
    color_mapping = {
        (1, 1, 1): (255, 0, 0),    # Tooth: 빨간색
        (2, 2, 2): (0, 255, 0),    # Root: 초록색
        (3, 3, 3): (0, 0, 255),    # 기타: 파란색
    }
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 원본 마스크 (색상 확대)
    plt.subplot(2, 3, 1)
    enhanced_mask = np.zeros_like(mask_rgb)
    for color in non_black_colors:
        if tuple(color) in color_mapping:
            mask = cv2.inRange(mask_image, color, color)
            enhanced_mask[mask > 0] = color_mapping[tuple(color)]
    
    plt.imshow(enhanced_mask)
    plt.title(f'Enhanced Mask\n{os.path.basename(mask_path)}')
    plt.axis('off')
    
    # 원본 마스크 (그레이스케일로 표시)
    plt.subplot(2, 3, 2)
    gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_mask, cmap='gray')
    plt.title(f'Original Grayscale\n{os.path.basename(mask_path)}')
    plt.axis('off')
    
    # 색상별 분리된 마스크들
    for i, color in enumerate(non_black_colors[:4]):  # 최대 4개 색상만
        plt.subplot(2, 3, i + 3)
        
        # 해당 색상만 추출
        mask = cv2.inRange(mask_image, color, color)
        
        # 색상 마스크를 시각화용 색상으로 변환
        color_mask_vis = np.zeros_like(mask_rgb)
        if tuple(color) in color_mapping:
            color_mask_vis[mask > 0] = color_mapping[tuple(color)]
        else:
            # 매핑되지 않은 색상은 원본 색상 사용
            color_mask_vis[mask > 0] = color
        
        plt.imshow(color_mask_vis)
        plt.title(f'Color {i+1}: RGB{color}\nPixels: {np.sum(mask > 0):,}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"시각화 결과가 저장되었습니다: {save_path}")
    
    plt.show()

def analyze_gt_folder(folder_path, output_csv=None, visualize_first_n=5):
    """
    GT 마스크 폴더를 분석합니다.
    
    Args:
        folder_path (str): GT 마스크 폴더 경로
        output_csv (str): CSV 출력 파일명 (선택사항)
        visualize_first_n (int): 처음 n개 이미지 시각화
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"오류: 폴더를 찾을 수 없습니다: {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"오류: 지정된 경로가 폴더가 아닙니다: {folder_path}")
        return
    
    # 지원하는 이미지 확장자
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    # 폴더 내 이미지 파일 찾기
    mask_files = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            mask_files.append(file)
    
    if not mask_files:
        print(f"오류: 폴더에 이미지 파일이 없습니다: {folder_path}")
        return
    
    print(f"폴더에서 {len(mask_files)}개의 GT 마스크 이미지를 찾았습니다.")
    
    # 결과 저장 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"gt_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 마스크 이미지 분석
    results = []
    for i, mask_file in enumerate(mask_files, 1):
        print(f"\n[{i}/{len(mask_files)}] 분석 중: {mask_file.name}")
        
        result = analyze_gt_mask(str(mask_file))
        results.append(result)
        
        if result['error']:
            print(f"  - 오류: {result['error']}")
        else:
            print(f"  - Tooth 픽셀: {result['tooth_pixels']:,}")
            print(f"  - Root 픽셀: {result['root_pixels']:,}")
            print(f"  - 비율: {result['ratio']}%")
            
            # 색상 정보 출력 (첫 번째 이미지만)
            if i == 1:
                print(f"  - 고유 색상 수: {len(result['unique_colors'])}")
                print(f"  - 색상별 픽셀 수:")
                for color, count in result['color_counts'].items():
                    print(f"    RGB{color}: {count:,} 픽셀")
        
        # 처음 n개 이미지 시각화
        if i <= visualize_first_n:
            print(f"  - 시각화 중...")
            viz_path = os.path.join(output_dir, f"viz_{mask_file.stem}.png")
            visualize_mask(str(mask_file), viz_path)
    
    # CSV 파일로 저장
    if output_csv:
        save_results_to_csv(results, output_csv)
    else:
        # 기본 CSV 파일명 생성
        default_csv_filename = os.path.join(output_dir, f"gt_analysis_{timestamp}.csv")
        save_results_to_csv(results, default_csv_filename)
    
    return results, output_dir

def save_results_to_csv(results, csv_filename):
    """
    분석 결과를 CSV 파일로 저장합니다.
    
    Args:
        results (list): 분석 결과 리스트
        csv_filename (str): CSV 파일명
    """
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['파일명', 'Tooth_픽셀수', 'Root_픽셀수', 'Root_비율(%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            if result['error']:
                writer.writerow({
                    '파일명': result['filename'],
                    'Tooth_픽셀수': 0,
                    'Root_픽셀수': 0,
                    'Root_비율(%)': 0
                })
            else:
                writer.writerow({
                    '파일명': result['filename'],
                    'Tooth_픽셀수': result['tooth_pixels'],
                    'Root_픽셀수': result['root_pixels'],
                    'Root_비율(%)': result['ratio']
                })
    
    print(f"\n분석 결과가 {csv_filename}에 저장되었습니다.")

def print_summary(results):
    """
    분석 결과 요약을 출력합니다.
    
    Args:
        results (list): 분석 결과 리스트
    """
    if not results:
        return
    
    valid_results = [r for r in results if not r['error']]
    
    if not valid_results:
        print("유효한 분석 결과가 없습니다.")
        return
    
    print("\n" + "="*60)
    print("GT 마스크 분석 결과 요약")
    print("="*60)
    
    total_tooth = sum(r['tooth_pixels'] for r in valid_results)
    total_root = sum(r['root_pixels'] for r in valid_results)
    avg_ratio = sum(r['ratio'] for r in valid_results) / len(valid_results)
    
    print(f"총 분석 파일 수: {len(valid_results)}개")
    print(f"총 Tooth 픽셀 수: {total_tooth:,}")
    print(f"총 Root 픽셀 수: {total_root:,}")
    print(f"평균 Root 비율: {avg_ratio:.2f}%")
    
    # 비율별 분포
    low_ratio = len([r for r in valid_results if r['ratio'] < 5])
    medium_ratio = len([r for r in valid_results if 5 <= r['ratio'] < 15])
    high_ratio = len([r for r in valid_results if 15 <= r['ratio'] < 30])
    very_high_ratio = len([r for r in valid_results if r['ratio'] >= 30])
    
    print(f"\n비율 분포:")
    print(f"  - < 5% (양호): {low_ratio}개")
    print(f"  - 5-15% (관리 필요): {medium_ratio}개")
    print(f"  - 15-30% (주의): {high_ratio}개")
    print(f"  - ≥ 30% (위험): {very_high_ratio}개")
    
    print("="*60)

def main():
    """
    메인 함수 - 명령행 인터페이스
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='GT 마스크 이미지 분석 및 시각화')
    parser.add_argument('folder_path', help='분석할 GT 마스크 폴더 경로')
    parser.add_argument('-o', '--output', help='CSV 출력 파일명')
    parser.add_argument('-v', '--visualize', type=int, default=5, help='시각화할 이미지 수 (기본값: 5)')
    
    args = parser.parse_args()
    
    print(f"GT 마스크 폴더 분석 시작: {args.folder_path}")
    
    # 폴더 분석 실행
    results, output_dir = analyze_gt_folder(args.folder_path, args.output, args.visualize)
    
    if results:
        # 결과 요약 출력
        print_summary(results)
    
    print(f"\n분석 완료! 결과 폴더: {output_dir}")

if __name__ == '__main__':
    main() 