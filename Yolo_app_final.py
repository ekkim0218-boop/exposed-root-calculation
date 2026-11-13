import os
import sys
import cv2
import numpy as np
import torch
import argparse
import csv
from pathlib import Path

# Patch for the Git issue - apply before importing ultralytics
import subprocess
from pathlib import Path

# Monkey patch subprocess.run to handle Git commands
original_run = subprocess.run
def patched_run(args, *argsp, **kwargs):
    # Check if this is a git command
    if isinstance(args, list) and args and args[0] == 'git':
        print("Git command intercepted:", args)
        # Create a dummy successful result
        if args[1:] and args[1] == 'rev-parse':
            class DummyResult:
                returncode = 0
                stdout = b''
            return DummyResult()
        # For any other git command
        raise FileNotFoundError("Git command intercepted and blocked for safety")
    # For non-git commands, use the original function
    return original_run(args, *argsp, **kwargs)

# Apply the patch
subprocess.run = patched_run

# Patch Path class to handle get_git_root_dir
original_path = Path

# Now we can import Ultralytics safely
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
from datetime import datetime

# 설정
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join('models', 'YOLO_best.pt')  # 파인튜닝된 모델 경로

# 모델 로드 (스크립트 시작시 한 번만 로드)
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    device = torch.device('cpu')  # 또는 'cuda'로 설정하여 GPU 사용
    model.to(device)
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")
    print("Will attempt to use a pretrained YOLO model instead.")
    model = YOLO('yolov8n-seg.pt')  # 기본 모델 사용

# 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 치아 및 치석 면적 계산 함수
def calculate_areas(masks, classes):
    # Tooth(class 0)와 Root(class 1)의 면적 계산
    tooth_area = 0
    root_area = 0
    
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            if class_id == 0:  # Tooth 클래스
                tooth_area += np.sum(mask)
            elif class_id == 1:  # Root 클래스
                root_area += np.sum(mask)
    
    # Root 비율 계산 (Tooth 면적이 0인 경우 예외 처리)
    if tooth_area > 0:
        root_ratio = (root_area / tooth_area) * 100
    else:
        root_ratio = 0
    
    return tooth_area, root_area, root_ratio

# 세그멘테이션 마스크 시각화 함수
def visualize_segmentation(image, masks, classes):
    if masks is None or len(masks) == 0:
        return image, None
    
    # 원본 이미지 복사
    result_img = image.copy()
    mask_img = np.zeros_like(image)
    
    # 클래스별 색상 설정 (BGR 형식)
    colors = {
        0: (255, 0, 0),    # Tooth: 파란색
        1: (0, 255, 255)   # Root: 노란색
    }
    
    # 각 마스크 적용
    for i, mask in enumerate(masks):
        class_id = int(classes[i])
        color = colors.get(class_id, (255, 255, 255))  # 기본값: 흰색
        
        # 마스크를 3채널로 확장하여 색상 적용
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = color
        
        # 마스크 이미지에 누적
        mask_img = cv2.addWeighted(mask_img, 1, colored_mask, 0.5, 0)
        
        # 결과 이미지에 마스크 오버레이 - 클래스별 투명도 조정
        if class_id == 1:  # Root 클래스 - 더 높은 투명도
            result_img = cv2.addWeighted(result_img, 1, colored_mask, 0.3, 0)
        else:  # Tooth 클래스 - 기존 투명도
            result_img = cv2.addWeighted(result_img, 1, colored_mask, 0.5, 0)
    
    return result_img, mask_img

# 마스크 OR 연산 함수 (클래스별로 모든 마스크를 하나로 합치기)
def merge_masks_by_class(masks, classes):
    """
    클래스별로 모든 마스크를 OR 연산으로 하나로 합칩니다.
    
    Args:
        masks (list): 마스크 리스트
        classes (list): 클래스 리스트
    
    Returns:
        tuple: (merged_masks, merged_classes)
    """
    if len(masks) == 0:
        return [], []
    
    # 클래스별로 마스크 그룹화
    class_masks = {}
    for mask, class_id in zip(masks, classes):
        class_id = int(class_id)
        if class_id not in class_masks:
            class_masks[class_id] = []
        class_masks[class_id].append(mask)
    
    # 각 클래스별로 마스크들을 OR 연산으로 합치기
    merged_masks = []
    merged_classes = []
    
    for class_id, mask_list in class_masks.items():
        if len(mask_list) == 1:
            # 마스크가 하나만 있는 경우
            merged_masks.append(mask_list[0])
            merged_classes.append(class_id)
        else:
            # 여러 마스크가 있는 경우 OR 연산으로 합치기
            merged_mask = np.zeros_like(mask_list[0], dtype=np.uint8)
            for mask in mask_list:
                merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8)
            merged_masks.append(merged_mask)
            merged_classes.append(class_id)
    
    return merged_masks, merged_classes

def process_image(image_path, output_dir=None):
    """이미지를 처리하고 분석 결과를 반환합니다."""
    # 이미지 로드
    image = cv2.imread(image_path)
    
    if image is None:
        return {"error": "Failed to load image. File may be corrupt or in an unsupported format."}
    
    # 원본 이미지 크기 저장
    original_height, original_width = image.shape[:2]
    
    try:
        # 모델 추론
        abs_path = os.path.abspath(image_path)
        results = model.predict(source=abs_path, conf=0.15, save=False)
    except Exception as e:
        return {"error": f"Error during model inference: {str(e)}"}
    
    # 결과 처리
    result_data = {}
    result_data['original_image'] = os.path.basename(image_path)
    
    if len(results) > 0:
        result = results[0]
        
        # 마스크와 클래스 추출
        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
            try:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # 각 마스크를 원본 이미지 크기로 리사이즈
                resized_masks = []
                for mask in masks:
                    # 마스크를 float32로 변환
                    mask = mask.astype(np.float32)
                    # 리사이즈
                    resized_mask = cv2.resize(mask, (original_width, original_height))
                    # 이진화 (0.5 임계값 사용)
                    resized_mask = (resized_mask > 0.5).astype(np.uint8)
                    resized_masks.append(resized_mask)
                
                # Filter predictions with confidence > 0.25 for 'Tooth' class and > 0.15 for 'calculus' class
                filtered_masks = []
                filtered_classes = []
                tooth_count = 0
                root_count = 0
                
                for i, (mask, class_id) in enumerate(zip(resized_masks, classes)):
                    if (class_id == 0 and result.boxes.conf[i] > 0.40) or (class_id == 1 and result.boxes.conf[i] > 0.22):
                        filtered_masks.append(mask)
                        filtered_classes.append(class_id)
                
                # 클래스별로 마스크들을 OR 연산으로 합치기
                merged_masks, merged_classes = merge_masks_by_class(filtered_masks, filtered_classes)
                
                # 합쳐진 마스크의 클래스별 개수 계산
                for class_id in merged_classes:
                    if class_id == 0:  # Tooth
                        tooth_count += 1
                    elif class_id == 1:  # Root
                        root_count += 1

                # Use merged masks and classes for further processing
                tooth_area, root_area, root_ratio = calculate_areas(merged_masks, merged_classes)

                # Recalculate plaque ratio
                if tooth_area > 0:
                    root_ratio = (root_area / tooth_area) * 100
                else:
                    root_ratio = 0

                result_data['tooth_area'] = tooth_area
                result_data['root_area'] = root_area
                result_data['root_ratio'] = round(root_ratio, 2)
                result_data['tooth_count'] = tooth_count
                result_data['root_count'] = root_count
                
                # Visualize segmentation with merged results
                result_image, _ = visualize_segmentation(image, merged_masks, merged_classes)
                
                # 결과 이미지 저장
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    result_image_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
                    cv2.imwrite(result_image_path, result_image)
                    result_data['result_image'] = result_image_path
                else:
                    # 현재 디렉토리에 저장
                    result_image_path = f"result_{os.path.basename(image_path)}"
                    cv2.imwrite(result_image_path, result_image)
                    result_data['result_image'] = result_image_path

            except Exception as e:
                result_data['error'] = f"Error processing segmentation results: {str(e)}"
        else:
            result_data['error'] = "No segmentation masks found in the results"
    else:
        result_data['error'] = "No detection results"
    
    return result_data

def print_results(results):
    """분석 결과를 콘솔에 출력합니다."""
    print("\n" + "="*50)
    print("치아 및 치근 분석 결과")
    print("="*50)
    
    if 'error' in results:
        print(f"오류: {results['error']}")
        return
    
    print(f"원본 이미지: {results['original_image']}")
    print(f"치아 감지 개수: {results['tooth_count']}개")
    print(f"치근 감지 개수: {results['root_count']}개")
    print(f"치아 감지 영역: {results['tooth_area']:,} 픽셀")
    print(f"치근 감지 영역: {results['root_area']:,} 픽셀")
    print(f"치근 비율: {results['root_ratio']}%")
    
    # 결과 이미지 경로 출력
    if 'result_image' in results:
        print(f"결과 이미지: {results['result_image']}")
    
    # 상태 평가
    print("\n상태 평가:")
    if results['root_ratio'] < 5:
        print("양호: 치근 노출이 적습니다. 치아 상태가 양호합니다.")
    elif results['root_ratio'] < 15:
        print("양호: 치근이 어느 정도 감지되었습니다. 치아 관리를 더 신경 써야 할 수 있습니다.")
    elif results['root_ratio'] < 30:
        print("주의: 치근 노출이 비교적 많이 감지되었습니다. 치과에 방문해서 검진을 받기를 권장합니다.")
    else:
        print("위험: 치근 노출이 매우 높습니다. 가능한 빨리 치과에 방문해서 치료를 받아야 합니다.")
    
    print("="*50)

def save_results_to_csv(results_list, csv_filename):
    """분석 결과들을 CSV 파일로 저장합니다."""
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['원본이미지파일이름', '치아감지영역', '치근감지영역', '치근비율', '치아감지개수', '치근감지개수']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results_list:
            if 'error' in result:
                # 오류가 있는 경우 기본값으로 저장
                writer.writerow({
                    '원본이미지파일이름': result.get('original_image', 'Unknown'),
                    '치아감지영역': 0,
                    '치근감지영역': 0,
                    '치근비율': 0,
                    '치아감지개수': 0,
                    '치근감지개수': 0
                })
            else:
                # 치근 비율 계산 (치근감지영역 / 치아감지영역 * 100)
                tooth_area = result.get('tooth_area', 0)
                root_area = result.get('root_area', 0)
                
                if tooth_area > 0:
                    root_ratio = (root_area / tooth_area) * 100
                else:
                    root_ratio = 0
                
                writer.writerow({
                    '원본이미지파일이름': result.get('original_image', 'Unknown'),
                    '치아감지영역': tooth_area,
                    '치근감지영역': root_area,
                    '치근비율': round(root_ratio, 2),
                    '치아감지개수': result.get('tooth_count', 0),
                    '치근감지개수': result.get('root_count', 0)
                })
    
    print(f"\n분석 결과가 {csv_filename}에 저장되었습니다.")
    print("Excel에서 열 때 한글이 깨지면 '데이터' → '텍스트/CSV'로 열어주세요.")

def process_folder(folder_path, output_dir=None, csv_filename=None):
    """폴더 내 모든 이미지를 분석하고 결과를 CSV로 저장합니다."""
    if not os.path.exists(folder_path):
        print(f"오류: 폴더를 찾을 수 없습니다: {folder_path}")
        return
    
    if not os.path.isdir(folder_path):
        print(f"오류: 지정된 경로가 폴더가 아닙니다: {folder_path}")
        return
    
    # 폴더 내 이미지 파일 찾기
    image_files = []
    for file in os.listdir(folder_path):
        if allowed_file(file):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"오류: 폴더에 지원하는 이미지 파일이 없습니다: {folder_path}")
        return
    
    print(f"폴더에서 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    
    # 결과 이미지 저장 폴더 생성
    if output_dir:
        result_images_dir = output_dir
    else:
        # 기본 결과 이미지 폴더명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_images_dir = f"result_images_{timestamp}"
    
    os.makedirs(result_images_dir, exist_ok=True)
    print(f"결과 이미지 저장 폴더: {result_images_dir}")
    
    # 각 이미지 분석
    results_list = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 분석 중: {os.path.basename(image_path)}")
        result = process_image(image_path, result_images_dir)
        results_list.append(result)
        
        # 간단한 진행 상황 출력
        if 'error' in result:
            print(f"  - 오류: {result['error']}")
        else:
            print(f"  - 치아 영역: {result.get('tooth_area', 0):,} 픽셀")
            print(f"  - 치근 영역: {result.get('root_area', 0):,} 픽셀")
            print(f"  - 치근 비율: {result.get('root_ratio', 0)}%")
    
    # CSV 파일로 저장
    if csv_filename:
        save_results_to_csv(results_list, csv_filename)
    else:
        # 기본 CSV 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_csv_filename = f"analysis_results_{timestamp}.csv"
        save_results_to_csv(results_list, default_csv_filename)

def main():
    parser = argparse.ArgumentParser(description='치아 및 치근 분석 시스템')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('image_path', nargs='?', help='분석할 이미지 파일 경로')
    group.add_argument('-f', '--folder', help='분석할 폴더 경로')
    parser.add_argument('-o', '--output', help='결과 이미지 저장 디렉토리')
    parser.add_argument('-c', '--csv', help='CSV 결과 파일명')
    
    args = parser.parse_args()
    
    # 폴더 분석 모드
    if args.folder:
        process_folder(args.folder, args.output, args.csv)
        return
    
    # 단일 이미지 분석 모드
    image_path = args.image_path
    
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"오류: 파일을 찾을 수 없습니다: {image_path}")
        return
    
    # 파일 확장자 확인
    if not allowed_file(image_path):
        print(f"오류: 지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(ALLOWED_EXTENSIONS)}")
        return
    
    print(f"이미지 분석 중: {image_path}")
    
    # 이미지 처리
    results = process_image(image_path, args.output)
    
    # 결과 출력
    print_results(results)

if __name__ == '__main__':
    # 모델 폴더가 없으면 생성
    os.makedirs('models', exist_ok=True)
    
    main()