import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

def analyze_demographics(excel_file_path, output_dir='Result'):
    """
    Excel 파일에서 대상자의 나이와 성별에 대한 평균과 표준편차를 계산합니다.
    
    Args:
        excel_file_path (str): Excel 파일 경로
        output_dir (str): 결과 저장 디렉토리
    """
    
    # Result 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Excel 파일 읽기
        print(f"Excel 파일을 읽는 중: {excel_file_path}")
        df = pd.read_excel(excel_file_path)
        
        print(f"데이터 형태: {df.shape}")
        print(f"컬럼명: {list(df.columns)}")
        
        # 데이터 기본 정보 출력
        print("\n데이터 기본 정보:")
        print(df.info())
        print("\n처음 5행:")
        print(df.head())
        
        # 나이와 성별 컬럼 찾기 (대소문자 구분 없이)
        age_columns = [col for col in df.columns if 'age' in col.lower() or '나이' in col or '연령' in col]
        gender_columns = [col for col in df.columns if 'gender' in col.lower() or 'sex' in col.lower() or '성별' in col]
        
        print(f"\n발견된 나이 관련 컬럼: {age_columns}")
        print(f"발견된 성별 관련 컬럼: {gender_columns}")
        
        # 나이 컬럼이 없으면 첫 번째 숫자 컬럼을 사용
        if not age_columns:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                age_columns = [numeric_columns[0]]
                print(f"나이 컬럼을 찾을 수 없어 첫 번째 숫자 컬럼을 사용: {age_columns[0]}")
        
        # 성별 컬럼이 없으면 첫 번째 문자열 컬럼을 사용
        if not gender_columns:
            string_columns = df.select_dtypes(include=['object']).columns
            if len(string_columns) > 0:
                gender_columns = [string_columns[0]]
                print(f"성별 컬럼을 찾을 수 없어 첫 번째 문자열 컬럼을 사용: {gender_columns[0]}")
        
        if not age_columns or not gender_columns:
            print("나이 또는 성별 컬럼을 찾을 수 없습니다.")
            return
        
        age_col = age_columns[0]
        gender_col = gender_columns[0]
        
        print(f"\n사용할 컬럼:")
        print(f"나이: {age_col}")
        print(f"성별: {gender_col}")
        
        # 데이터 전처리
        # 나이 데이터 정리 (숫자가 아닌 값 제거)
        df_clean = df.copy()
        df_clean[age_col] = pd.to_numeric(df_clean[age_col], errors='coerce')
        df_clean = df_clean.dropna(subset=[age_col])
        
        # 성별 데이터 정리 및 매핑
        df_clean[gender_col] = df_clean[gender_col].astype(str).str.strip()
        
        # 성별 숫자를 의미있는 텍스트로 변환
        gender_mapping = {
            '0': '여성',
            '1': '남성',
            'F': '여성',
            'M': '남성',
            'female': '여성',
            'male': '남성',
            '여성': '여성',
            '남성': '남성'
        }
        
        df_clean['성별_텍스트'] = df_clean[gender_col].map(gender_mapping).fillna(df_clean[gender_col])
        
        print(f"\n전처리 후 데이터 수: {len(df_clean)}")
        print(f"성별 분포: {df_clean['성별_텍스트'].value_counts().to_dict()}")
        
        # 전체 통계
        total_stats = {
            '총 대상자 수': len(df_clean),
            '평균 나이': df_clean[age_col].mean(),
            '나이 표준편차': df_clean[age_col].std(),
            '최소 나이': df_clean[age_col].min(),
            '최대 나이': df_clean[age_col].max(),
            '중앙값 나이': df_clean[age_col].median()
        }
        
        # 성별별 통계
        gender_stats = df_clean.groupby('성별_텍스트')[age_col].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        
        gender_stats.columns = ['인원수', '평균나이', '표준편차', '최소나이', '최대나이', '중앙값']
        
        # 나이 구간별 통계
        age_bins = [0, 40, 50, 60, 70, 100]
        age_labels = ['40세 미만', '40-49세', '50-59세', '60-69세', '70세 이상']
        df_clean['나이구간'] = pd.cut(df_clean[age_col], bins=age_bins, labels=age_labels, right=False)
        
        age_group_stats = df_clean.groupby('나이구간')[age_col].agg([
            'count', 'mean', 'std'
        ]).round(2)
        
        age_group_stats.columns = ['인원수', '평균나이', '표준편차']
        
        # 성별별 나이 구간 분포
        gender_age_cross = pd.crosstab(df_clean['성별_텍스트'], df_clean['나이구간'], margins=True)
        
        # 마크다운 파일 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = f"demographics_analysis_{timestamp}.md"
        md_path = os.path.join(output_dir, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 치아 이미지 분석 대상자 인구통계학적 특성 분석\n\n")
            f.write(f"**분석 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n\n")
            f.write(f"**데이터 소스**: {os.path.basename(excel_file_path)}\n\n")
            f.write(f"**총 대상자 수**: {len(df_clean):,}명\n\n")
            
            f.write("## 1. 전체 대상자 특성\n\n")
            f.write("| 항목 | 값 |\n")
            f.write("|------|----|\n")
            f.write(f"| 총 대상자 수 | {total_stats['총 대상자 수']:,}명 |\n")
            f.write(f"| 평균 나이 | {total_stats['평균 나이']:.1f} ± {total_stats['나이 표준편차']:.1f}세 |\n")
            f.write(f"| 중앙값 나이 | {total_stats['중앙값 나이']:.1f}세 |\n")
            f.write(f"| 나이 범위 | {total_stats['최소 나이']:.0f} - {total_stats['최대 나이']:.0f}세 |\n\n")
            
            f.write("## 2. 성별별 특성\n\n")
            f.write("| 성별 | 인원수 | 비율(%) | 평균나이 | 표준편차 | 최소나이 | 최대나이 | 중앙값 |\n")
            f.write("|------|--------|---------|----------|----------|----------|----------|--------|\n")
            
            for gender, row in gender_stats.iterrows():
                ratio = (row['인원수'] / total_stats['총 대상자 수']) * 100
                f.write(f"| {gender} | {row['인원수']:,}명 | {ratio:.1f}% | {row['평균나이']:.1f}세 | {row['표준편차']:.1f} | {row['최소나이']:.0f}세 | {row['최대나이']:.0f}세 | {row['중앙값']:.1f}세 |\n")
            
            f.write("\n")
            
            f.write("## 3. 나이 구간별 분포\n\n")
            f.write("| 나이 구간 | 인원수 | 비율(%) | 평균나이 | 표준편차 |\n")
            f.write("|-----------|--------|---------|----------|----------|\n")
            
            for age_group, row in age_group_stats.iterrows():
                if pd.notna(age_group):
                    ratio = (row['인원수'] / total_stats['총 대상자 수']) * 100
                    f.write(f"| {age_group} | {row['인원수']:,}명 | {ratio:.1f}% | {row['평균나이']:.1f}세 | {row['표준편차']:.1f} |\n")
            
            f.write("\n")
            
            f.write("## 4. 성별별 나이 구간 분포\n\n")
            f.write("| 성별 | 40세 미만 | 40-49세 | 50-59세 | 60-69세 | 70세 이상 | 전체 |\n")
            f.write("|------|-----------|---------|---------|---------|-----------|------|\n")
            
            for gender in gender_stats.index:
                row_data = []
                row_data.append(gender)
                for age_group in age_labels:
                    count = gender_age_cross.loc[gender, age_group] if age_group in gender_age_cross.columns else 0
                    row_data.append(f"{count}명")
                total_count = gender_age_cross.loc[gender, 'All']
                row_data.append(f"{total_count}명")
                f.write("| " + " | ".join(row_data) + " |\n")
            
            f.write("\n")
            
            f.write("## 5. 데이터 품질 정보\n\n")
            f.write(f"- **원본 데이터 수**: {len(df):,}개\n")
            f.write(f"- **유효 데이터 수**: {len(df_clean):,}개\n")
            f.write(f"- **데이터 손실률**: {((len(df) - len(df_clean)) / len(df) * 100):.1f}%\n")
            f.write(f"- **사용된 나이 컬럼**: {age_col}\n")
            f.write(f"- **사용된 성별 컬럼**: {gender_col}\n")
            f.write(f"- **성별 매핑**: {gender_mapping}\n\n")
            
            f.write("## 6. 분석 메모\n\n")
            f.write("- 나이 데이터는 숫자형으로 변환하여 분석했습니다.\n")
            f.write("- 성별 데이터는 숫자에서 의미있는 텍스트로 변환했습니다.\n")
            f.write("- 결측값이 있는 행은 분석에서 제외했습니다.\n")
            f.write("- 나이 구간은 10세 단위로 구분했습니다.\n")
            f.write("- 성별별 나이 구간 분포를 교차 분석했습니다.\n\n")
            
            f.write("---\n")
            f.write("*이 보고서는 자동으로 생성되었습니다.*\n")
        
        print(f"\n분석 완료! 결과가 저장되었습니다: {md_path}")
        
        # 콘솔에도 요약 출력
        print("\n" + "="*60)
        print("분석 결과 요약")
        print("="*60)
        print(f"총 대상자 수: {total_stats['총 대상자 수']:,}명")
        print(f"평균 나이: {total_stats['평균 나이']:.1f} ± {total_stats['나이 표준편차']:.1f}세")
        print(f"나이 범위: {total_stats['최소 나이']:.0f} - {total_stats['최대 나이']:.0f}세")
        print("\n성별별 분포:")
        for gender, row in gender_stats.iterrows():
            ratio = (row['인원수'] / total_stats['총 대상자 수']) * 100
            print(f"  {gender}: {row['인원수']:,}명 ({ratio:.1f}%) - {row['평균나이']:.1f} ± {row['표준편차']:.1f}세")
        
        return md_path
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

def main():
    """메인 함수"""
    # Excel 파일 경로
    excel_file_path = r"C:\Users\HP\Exposed-Root-Calculation\Data\merged_panorama_analysis_over35.xlsx"
    
    # 파일 존재 확인
    if not os.path.exists(excel_file_path):
        print(f"오류: 파일을 찾을 수 없습니다: {excel_file_path}")
        return
    
    # 분석 실행
    result_path = analyze_demographics(excel_file_path)
    
    if result_path:
        print(f"\n분석이 성공적으로 완료되었습니다!")
        print(f"결과 파일: {result_path}")
    else:
        print("\n분석 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 