import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 개선
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16  # 기본 폰트 크기 증가
plt.rcParams['axes.titlesize'] = 18  # 축 제목 폰트 크기 증가
plt.rcParams['axes.labelsize'] = 16  # 축 라벨 폰트 크기 증가
plt.rcParams['xtick.labelsize'] = 14  # x축 눈금 라벨 폰트 크기 증가
plt.rcParams['ytick.labelsize'] = 14  # y축 눈금 라벨 폰트 크기 증가
plt.rcParams['legend.fontsize'] = 14  # 범례 폰트 크기 증가

def load_and_explore_data(file_path):
    """데이터 로드 및 기본 탐색"""
    df = pd.read_csv(file_path)
    
    # Class 값 정규화 (대소문자 통일)
    df['Class'] = df['Class'].str.title()
    
    # Metric 이름 통일
    metric_mapping = {
        'acc': 'accuracy',
        'spec': 'specificity', 
        'rec': 'recall',
        'prec': 'precision'
    }
    
    df['Metric'] = df['Metric'].replace(metric_mapping)
    
    print("=== 데이터 기본 정보 ===")
    print(f"데이터 크기: {df.shape}")
    print(f"\n컬럼 정보:")
    print(df.info())
    print(f"\n첫 5행:")
    print(df.head())
    print(f"\n기술통계:")
    print(df.describe())
    print(f"\nClass 값들: {df['Class'].unique()}")
    print(f"\nMetric 값들: {df['Metric'].unique()}")
    return df

def calculate_p_value(unet_mean, yolo_mean, unet_ci_lower, unet_ci_upper, yolo_ci_lower, yolo_ci_upper):
    """신뢰구간을 이용한 p-value 계산 (근사치)"""
    # 신뢰구간의 표준오차 계산
    unet_se = (unet_ci_upper - unet_ci_lower) / (2 * 1.96)  # 95% CI 기준
    yolo_se = (yolo_ci_upper - yolo_ci_lower) / (2 * 1.96)
    
    # 통합 표준오차
    pooled_se = np.sqrt(unet_se**2 + yolo_se**2)
    
    # t-통계량 계산
    t_stat = (yolo_mean - unet_mean) / pooled_se
    
    # 자유도 (근사치)
    df = 30  # 일반적인 샘플 크기 가정
    
    # p-value 계산 (양측 검정)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return p_value

def perform_statistical_analysis(df):
    """Model 간 Class별 metrics에 대한 통계분석"""
    print("\n" + "="*60)
    print("MODEL 간 CLASS별 METRICS 통계분석")
    print("="*60)
    
    # Class와 Metric 조합별로 분석
    classes = df['Class'].unique()
    metrics = df['Metric'].unique()
    
    results = []
    
    for class_name in classes:
        for metric in metrics:
            # 해당 Class와 Metric의 데이터 추출
            subset = df[(df['Class'] == class_name) & (df['Metric'] == metric)]
            
            if len(subset) == 2:  # 두 모델 모두 있는 경우
                unet_data = subset[subset['Model'] == 'Unet']
                yolo_data = subset[subset['Model'] == 'Yolo']
                
                unet_mean = unet_data['Mean'].values[0]
                yolo_mean = yolo_data['Mean'].values[0]
                
                # 차이 계산
                difference = yolo_mean - unet_mean
                percent_change = (difference / unet_mean) * 100
                
                # 신뢰구간 비교
                unet_ci_lower = unet_data['CI_Lower'].values[0]
                unet_ci_upper = unet_data['CI_Upper'].values[0]
                yolo_ci_lower = yolo_data['CI_Lower'].values[0]
                yolo_ci_upper = yolo_data['CI_Upper'].values[0]
                
                # 신뢰구간 겹침 확인
                ci_overlap = not (unet_ci_upper < yolo_ci_lower or yolo_ci_upper < unet_ci_lower)
                
                # p-value 계산
                p_value = calculate_p_value(unet_mean, yolo_mean, 
                                         unet_ci_lower, unet_ci_upper,
                                         yolo_ci_lower, yolo_ci_upper)
                
                # 통계적 유의성 판단
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                
                results.append({
                    'Class': class_name,
                    'Metric': metric,
                    'Unet_Mean': unet_mean,
                    'Yolo_Mean': yolo_mean,
                    'Difference': difference,
                    'Percent_Change': percent_change,
                    'CI_Overlap': ci_overlap,
                    'P_Value': p_value,
                    'Significance': significance,
                    'Better_Model': 'Yolo' if yolo_mean > unet_mean else 'Unet'
                })
    
    results_df = pd.DataFrame(results)
    return results_df

def print_detailed_analysis(results_df):
    """상세 분석 결과 출력"""
    print("\n=== 상세 분석 결과 ===")
    
    # 전체 성능 비교
    print("\n1. 전체 성능 비교 (Overall Class)")
    overall_results = results_df[results_df['Class'] == 'Overall']
    if len(overall_results) > 0:
        display_cols = ['Metric', 'Unet_Mean', 'Yolo_Mean', 'Difference', 'Percent_Change', 'P_Value', 'Significance', 'Better_Model']
        print(overall_results[display_cols].round(3))
    else:
        print("Overall Class 데이터가 없습니다.")
    
    # Root Class 분석
    print("\n2. Root Class 분석")
    root_results = results_df[results_df['Class'] == 'Root']
    if len(root_results) > 0:
        display_cols = ['Metric', 'Unet_Mean', 'Yolo_Mean', 'Difference', 'Percent_Change', 'P_Value', 'Significance', 'Better_Model']
        print(root_results[display_cols].round(3))
    else:
        print("Root Class 데이터가 없습니다.")
    
    # Tooth Class 분석
    print("\n3. Tooth Class 분석")
    tooth_results = results_df[results_df['Class'] == 'Tooth']
    if len(tooth_results) > 0:
        display_cols = ['Metric', 'Unet_Mean', 'Yolo_Mean', 'Difference', 'Percent_Change', 'P_Value', 'Significance', 'Better_Model']
        print(tooth_results[display_cols].round(3))
    else:
        print("Tooth Class 데이터가 없습니다.")
    
    # 모델별 우수성 통계
    print("\n4. 모델별 우수성 통계")
    better_model_counts = results_df['Better_Model'].value_counts()
    print(better_model_counts)
    
    # 신뢰구간 겹침 통계
    print("\n5. 신뢰구간 겹침 통계")
    ci_overlap_stats = results_df['CI_Overlap'].value_counts()
    print(ci_overlap_stats)
    
    # 통계적 유의성 통계
    print("\n6. 통계적 유의성 통계")
    significance_stats = results_df['Significance'].value_counts()
    print(significance_stats)
    
    return overall_results, root_results, tooth_results

def create_visualizations(df, results_df):
    """시각화 생성"""
    # DPI 설정으로 선명도 향상 - 해상도 증가
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Model Comparison Analysis by Class and Metrics', fontsize=22, fontweight='bold')  # 메인 제목 폰트 크기 증가
    
    # 1. Class별 평균 성능 비교 (Heatmap)
    try:
        pivot_data = df.pivot_table(values='Mean', index='Class', columns=['Model', 'Metric'], aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0,0], 
                   cbar_kws={'label': 'Mean Value'}, annot_kws={'size': 12})  # 히트맵 주석 폰트 크기 증가
        axes[0,0].set_title('Class vs Metrics Average Performance', fontsize=18, fontweight='bold')  # 제목 폰트 크기 증가
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=14)  # x축 눈금 라벨 폰트 크기 증가
        axes[0,0].tick_params(axis='y', labelsize=16)  # y축 눈금 라벨 폰트 크기 증가
    except Exception as e:
        axes[0,0].text(0.5, 0.5, 'Heatmap generation failed', ha='center', va='center', fontsize=18)  # 에러 메시지 폰트 크기 증가
        axes[0,0].set_title('Class vs Metrics Average Performance', fontsize=18, fontweight='bold')
    
    # 2. Model별 성능 비교 (Bar plot)
    try:
        model_performance = df.groupby(['Model', 'Class'])['Mean'].mean().unstack()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 더 선명한 색상
        model_performance.plot(kind='bar', ax=axes[0,1], color=colors, width=0.8)
        axes[0,1].set_title('Model Performance by Class', fontsize=18, fontweight='bold')  # 제목 폰트 크기 증가
        axes[0,1].set_ylabel('Average Performance', fontsize=16)  # y축 라벨 폰트 크기 증가
        axes[0,1].legend(title='Class', fontsize=18, title_fontsize=20)  # 범례 폰트 크기 증가
        axes[0,1].tick_params(axis='x', rotation=45, labelsize=16)  # x축 눈금 라벨 폰트 크기 증가
        axes[0,1].grid(axis='y', alpha=0.3)
    except Exception as e:
        axes[0,1].text(0.5, 0.5, 'Bar plot generation failed', ha='center', va='center', fontsize=18)  # 에러 메시지 폰트 크기 증가
        axes[0,1].set_title('Model Performance by Class', fontsize=18, fontweight='bold')
    
    # 3. Metrics별 성능 차이
    try:
        metric_diff = results_df.pivot(index='Metric', columns='Class', values='Difference')
        colors = ['#ff7f0e', '#2ca02c', '#d62728']  # 더 선명한 색상
        metric_diff.plot(kind='bar', ax=axes[1,0], color=colors, width=0.8)
        axes[1,0].set_title('Performance Difference (Yolo - Unet)', fontsize=18, fontweight='bold')  # 제목 폰트 크기 증가
        axes[1,0].set_ylabel('Performance Difference', fontsize=16)  # y축 라벨 폰트 크기 증가
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=16)  # x축 눈금 라벨 폰트 크기 증가
        axes[1,0].legend(fontsize=18)  # 범례 폰트 크기 증가
        axes[1,0].grid(axis='y', alpha=0.3)
    except Exception as e:
        axes[1,0].text(0.5, 0.5, 'Metrics difference plot failed', ha='center', va='center', fontsize=18)  # 에러 메시지 폰트 크기 증가
        axes[1,0].set_title('Performance Difference (Yolo - Unet)', fontsize=18, fontweight='bold')
    
    # 4. Class별 성능 차이 분포
    try:
        class_diff = results_df.groupby('Class')['Difference'].mean()
        colors = ['#9467bd', '#8c564b', '#e377c2']  # 더 선명한 색상
        class_diff.plot(kind='bar', ax=axes[1,1], color=colors, width=0.6)
        axes[1,1].set_title('Average Performance Difference by Class', fontsize=18, fontweight='bold')  # 제목 폰트 크기 증가
        axes[1,1].set_ylabel('Average Performance Difference', fontsize=16)  # y축 라벨 폰트 크기 증가
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=16)  # x축 눈금 라벨 폰트 크기 증가
        axes[1,1].grid(axis='y', alpha=0.3)
    except Exception as e:
        axes[1,1].text(0.5, 0.5, 'Class difference plot failed', ha='center', va='center', fontsize=18)  # 에러 메시지 폰트 크기 증가
        axes[1,1].set_title('Average Performance Difference by Class', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.show()

def generate_summary_report(results_df):
    """요약 리포트 생성"""
    print("\n" + "="*60)
    print("통계분석 요약 리포트")
    print("="*60)
    
    if len(results_df) == 0:
        print("분석할 데이터가 없습니다.")
        return
    
    # 주요 발견사항
    print("\n📊 주요 발견사항:")
    
    # 1. 전체적으로 더 나은 모델
    overall_data = results_df[results_df['Class'] == 'Overall']
    if len(overall_data) > 0:
        overall_better = overall_data['Better_Model'].mode()[0]
        print(f"• Overall Class에서 더 나은 모델: {overall_better}")
    else:
        print("• Overall Class 데이터가 없습니다.")
    
    # 2. 가장 큰 성능 차이
    if len(results_df) > 0:
        max_diff_idx = results_df['Difference'].abs().idxmax()
        max_diff_row = results_df.loc[max_diff_idx]
        print(f"• 가장 큰 성능 차이: {max_diff_row['Class']} - {max_diff_row['Metric']}")
        print(f"  차이값: {max_diff_row['Difference']:.3f} ({max_diff_row['Percent_Change']:.2f}%)")
        print(f"  p-value: {max_diff_row['P_Value']:.3f} ({max_diff_row['Significance']})")
    
    # 3. 신뢰구간 겹침 비율
    if len(results_df) > 0:
        overlap_ratio = results_df['CI_Overlap'].mean() * 100
        print(f"• 신뢰구간 겹침 비율: {overlap_ratio:.1f}%")
    
    # 4. 통계적 유의성 비율
    if len(results_df) > 0:
        significant_ratio = (results_df['Significance'] == 'Significant').mean() * 100
        print(f"• 통계적으로 유의한 차이 비율: {significant_ratio:.1f}%")
    
    # 5. Class별 성능 패턴
    print("\n📈 Class별 성능 패턴:")
    for class_name in results_df['Class'].unique():
        class_data = results_df[results_df['Class'] == class_name]
        if len(class_data) > 0:
            better_model = class_data['Better_Model'].mode()[0]
            avg_diff = class_data['Difference'].mean()
            significant_count = (class_data['Significance'] == 'Significant').sum()
            print(f"• {class_name}: {better_model} 우세 (평균 차이: {avg_diff:.3f}, 유의한 차이: {significant_count}/{len(class_data)})")
    
    # 6. Metrics별 성능 패턴
    print("\n📊 Metrics별 성능 패턴:")
    for metric in results_df['Metric'].unique():
        metric_data = results_df[results_df['Metric'] == metric]
        if len(metric_data) > 0:
            better_model = metric_data['Better_Model'].mode()[0]
            avg_diff = metric_data['Difference'].mean()
            significant_count = (metric_data['Significance'] == 'Significant').sum()
            print(f"• {metric}: {better_model} 우세 (평균 차이: {avg_diff:.3f}, 유의한 차이: {significant_count}/{len(metric_data)})")

def main():
    """메인 실행 함수"""
    # 데이터 로드
    df = load_and_explore_data('test_Total_metrics.csv')
    
    # 통계분석 수행
    results_df = perform_statistical_analysis(df)
    
    # 상세 분석 결과 출력
    overall_results, root_results, tooth_results = print_detailed_analysis(results_df)
    
    # 시각화 생성
    create_visualizations(df, results_df)
    
    # 요약 리포트 생성
    generate_summary_report(results_df)
    
    # 결과를 CSV로 저장 (소수점 3자리까지)
    if len(results_df) > 0:
        # 수치 컬럼들을 소수점 3자리까지 반올림
        numeric_columns = ['Unet_Mean', 'Yolo_Mean', 'Difference', 'Percent_Change', 'P_Value']
        results_df[numeric_columns] = results_df[numeric_columns].round(3)
        
        results_df.to_csv('statistical_analysis_results.csv', index=False)
        print(f"\n✅ 분석 결과가 'statistical_analysis_results.csv'에 저장되었습니다.")
    print(f"✅ 시각화가 'model_comparison_analysis.png'에 저장되었습니다.")

if __name__ == "__main__":
    main() 