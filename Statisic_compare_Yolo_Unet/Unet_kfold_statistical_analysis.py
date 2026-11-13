

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, kruskal
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_and_explore_data(file_path):
    """데이터 로드 및 기본 탐색"""
    df = pd.read_csv(file_path)
    
    print("=== K-Fold 데이터 기본 정보 ===")
    print(f"데이터 크기: {df.shape}")
    print(f"\n컬럼 정보:")
    print(df.info())
    print(f"\n첫 3행:")
    print(df.head(3))
    print(f"\n기술통계:")
    print(df.describe())
    print(f"\nFold 값들: {df['fold'].unique()}")
    
    return df

def extract_metrics_by_class(df):
    """Class별로 메트릭을 분리"""
    # Overall 메트릭
    overall_cols = [col for col in df.columns if col.startswith('Overall_')]
    overall_metrics = df[['fold'] + overall_cols].copy()
    
    # Root 메트릭
    root_cols = [col for col in df.columns if col.startswith('Root_')]
    root_metrics = df[['fold'] + root_cols].copy()
    
    # Tooth 메트릭
    tooth_cols = [col for col in df.columns if col.startswith('Tooth_')]
    tooth_metrics = df[['fold'] + tooth_cols].copy()
    
    return {
        'Overall': overall_metrics,
        'Root': root_metrics,
        'Tooth': tooth_metrics
    }

def perform_statistical_tests(df):
    """각 메트릭에 대해 통계적 검정 수행"""
    results = []
    
    # fold 컬럼 제외한 모든 메트릭
    metric_cols = [col for col in df.columns if col != 'fold']
    
    for metric in metric_cols:
        # 각 fold의 값들
        fold_values = df[metric].values
        
        # 기술통계
        mean_val = np.mean(fold_values)
        std_val = np.std(fold_values)
        min_val = np.min(fold_values)
        max_val = np.max(fold_values)
        cv = std_val / mean_val * 100  # 변동계수
        
        # 정규성 검정 (Shapiro-Wilk test)
        try:
            shapiro_stat, shapiro_p = stats.shapiro(fold_values)
            is_normal = shapiro_p > 0.05
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan
            is_normal = False
        
        # 일원분산분석 (One-way ANOVA) - 정규성 만족시
        if is_normal:
            f_stat, anova_p = stats.f_oneway(*[df[df['fold'] == fold][metric].values for fold in df['fold'].unique()])
            test_type = "One-way ANOVA"
        else:
            # Kruskal-Wallis H-test - 정규성 불만족시
            try:
                h_stat, kruskal_p = stats.kruskal(*[df[df['fold'] == fold][metric].values for fold in df['fold'].unique()])
                f_stat, anova_p = h_stat, kruskal_p
                test_type = "Kruskal-Wallis H-test"
            except:
                f_stat, anova_p = np.nan, np.nan
                test_type = "Test failed"
        
        # 효과 크기 (Eta-squared for ANOVA, Epsilon-squared for Kruskal-Wallis)
        if not np.isnan(f_stat):
            if test_type == "One-way ANOVA":
                # Eta-squared 계산
                ss_between = f_stat * (len(fold_values) - len(df['fold'].unique())) / (len(df['fold'].unique()) - 1)
                ss_total = np.var(fold_values) * (len(fold_values) - 1)
                effect_size = ss_between / ss_total if ss_total != 0 else 0
            else:
                # Epsilon-squared 계산 (Kruskal-Wallis용)
                n = len(fold_values)
                effect_size = (f_stat - len(df['fold'].unique()) + 1) / (n - len(df['fold'].unique()))
        else:
            effect_size = np.nan
        
        # 유의성 판단
        significance = "Significant" if anova_p < 0.05 else "Not Significant"
        
        results.append({
            'Metric': metric,
            'Mean': mean_val,
            'Std': std_val,
            'Min': min_val,
            'Max': max_val,
            'CV(%)': cv,
            'Test_Type': test_type,
            'F_Statistic': f_stat,
            'P_Value': anova_p,
            'Effect_Size': effect_size,
            'Significance': significance,
            'Is_Normal': is_normal,
            'Shapiro_P': shapiro_p
        })
    
    return pd.DataFrame(results)

def print_detailed_analysis(results_df):
    """상세 분석 결과 출력"""
    print("\n=== 상세 분석 결과 ===")
    
    # Class별로 결과 분류 (Overall 제거하고 Tooth가 먼저 나오도록 순서 변경)
    classes = ['Tooth', 'Root']
    
    for class_name in classes:
        class_results = results_df[results_df['Metric'].str.startswith(class_name)]
        if len(class_results) > 0:
            print(f"\n{class_name} Class 분석:")
            display_cols = ['Metric', 'Mean', 'Std', 'CV(%)', 'Test_Type', 'P_Value', 'Effect_Size', 'Significance']
            print(class_results[display_cols].round(4))
    
    # 전체 통계 요약
    print("\n=== 전체 통계 요약 ===")
    print(f"총 메트릭 수: {len(results_df)}")
    print(f"통계적으로 유의한 차이: {(results_df['Significance'] == 'Significant').sum()}")
    print(f"유의하지 않은 차이: {(results_df['Significance'] == 'Not Significant').sum()}")
    print(f"정규성 만족: {results_df['Is_Normal'].sum()}")
    print(f"정규성 불만족: {(~results_df['Is_Normal']).sum()}")
    
    # 평균 변동계수
    mean_cv = results_df['CV(%)'].mean()
    print(f"평균 변동계수: {mean_cv:.2f}%")
    
    return results_df

def create_visualizations(df, results_df):
    """시각화 생성 - 폴드별 성능지표 개선"""
    # DPI 설정
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # Background 제외하고 시각화할 데이터 필터링 (Overall 포함)
    filtered_results = results_df[~results_df['Metric'].str.startswith('Background')]
    
    # 더 큰 그래프 크기로 설정
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('UNet K-Fold Cross-Validation Performance Analysis', fontsize=40, fontweight='bold')
    
    # 1. 폴드별 성능 변화 (Tooth 클래스)
    tooth_metrics = ['Tooth_f1', 'Tooth_iou', 'Tooth_prec', 'Tooth_rec']
    tooth_data = df[['fold'] + tooth_metrics].melt(id_vars=['fold'], var_name='Metric', value_name='Value')
    tooth_data['Metric_Type'] = tooth_data['Metric'].str.split('_').str[1]
    
    for i, metric in enumerate(['f1', 'iou', 'prec', 'rec']):
        metric_data = tooth_data[tooth_data['Metric_Type'] == metric]
        axes[0,0].plot(metric_data['fold'], metric_data['Value'], marker='o', linewidth=2, 
                      label=metric.upper(), markersize=8)
    
    axes[0,0].set_title('Tooth Class Performance by Fold', fontsize=32, fontweight='bold')
    axes[0,0].set_xlabel('Fold', fontsize=28)
    axes[0,0].set_ylabel('Performance Value', fontsize=28)
    axes[0,0].set_xticks([1, 2, 3, 4, 5])
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(fontsize=24)
    axes[0,0].tick_params(labelsize=24)
    
    # 2. 폴드별 성능 변화 (Root 클래스)
    root_metrics = ['Root_f1', 'Root_iou', 'Root_prec', 'Root_rec']
    root_data = df[['fold'] + root_metrics].melt(id_vars=['fold'], var_name='Metric', value_name='Value')
    root_data['Metric_Type'] = root_data['Metric'].str.split('_').str[1]
    
    for i, metric in enumerate(['f1', 'iou', 'prec', 'rec']):
        metric_data = root_data[root_data['Metric_Type'] == metric]
        axes[0,1].plot(metric_data['fold'], metric_data['Value'], marker='s', linewidth=2, 
                      label=metric.upper(), markersize=8)
    
    axes[0,1].set_title('Root Class Performance by Fold', fontsize=32, fontweight='bold')
    axes[0,1].set_xlabel('Fold', fontsize=28)
    axes[0,1].set_ylabel('Performance Value', fontsize=28)
    axes[0,1].set_xticks([1, 2, 3, 4, 5])
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend(fontsize=24)
    axes[0,1].tick_params(labelsize=24)
    
    # 3. 변동계수 비교 (Box Plot)
    cv_data = filtered_results[['Metric', 'CV(%)']].copy()
    cv_data['Class'] = cv_data['Metric'].str.split('_').str[0]
    cv_data['Metric_Type'] = cv_data['Metric'].str.split('_').str[1]
    
    class_order = ['Tooth', 'Root', 'Overall']
    cv_data['Class'] = pd.Categorical(cv_data['Class'], categories=class_order, ordered=True)
    cv_data = cv_data.sort_values('Class')
    
    sns.boxplot(data=cv_data, x='Class', y='CV(%)', ax=axes[0,2], order=class_order)
    axes[0,2].set_title('Coefficient of Variation by Class', fontsize=32, fontweight='bold')
    axes[0,2].set_ylabel('CV (%)', fontsize=28)
    axes[0,2].set_xlabel('Class', fontsize=28)
    axes[0,2].tick_params(labelsize=24)
    
    # 4. 폴드별 평균 성능 비교 (Heatmap)
    performance_data = []
    for fold in df['fold']:
        fold_data = df[df['fold'] == fold]
        tooth_avg = fold_data[tooth_metrics].mean(axis=1).values[0]
        root_avg = fold_data[root_metrics].mean(axis=1).values[0]
        overall_avg = fold_data[['Overall_f1', 'Overall_iou', 'Overall_prec', 'Overall_rec']].mean(axis=1).values[0]
        performance_data.append([tooth_avg, root_avg, overall_avg])
    
    performance_df = pd.DataFrame(performance_data, 
                                index=[f'Fold {i}' for i in df['fold']], 
                                columns=['Tooth', 'Root', 'Overall'])
    
    sns.heatmap(performance_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,0], 
                cbar_kws={'label': 'Average Performance'}, annot_kws={'size': 20})
    axes[1,0].set_title('Average Performance by Fold and Class', fontsize=32, fontweight='bold')
    axes[1,0].tick_params(labelsize=24)
    
    # 5. 성능 지표별 변동성 (Bar Plot)
    metric_cv = cv_data.groupby('Metric_Type')['CV(%)'].mean().reset_index()
    metric_cv = metric_cv.sort_values('CV(%)', ascending=True)
    
    bars = axes[1,1].bar(metric_cv['Metric_Type'], metric_cv['CV(%)'], 
                        color=['lightgreen', 'orange', 'pink', 'lightcoral'])
    axes[1,1].set_title('Average CV by Metric Type', fontsize=32, fontweight='bold')
    axes[1,1].set_xlabel('Metric Type', fontsize=28)
    axes[1,1].set_ylabel('Average CV (%)', fontsize=28)
    axes[1,1].tick_params(labelsize=24)
    
    # 값 표시
    for bar, value in zip(bars, metric_cv['CV(%)']):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.2f}%', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    # 6. 클래스별 성능 비교 (Bar Plot)
    tooth_means = df[tooth_metrics].mean()
    root_means = df[root_metrics].mean()
    overall_means = df[['Overall_f1', 'Overall_iou', 'Overall_prec', 'Overall_rec']].mean()
    
    metrics = ['F1-Score', 'IoU', 'Precision', 'Recall']
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = axes[1,2].bar(x - width, tooth_means.values, width, label='Tooth', alpha=0.8)
    bars2 = axes[1,2].bar(x, root_means.values, width, label='Root', alpha=0.8)
    bars3 = axes[1,2].bar(x + width, overall_means.values, width, label='Overall', alpha=0.8)
    
    axes[1,2].set_title('Average Performance Comparison', fontsize=32, fontweight='bold')
    axes[1,2].set_xlabel('Metrics', fontsize=28)
    axes[1,2].set_ylabel('Performance Value', fontsize=28)
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(metrics, rotation=45)
    axes[1,2].legend(fontsize=24)
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].tick_params(labelsize=24)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=20)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=20)
    
    for bar in bars3:
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('Unet_kfold_statistical_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.show()

def generate_summary_report(results_df):
    """요약 리포트 생성"""
    print("\n" + "="*60)
    print("K-FOLD 통계분석 요약 리포트")
    print("="*60)
    
    # 주요 발견사항
    print("\n📊 주요 발견사항:")
    
    # 1. 가장 변동이 큰 메트릭
    max_cv_idx = results_df['CV(%)'].idxmax()
    max_cv_row = results_df.loc[max_cv_idx]
    print(f"• 가장 변동이 큰 메트릭: {max_cv_row['Metric']}")
    print(f"  변동계수: {max_cv_row['CV(%)']:.2f}%")
    
    # 2. 가장 변동이 작은 메트릭
    min_cv_idx = results_df['CV(%)'].idxmin()
    min_cv_row = results_df.loc[min_cv_idx]
    print(f"• 가장 변동이 작은 메트릭: {min_cv_row['Metric']}")
    print(f"  변동계수: {min_cv_row['CV(%)']:.2f}%")
    
    # 3. 통계적 유의성
    significant_count = (results_df['Significance'] == 'Significant').sum()
    total_count = len(results_df)
    print(f"• 통계적으로 유의한 차이: {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
    
    # 4. Class별 분석
    print("\n📈 Class별 분석:")
    classes = ['Tooth', 'Root']  # Overall 제거하고 Tooth가 먼저 나오도록 순서 변경
    for class_name in classes:
        class_data = results_df[results_df['Metric'].str.startswith(class_name)]
        if len(class_data) > 0:
            avg_cv = class_data['CV(%)'].mean()
            significant_count = (class_data['Significance'] == 'Significant').sum()
            print(f"• {class_name}: 평균 CV {avg_cv:.2f}%, 유의한 차이 {significant_count}/{len(class_data)}")
    
    # 5. 메트릭 타입별 분석
    print("\n📊 메트릭 타입별 분석:")
    metric_types = ['acc', 'f1', 'iou', 'prec', 'rec']
    for metric_type in metric_types:
        metric_data = results_df[results_df['Metric'].str.contains(f'_{metric_type}')]
        if len(metric_data) > 0:
            avg_cv = metric_data['CV(%)'].mean()
            significant_count = (metric_data['Significance'] == 'Significant').sum()
            print(f"• {metric_type.upper()}: 평균 CV {avg_cv:.2f}%, 유의한 차이 {significant_count}/{len(metric_data)}")

def main():
    """메인 실행 함수"""
    # 데이터 로드
    df = load_and_explore_data('Unet_kfold_metrics.csv')
    
    # 통계적 검정 수행
    results_df = perform_statistical_tests(df)
    
    # 상세 분석 결과 출력
    results_df = print_detailed_analysis(results_df)
    
    # 시각화 생성
    create_visualizations(df, results_df)
    
    # 요약 리포트 생성
    generate_summary_report(results_df)
    
    # 결과를 CSV로 저장 (소수점 4자리까지)
    numeric_columns = ['Mean', 'Std', 'Min', 'Max', 'CV(%)', 'F_Statistic', 'P_Value', 'Effect_Size', 'Shapiro_P']
    results_df[numeric_columns] = results_df[numeric_columns].round(4)
    
    results_df.to_csv('kfold_statistical_results.csv', index=False)
    print(f"\n✅ 분석 결과가 'kfold_statistical_results.csv'에 저장되었습니다.")
    print(f"✅ 시각화가 'kfold_statistical_analysis.png'에 저장되었습니다.")

if __name__ == "__main__":
    main() 