import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Malgun Gothic', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24  # 기본 폰트 크기
plt.rcParams['axes.titlesize'] = 26  # 축 제목 폰트 크기
plt.rcParams['axes.labelsize'] = 24  # 축 라벨 폰트 크기
plt.rcParams['xtick.labelsize'] = 22  # x축 눈금 라벨 폰트 크기
plt.rcParams['ytick.labelsize'] = 22  # y축 눈금 라벨 폰트 크기
plt.rcParams['legend.fontsize'] = 22  # 범례 폰트 크기

def load_kfold_data():
    """K-fold 데이터 로드"""
    unet_df = pd.read_csv('Unet_kfold_metrics.csv')
    yolo_df = pd.read_csv('Yolo_kfold_training_results.csv')
    
    print("=== UNet K-fold 데이터 정보 ===")
    print(f"데이터 크기: {unet_df.shape}")
    print(f"Fold 수: {len(unet_df)}")
    print(f"메트릭 수: {len(unet_df.columns) - 1}")
    
    print("\n=== YOLO K-fold 데이터 정보 ===")
    print(f"데이터 크기: {yolo_df.shape}")
    print(f"Fold 수: {len(yolo_df)}")
    print(f"메트릭 수: {len(yolo_df.columns) - 1}")
    
    return unet_df, yolo_df

def calculate_variation_statistics(df, model_name):
    """각 메트릭의 variation 통계 계산"""
    results = []
    
    # fold 컬럼 제외한 모든 메트릭
    metric_cols = [col for col in df.columns if col != 'fold']
    
    for metric in metric_cols:
        fold_values = df[metric].values
        
        # 기본 통계
        mean_val = np.mean(fold_values)
        std_val = np.std(fold_values)
        min_val = np.min(fold_values)
        max_val = np.max(fold_values)
        range_val = max_val - min_val
        cv = std_val / mean_val * 100  # 변동계수
        
        # IQR (Interquartile Range)
        q25 = np.percentile(fold_values, 25)
        q75 = np.percentile(fold_values, 75)
        iqr = q75 - q25
        
        # 정규성 검정
        try:
            shapiro_stat, shapiro_p = stats.shapiro(fold_values)
            is_normal = shapiro_p > 0.05
        except:
            shapiro_stat, shapiro_p = np.nan, np.nan
            is_normal = False
        
        results.append({
            'Metric': metric,
            'Model': model_name,
            'Mean': mean_val,
            'Std': std_val,
            'Min': min_val,
            'Max': max_val,
            'Range': range_val,
            'CV(%)': cv,
            'Q25': q25,
            'Q75': q75,
            'IQR': iqr,
            'Is_Normal': is_normal,
            'Shapiro_P': shapiro_p
        })
    
    return pd.DataFrame(results)

def create_variation_visualizations(unet_df, yolo_df, unet_stats, yolo_stats):
    """Fold간 variation 시각화 생성"""
    # DPI 설정
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    
    # 3x2 서브플롯 생성
    fig, axes = plt.subplots(3, 2, figsize=(24, 30))
    fig.suptitle('K-Fold Cross-Validation Performance Variation Analysis (Excluding Overall)', fontsize=28, fontweight='bold')
    
    # 1. 변동계수 비교 (Boxplot)
    try:
        # UNet과 YOLO의 변동계수 데이터 결합
        cv_data = pd.concat([
            unet_stats[['Metric', 'CV(%)']].assign(Model='UNet'),
            yolo_stats[['Metric', 'CV(%)']].assign(Model='YOLO')
        ])
        
        # 메트릭 카테고리 분류
        cv_data['Category'] = cv_data['Metric'].str.split('_').str[0]
        cv_data['Metric_Type'] = cv_data['Metric'].str.split('_').str[1]
        
        # Background, Total, Overall 제외
        cv_data = cv_data[~cv_data['Category'].isin(['Background', 'Total', 'Overall'])]
        
        sns.boxplot(data=cv_data, x='Category', y='CV(%)', hue='Model', ax=axes[0,0])
        axes[0,0].set_title('Coefficient of Variation by Category and Model', fontsize=26, fontweight='bold')
        axes[0,0].set_ylabel('CV (%)', fontsize=24)
        axes[0,0].set_xlabel('Category', fontsize=24)
        axes[0,0].tick_params(axis='x', rotation=45, labelsize=22)
        axes[0,0].tick_params(axis='y', labelsize=22)
        axes[0,0].legend(fontsize=22)
        
    except Exception as e:
        axes[0,0].text(0.5, 0.5, 'CV comparison plot failed', ha='center', va='center', fontsize=24)
        axes[0,0].set_title('Coefficient of Variation Comparison', fontsize=26, fontweight='bold')
    
    # 2. Fold별 성능 변화 (Line plot)
    try:
        # UNet 데이터 준비 (Overall 제외)
        unet_metrics = ['Root_f1', 'Tooth_f1']
        for i, metric in enumerate(unet_metrics):
            if metric in unet_df.columns:
                axes[0,1].plot(unet_df['fold'], unet_df[metric], 
                              marker='o', linewidth=3, markersize=8, 
                              label=f'UNet {metric.split("_")[0]}', alpha=0.8)
        
        # YOLO 데이터 준비 (Total 제외)
        yolo_metrics = ['Root_F1', 'Tooth_F1']
        for i, metric in enumerate(yolo_metrics):
            if metric in yolo_df.columns:
                axes[0,1].plot(yolo_df['fold'], yolo_df[metric], 
                              marker='s', linewidth=3, markersize=8, 
                              label=f'YOLO {metric.split("_")[0]}', alpha=0.8)
        
        axes[0,1].set_title('F1 Score Variation Across Folds (Excluding Overall/Total)', fontsize=26, fontweight='bold')
        axes[0,1].set_xlabel('Fold', fontsize=24)
        axes[0,1].set_ylabel('F1 Score', fontsize=24)
        axes[0,1].tick_params(axis='x', labelsize=22)
        axes[0,1].tick_params(axis='y', labelsize=22)
        axes[0,1].legend(fontsize=20)
        axes[0,1].grid(True, alpha=0.3)
        
    except Exception as e:
        axes[0,1].text(0.5, 0.5, 'F1 variation plot failed', ha='center', va='center', fontsize=24)
        axes[0,1].set_title('F1 Score Variation Across Folds', fontsize=26, fontweight='bold')
    
    # 3. 표준편차 비교 (Bar plot)
    try:
        # UNet과 YOLO의 표준편차 비교
        std_data = pd.concat([
            unet_stats[['Metric', 'Std']].assign(Model='UNet'),
            yolo_stats[['Metric', 'Std']].assign(Model='YOLO')
        ])
        
        # 메트릭 카테고리 분류
        std_data['Category'] = std_data['Metric'].str.split('_').str[0]
        std_data['Metric_Type'] = std_data['Metric'].str.split('_').str[1]
        
        # Background, Total, Overall 제외
        std_data = std_data[~std_data['Category'].isin(['Background', 'Total', 'Overall'])]
        
        # 평균 표준편차 계산
        avg_std = std_data.groupby(['Model', 'Category'])['Std'].mean().reset_index()
        
        pivot_std = avg_std.pivot(index='Category', columns='Model', values='Std')
        pivot_std.plot(kind='bar', ax=axes[1,0], width=0.8)
        
        axes[1,0].set_title('Average Standard Deviation by Category (Excluding Overall)', fontsize=26, fontweight='bold')
        axes[1,0].set_ylabel('Standard Deviation', fontsize=24)
        axes[1,0].set_xlabel('Category', fontsize=24)
        axes[1,0].tick_params(axis='x', rotation=45, labelsize=22)
        axes[1,0].tick_params(axis='y', labelsize=22)
        axes[1,0].legend(fontsize=22)
        axes[1,0].grid(axis='y', alpha=0.3)
        
    except Exception as e:
        axes[1,0].text(0.5, 0.5, 'Standard deviation plot failed', ha='center', va='center', fontsize=24)
        axes[1,0].set_title('Average Standard Deviation by Category', fontsize=26, fontweight='bold')
    
    # 4. 범위(Range) 비교
    try:
        # UNet과 YOLO의 범위 비교
        range_data = pd.concat([
            unet_stats[['Metric', 'Range']].assign(Model='UNet'),
            yolo_stats[['Metric', 'Range']].assign(Model='YOLO')
        ])
        
        # 메트릭 카테고리 분류
        range_data['Category'] = range_data['Metric'].str.split('_').str[0]
        range_data['Metric_Type'] = range_data['Metric'].str.split('_').str[1]
        
        # Background, Total, Overall 제외
        range_data = range_data[~range_data['Category'].isin(['Background', 'Total', 'Overall'])]
        
        # 평균 범위 계산
        avg_range = range_data.groupby(['Model', 'Category'])['Range'].mean().reset_index()
        
        pivot_range = avg_range.pivot(index='Category', columns='Model', values='Range')
        pivot_range.plot(kind='bar', ax=axes[1,1], width=0.8)
        
        axes[1,1].set_title('Average Range by Category (Excluding Overall)', fontsize=26, fontweight='bold')
        axes[1,1].set_ylabel('Range (Max - Min)', fontsize=24)
        axes[1,1].set_xlabel('Category', fontsize=24)
        axes[1,1].tick_params(axis='x', rotation=45, labelsize=22)
        axes[1,1].tick_params(axis='y', labelsize=22)
        axes[1,1].legend(fontsize=22)
        axes[1,1].grid(axis='y', alpha=0.3)
        
    except Exception as e:
        axes[1,1].text(0.5, 0.5, 'Range comparison plot failed', ha='center', va='center', fontsize=24)
        axes[1,1].set_title('Average Range by Category', fontsize=26, fontweight='bold')
    
    # 5. IQR 비교
    try:
        # UNet과 YOLO의 IQR 비교
        iqr_data = pd.concat([
            unet_stats[['Metric', 'IQR']].assign(Model='UNet'),
            yolo_stats[['Metric', 'IQR']].assign(Model='YOLO')
        ])
        
        # 메트릭 카테고리 분류
        iqr_data['Category'] = iqr_data['Metric'].str.split('_').str[0]
        iqr_data['Metric_Type'] = iqr_data['Metric'].str.split('_').str[1]
        
        # Background, Total, Overall 제외
        iqr_data = iqr_data[~iqr_data['Category'].isin(['Background', 'Total', 'Overall'])]
        
        # 평균 IQR 계산
        avg_iqr = iqr_data.groupby(['Model', 'Category'])['IQR'].mean().reset_index()
        
        pivot_iqr = avg_iqr.pivot(index='Category', columns='Model', values='IQR')
        pivot_iqr.plot(kind='bar', ax=axes[2,0], width=0.8)
        
        axes[2,0].set_title('Average IQR by Category (Excluding Overall)', fontsize=26, fontweight='bold')
        axes[2,0].set_ylabel('IQR (Q75 - Q25)', fontsize=24)
        axes[2,0].set_xlabel('Category', fontsize=24)
        axes[2,0].tick_params(axis='x', rotation=45, labelsize=22)
        axes[2,0].tick_params(axis='y', labelsize=22)
        axes[2,0].legend(fontsize=22)
        axes[2,0].grid(axis='y', alpha=0.3)
        
    except Exception as e:
        axes[2,0].text(0.5, 0.5, 'IQR comparison plot failed', ha='center', va='center', fontsize=24)
        axes[2,0].set_title('Average IQR by Category', fontsize=26, fontweight='bold')
    
    # 6. 정규성 검정 결과
    try:
        # 정규성 검정 결과 비교
        normality_data = pd.concat([
            unet_stats[['Metric', 'Is_Normal']].assign(Model='UNet'),
            yolo_stats[['Metric', 'Is_Normal']].assign(Model='YOLO')
        ])
        
        # 메트릭 카테고리 분류
        normality_data['Category'] = normality_data['Metric'].str.split('_').str[0]
        normality_data['Metric_Type'] = normality_data['Metric'].str.split('_').str[1]
        
        # Background, Total, Overall 제외
        normality_data = normality_data[~normality_data['Category'].isin(['Background', 'Total', 'Overall'])]
        
        # 정규성 만족 비율 계산
        normality_ratio = normality_data.groupby(['Model', 'Category'])['Is_Normal'].mean().reset_index()
        
        pivot_normality = normality_ratio.pivot(index='Category', columns='Model', values='Is_Normal')
        pivot_normality.plot(kind='bar', ax=axes[2,1], width=0.8)
        
        axes[2,1].set_title('Normality Test Results by Category (Excluding Overall)', fontsize=26, fontweight='bold')
        axes[2,1].set_ylabel('Proportion of Normal Distributions', fontsize=24)
        axes[2,1].set_xlabel('Category', fontsize=24)
        axes[2,1].tick_params(axis='x', rotation=45, labelsize=22)
        axes[2,1].tick_params(axis='y', labelsize=22)
        axes[2,1].legend(fontsize=22)
        axes[2,1].grid(axis='y', alpha=0.3)
        axes[2,1].set_ylim(0, 1)
        
    except Exception as e:
        axes[2,1].text(0.5, 0.5, 'Normality test plot failed', ha='center', va='center', fontsize=24)
        axes[2,1].set_title('Normality Test Results by Category', fontsize=26, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fold_variation_analysis_no_overall.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.show()

def create_detailed_variation_plots(unet_df, yolo_df):
    """상세한 variation 시각화"""
    # DPI 설정
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('Detailed K-Fold Performance Variation Analysis (Excluding Overall/Total)', fontsize=28, fontweight='bold')
    
    # 1. UNet fold별 성능 변화
    try:
        # UNet 주요 메트릭 선택 (Overall 제외)
        unet_metrics = ['Root_f1', 'Tooth_f1', 'Root_iou', 'Tooth_iou']
        available_metrics = [m for m in unet_metrics if m in unet_df.columns]
        
        for metric in available_metrics:
            category = metric.split('_')[0]
            metric_type = metric.split('_')[1]
            axes[0,0].plot(unet_df['fold'], unet_df[metric], 
                          marker='o', linewidth=2, markersize=6, 
                          label=f'{category} {metric_type.upper()}', alpha=0.8)
        
        axes[0,0].set_title('UNet Performance Variation Across Folds (Excluding Overall)', fontsize=26, fontweight='bold')
        axes[0,0].set_xlabel('Fold', fontsize=24)
        axes[0,0].set_ylabel('Performance Score', fontsize=24)
        axes[0,0].tick_params(axis='x', labelsize=22)
        axes[0,0].tick_params(axis='y', labelsize=22)
        axes[0,0].legend(fontsize=18)
        axes[0,0].grid(True, alpha=0.3)
        
        # x축 눈금을 1, 2, 3, 4, 5로 설정
        axes[0,0].set_xticks([1, 2, 3, 4, 5])
        axes[0,0].set_xticklabels(['1', '2', '3', '4', '5'])
        
    except Exception as e:
        axes[0,0].text(0.5, 0.5, 'UNet variation plot failed', ha='center', va='center', fontsize=24)
        axes[0,0].set_title('UNet Performance Variation', fontsize=26, fontweight='bold')
    
    # 2. YOLO fold별 성능 변화
    try:
        # YOLO 주요 메트릭 선택 (Total 제외)
        yolo_metrics = ['Root_F1', 'Tooth_F1', 'Root_mAP50', 'Tooth_mAP50']
        available_metrics = [m for m in yolo_metrics if m in yolo_df.columns]
        
        for metric in available_metrics:
            category = metric.split('_')[0]
            metric_type = metric.split('_')[1]
            axes[0,1].plot(yolo_df['fold'], yolo_df[metric], 
                          marker='s', linewidth=2, markersize=6, 
                          label=f'{category} {metric_type.upper()}', alpha=0.8)
        
        axes[0,1].set_title('YOLO Performance Variation Across Folds (Excluding Total)', fontsize=26, fontweight='bold')
        axes[0,1].set_xlabel('Fold', fontsize=24)
        axes[0,1].set_ylabel('Performance Score', fontsize=24)
        axes[0,1].tick_params(axis='x', labelsize=22)
        axes[0,1].tick_params(axis='y', labelsize=22)
        axes[0,1].legend(fontsize=18)
        axes[0,1].grid(True, alpha=0.3)
        
        # x축 눈금을 1, 2, 3, 4, 5로 설정
        axes[0,1].set_xticks([1, 2, 3, 4, 5])
        axes[0,1].set_xticklabels(['1', '2', '3', '4', '5'])
        
    except Exception as e:
        axes[0,1].text(0.5, 0.5, 'YOLO variation plot failed', ha='center', va='center', fontsize=24)
        axes[0,1].set_title('YOLO Performance Variation', fontsize=26, fontweight='bold')
    
    # 3. 모델별 변동성 비교 (Violin plot)
    try:
        # UNet과 YOLO의 F1 점수 변동성 비교
        f1_data = []
        
        # UNet F1 데이터 (Overall 제외)
        for category in ['Root', 'Tooth']:
            metric = f'{category}_f1'
            if metric in unet_df.columns:
                for fold, value in zip(unet_df['fold'], unet_df[metric]):
                    f1_data.append({
                        'Fold': fold,
                        'Value': value,
                        'Category': category,
                        'Model': 'UNet'
                    })
        
        # YOLO F1 데이터 (Total 제외)
        for category in ['Root', 'Tooth']:
            metric = f'{category}_F1'
            if metric in yolo_df.columns:
                for fold, value in zip(yolo_df['fold'], yolo_df[metric]):
                    f1_data.append({
                        'Fold': fold,
                        'Value': value,
                        'Category': category,
                        'Model': 'YOLO'
                    })
        
        f1_df = pd.DataFrame(f1_data)
        
        sns.violinplot(data=f1_df, x='Category', y='Value', hue='Model', ax=axes[1,0])
        axes[1,0].set_title('F1 Score Distribution by Category and Model (Excluding Overall/Total)', fontsize=26, fontweight='bold')
        axes[1,0].set_xlabel('Category', fontsize=24)
        axes[1,0].set_ylabel('F1 Score', fontsize=24)
        axes[1,0].tick_params(axis='x', labelsize=22)
        axes[1,0].tick_params(axis='y', labelsize=22)
        axes[1,0].legend(fontsize=22)
        
    except Exception as e:
        axes[1,0].text(0.5, 0.5, 'Violin plot failed', ha='center', va='center', fontsize=24)
        axes[1,0].set_title('F1 Score Distribution', fontsize=26, fontweight='bold')
    
    # 4. 변동계수 히트맵
    try:
        # UNet과 YOLO의 변동계수 계산
        cv_data = []
        
        # UNet 변동계수 (Overall 제외)
        for col in unet_df.columns:
            if col != 'fold' and not col.startswith('Overall_') and not col.startswith('Background_'):
                cv = unet_df[col].std() / unet_df[col].mean() * 100
                category = col.split('_')[0]
                metric_type = col.split('_')[1]
                cv_data.append({
                    'Category': category,
                    'Metric': metric_type,
                    'CV': cv,
                    'Model': 'UNet'
                })
        
        # YOLO 변동계수 (Total 제외)
        for col in yolo_df.columns:
            if col != 'fold' and not col.startswith('Total_'):
                cv = yolo_df[col].std() / yolo_df[col].mean() * 100
                category = col.split('_')[0]
                metric_type = col.split('_')[1]
                cv_data.append({
                    'Category': category,
                    'Metric': metric_type,
                    'CV': cv,
                    'Model': 'YOLO'
                })
        
        cv_df = pd.DataFrame(cv_data)
        
        # 히트맵 생성
        pivot_cv = cv_df.pivot_table(values='CV', index='Category', columns='Model', aggfunc='mean')
        
        sns.heatmap(pivot_cv, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=axes[1,1],
                   cbar_kws={'label': 'CV (%)'}, annot_kws={'size': 16})
        axes[1,1].set_title('Average Coefficient of Variation by Category (Excluding Overall/Total)', fontsize=26, fontweight='bold')
        axes[1,1].tick_params(axis='x', labelsize=22)
        axes[1,1].tick_params(axis='y', labelsize=22)
        
    except Exception as e:
        axes[1,1].text(0.5, 0.5, 'Heatmap failed', ha='center', va='center', fontsize=24)
        axes[1,1].set_title('Coefficient of Variation Heatmap', fontsize=26, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_fold_variation_analysis_no_overall.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.show()

def generate_variation_summary(unet_stats, yolo_stats):
    """Variation 분석 요약 리포트"""
    print("\n" + "="*60)
    print("K-FOLD VARIATION 분석 요약 리포트")
    print("="*60)
    
    # UNet 변동성 요약
    print("\n📊 UNet 변동성 요약:")
    unet_cv_mean = unet_stats['CV(%)'].mean()
    unet_cv_std = unet_stats['CV(%)'].std()
    unet_normal_ratio = unet_stats['Is_Normal'].mean() * 100
    
    print(f"• 평균 변동계수: {unet_cv_mean:.2f}% ± {unet_cv_std:.2f}%")
    print(f"• 정규성 만족 비율: {unet_normal_ratio:.1f}%")
    print(f"• 가장 변동이 큰 메트릭: {unet_stats.loc[unet_stats['CV(%)'].idxmax(), 'Metric']}")
    print(f"• 가장 변동이 작은 메트릭: {unet_stats.loc[unet_stats['CV(%)'].idxmin(), 'Metric']}")
    
    # YOLO 변동성 요약
    print("\n📊 YOLO 변동성 요약:")
    yolo_cv_mean = yolo_stats['CV(%)'].mean()
    yolo_cv_std = yolo_stats['CV(%)'].std()
    yolo_normal_ratio = yolo_stats['Is_Normal'].mean() * 100
    
    print(f"• 평균 변동계수: {yolo_cv_mean:.2f}% ± {yolo_cv_std:.2f}%")
    print(f"• 정규성 만족 비율: {yolo_normal_ratio:.1f}%")
    print(f"• 가장 변동이 큰 메트릭: {yolo_stats.loc[yolo_stats['CV(%)'].idxmax(), 'Metric']}")
    print(f"• 가장 변동이 작은 메트릭: {yolo_stats.loc[yolo_stats['CV(%)'].idxmin(), 'Metric']}")
    
    # 모델 간 비교
    print("\n📈 모델 간 변동성 비교:")
    if unet_cv_mean < yolo_cv_mean:
        print(f"• UNet이 YOLO보다 {yolo_cv_mean - unet_cv_mean:.2f}% 더 안정적")
    else:
        print(f"• YOLO가 UNet보다 {unet_cv_mean - yolo_cv_mean:.2f}% 더 안정적")
    
    if unet_normal_ratio > yolo_normal_ratio:
        print(f"• UNet이 YOLO보다 정규성 만족 비율이 {unet_normal_ratio - yolo_normal_ratio:.1f}% 높음")
    else:
        print(f"• YOLO가 UNet보다 정규성 만족 비율이 {yolo_normal_ratio - unet_normal_ratio:.1f}% 높음")

def main():
    """메인 실행 함수"""
    # 데이터 로드
    unet_df, yolo_df = load_kfold_data()
    
    # Variation 통계 계산
    unet_stats = calculate_variation_statistics(unet_df, 'UNet')
    yolo_stats = calculate_variation_statistics(yolo_df, 'YOLO')
    
    # 종합 variation 시각화 생성
    create_variation_visualizations(unet_df, yolo_df, unet_stats, yolo_stats)
    
    # 상세 variation 시각화 생성
    create_detailed_variation_plots(unet_df, yolo_df)
    
    # Variation 요약 리포트 생성
    generate_variation_summary(unet_stats, yolo_stats)
    
    # 결과를 CSV로 저장
    combined_stats = pd.concat([unet_stats, yolo_stats], ignore_index=True)
    combined_stats.to_csv('fold_variation_statistics.csv', index=False)
    
    print(f"\n✅ 종합 variation 분석이 'fold_variation_analysis.png'에 저장되었습니다.")
    print(f"✅ 상세 variation 분석이 'detailed_fold_variation_analysis.png'에 저장되었습니다.")
    print(f"✅ Variation 통계가 'fold_variation_statistics.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()
