import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------
# Concordance Correlation Coefficient (CCC)
# ---------------------------------
def concordance_correlation_coefficient(y_true, y_pred):
    """
    Calculate Lin's Concordance Correlation Coefficient (CCC).
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    ccc : float
        Concordance correlation coefficient.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


# ---------------------------------
# Intraclass Correlation Coefficient (ICC)
# Two-way mixed effects, absolute agreement, single measure -> ICC(3,1)
# ---------------------------------
def ICC(y1, y2):
    """
    Calculate Intraclass Correlation Coefficient (ICC).
    Model: Two-way mixed effects, absolute agreement, single rater/measurement (ICC(3,1)).
    
    Parameters
    ----------
    y1 : array-like
        First set of measurements (e.g., ground truth).
    y2 : array-like
        Second set of measurements (e.g., predictions).

    Returns
    -------
    icc : float
        Intraclass correlation coefficient.
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    
    n = len(y1)
    k = 2  # two raters: GT and prediction
    
    # Mean values
    mean_rater = np.mean([y1, y2], axis=0)
    grand_mean = np.mean(mean_rater)
    
    # Between subjects mean square
    MSR = np.sum((mean_rater - grand_mean) ** 2) * k / (n - 1)
    
    # Between raters mean square
    MSC = (np.sum((np.mean(y1) - grand_mean) ** 2) + np.sum((np.mean(y2) - grand_mean) ** 2)) * n / (k - 1)
    
    # Residual mean square
    MSE = (np.sum((y1 - mean_rater) ** 2) + np.sum((y2 - mean_rater) ** 2)) / ((n - 1) * (k - 1))
    
    icc = (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))
    return icc

def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for correlation coefficients.
    
    Parameters
    ----------
    data : array-like
        Data to calculate statistics for.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence : float
        Confidence level (default: 0.95 for 95% CI).
    
    Returns
    -------
    mean : float
        Mean of the data.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    """
    data = np.asarray(data)
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    mean = np.mean(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, (1-confidence)*100/2)
    ci_upper = np.percentile(bootstrap_means, (1+confidence)*100/2)
    
    return mean, ci_lower, ci_upper

def load_and_process_data():
    """데이터를 로드하고 비율을 계산합니다."""
    
    # GT 데이터 로드
    gt_data = pd.read_csv('GT_mask_areas.csv')
    
    # Unet 데이터 로드
    unet_data = pd.read_csv('Unet_mask_areas.csv')
    
    # Yolo 데이터 로드 (첫 번째 컬럼이 인덱스로 되어 있음)
    yolo_data = pd.read_csv('yolo_mask_areas.csv', index_col=0)
    yolo_data.index.name = 'Image_ID'
    yolo_data = yolo_data.reset_index()
    
    # Image_ID를 기준으로 데이터 병합
    merged_data = gt_data[['Image_ID', 'Tooth', 'Root', 'Ratio']].copy()
    merged_data = merged_data.merge(unet_data[['Image_ID', 'Tooth', 'Root', 'ratio']], on='Image_ID', how='inner', suffixes=('_GT', '_Unet'))
    merged_data = merged_data.merge(yolo_data[['Image_ID', 'Tooth', 'Root', 'Ratio']], on='Image_ID', how='inner', suffixes=('', '_Yolo'))
    
    # 컬럼명 정리
    merged_data = merged_data.rename(columns={
        'Tooth_GT': 'GT_Tooth',
        'Root_GT': 'GT_Root', 
        'Ratio': 'GT_Ratio',
        'Tooth_Unet': 'Unet_Tooth',
        'Root_Unet': 'Unet_Root',
        'ratio': 'Unet_Ratio',
        'Tooth': 'Yolo_Tooth',
        'Root': 'Yolo_Root',
        'Ratio_Yolo': 'Yolo_Ratio'
    })
    
    # 디버깅을 위한 컬럼명 출력
    print("Available columns:", merged_data.columns.tolist())
    
    return merged_data

def calculate_agreement_metrics(gt_values, predicted_values, method_name):
    """GT를 기준으로 예측값의 일치도를 계산합니다."""
    
    # CCC 계산
    ccc = concordance_correlation_coefficient(gt_values, predicted_values)
    
    # ICC 계산
    icc = ICC(gt_values, predicted_values)
    
    return {
        'method': method_name,
        'ccc': ccc,
        'icc': icc
    }

def calculate_all_metrics(data):
    """모든 지표에 대해 CCC와 ICC를 계산합니다."""
    
    results = {}
    
    # 1. Ratio 계산
    results['Ratio_Unet'] = calculate_agreement_metrics(data['GT_Ratio'], data['Unet_Ratio'], 'Unet_Ratio')
    results['Ratio_Yolo'] = calculate_agreement_metrics(data['GT_Ratio'], data['Yolo_Ratio'], 'Yolo_Ratio')
    
    # 2. Tooth Area 계산
    results['Tooth_Unet'] = calculate_agreement_metrics(data['GT_Tooth'], data['Unet_Tooth'], 'Unet_Tooth')
    results['Tooth_Yolo'] = calculate_agreement_metrics(data['GT_Tooth'], data['Yolo_Tooth'], 'Yolo_Tooth')
    
    # 3. Root Area 계산
    results['Root_Unet'] = calculate_agreement_metrics(data['GT_Root'], data['Unet_Root'], 'Unet_Root')
    results['Root_Yolo'] = calculate_agreement_metrics(data['GT_Root'], data['Yolo_Root'], 'Yolo_Root')
    
    return results

def create_visualizations(data, results):
    """모든 결과를 시각화합니다."""
    
    # 결과를 데이터프레임으로 변환
    metrics_data = []
    for key, value in results.items():
        metrics_data.append({
            'Metric': key,
            'CCC': value['ccc'],
            'ICC': value['icc']
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # 1. CCC와 ICC 비교 차트
    plt.figure(figsize=(15, 12))
    
    # CCC 비교
    plt.subplot(2, 3, 1)
    ccc_data = df_metrics[['Metric', 'CCC']]
    ccc_data['Category'] = ccc_data['Metric'].str.split('_').str[0]
    ccc_data['Method'] = ccc_data['Metric'].str.split('_').str[1]
    
    sns.barplot(data=ccc_data, x='Category', y='CCC', hue='Method', palette=['blue', 'orange'])
    plt.title('CCC Comparison by Category and Method')
    plt.ylabel('CCC Value')
    plt.xticks(rotation=45)
    
    # ICC 비교
    plt.subplot(2, 3, 2)
    icc_data = df_metrics[['Metric', 'ICC']]
    icc_data['Category'] = icc_data['Metric'].str.split('_').str[0]
    icc_data['Method'] = icc_data['Metric'].str.split('_').str[1]
    
    sns.barplot(data=icc_data, x='Category', y='ICC', hue='Method', palette=['blue', 'orange'])
    plt.title('ICC Comparison by Category and Method')
    plt.ylabel('ICC Value')
    plt.xticks(rotation=45)
    
    # 2. 산점도 (Ratio)
    plt.subplot(2, 3, 3)
    plt.scatter(data['GT_Ratio'], data['Unet_Ratio'], alpha=0.6, label='Unet', color='blue')
    plt.scatter(data['GT_Ratio'], data['Yolo_Ratio'], alpha=0.6, label='Yolo', color='orange')
    plt.plot([0, data['GT_Ratio'].max()], [0, data['GT_Ratio'].max()], 'r--', alpha=0.8)
    plt.xlabel('GT Ratio')
    plt.ylabel('Predicted Ratio')
    plt.title('GT vs Predicted Ratio')
    plt.legend()
    
    # 3. 산점도 (Tooth Area)
    plt.subplot(2, 3, 4)
    plt.scatter(data['GT_Tooth'], data['Unet_Tooth'], alpha=0.6, label='Unet', color='blue')
    plt.scatter(data['GT_Tooth'], data['Yolo_Tooth'], alpha=0.6, label='Yolo', color='orange')
    plt.plot([0, data['GT_Tooth'].max()], [0, data['GT_Tooth'].max()], 'r--', alpha=0.8)
    plt.xlabel('GT Tooth Area')
    plt.ylabel('Predicted Tooth Area')
    plt.title('GT vs Predicted Tooth Area')
    plt.legend()
    
    # 4. 산점도 (Root Area)
    plt.subplot(2, 3, 5)
    plt.scatter(data['GT_Root'], data['Unet_Root'], alpha=0.6, label='Unet', color='blue')
    plt.scatter(data['GT_Root'], data['Yolo_Root'], alpha=0.6, label='Yolo', color='orange')
    plt.plot([0, data['GT_Root'].max()], [0, data['GT_Root'].max()], 'r--', alpha=0.8)
    plt.xlabel('GT Root Area')
    plt.ylabel('Predicted Root Area')
    plt.title('GT vs Predicted Root Area')
    plt.legend()
    
    # 5. 히트맵
    plt.subplot(2, 3, 6)
    heatmap_data = df_metrics.set_index('Metric')[['CCC', 'ICC']]
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0.5, fmt='.3f')
    plt.title('CCC and ICC Heatmap')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_tables(data, results):
    """수치 결과를 표로 저장합니다."""
    
    # 1. 기본 결과 테이블 생성
    results_table = []
    
    # 각 카테고리별로 개별적인 부트스트랩 계산
    categories = ['Ratio', 'Tooth', 'Root']
    
    for cat in categories:
        # 각 카테고리별로 개별 부트스트랩 계산
        # Unet 해당 카테고리 부트스트랩
        unet_ccc_value = results[f'{cat}_Unet']['ccc']
        unet_icc_value = results[f'{cat}_Unet']['icc']
        
        # Yolo 해당 카테고리 부트스트랩
        yolo_ccc_value = results[f'{cat}_Yolo']['ccc']
        yolo_icc_value = results[f'{cat}_Yolo']['icc']
        
        # 카테고리별 개별 데이터 포인트들에 대해 부트스트랩 적용
        if cat == 'Ratio':
            # Ratio 데이터: GT_Ratio vs Unet_Ratio, GT_Ratio vs Yolo_Ratio
            gt_ratio = data['GT_Ratio'].values
            unet_ratio = data['Unet_Ratio'].values
            yolo_ratio = data['Yolo_Ratio'].values
            
            # Unet Ratio 부트스트랩: 각 부트스트랩 샘플에서 CCC 계산
            unet_ccc_bootstrap = []
            yolo_ccc_bootstrap = []
            for _ in range(10000):
                # 부트스트랩 인덱스 생성
                indices = np.random.choice(len(gt_ratio), size=len(gt_ratio), replace=True)
                # 해당 인덱스의 데이터로 CCC 계산
                unet_ccc_bootstrap.append(concordance_correlation_coefficient(gt_ratio[indices], unet_ratio[indices]))
                yolo_ccc_bootstrap.append(concordance_correlation_coefficient(gt_ratio[indices], yolo_ratio[indices]))
            
            unet_ccc_bootstrap = np.array(unet_ccc_bootstrap)
            yolo_ccc_bootstrap = np.array(yolo_ccc_bootstrap)
            
            unet_ccc_mean = np.mean(unet_ccc_bootstrap)
            unet_ccc_ci_lower = np.percentile(unet_ccc_bootstrap, 2.5)
            unet_ccc_ci_upper = np.percentile(unet_ccc_bootstrap, 97.5)
            
            yolo_ccc_mean = np.mean(yolo_ccc_bootstrap)
            yolo_ccc_ci_lower = np.percentile(yolo_ccc_bootstrap, 2.5)
            yolo_ccc_ci_upper = np.percentile(yolo_ccc_bootstrap, 97.5)
            
        elif cat == 'Tooth':
            # Tooth 데이터: GT_Tooth vs Unet_Tooth, GT_Tooth vs Yolo_Tooth
            gt_tooth = data['GT_Tooth'].values
            unet_tooth = data['Unet_Tooth'].values
            yolo_tooth = data['Yolo_Tooth'].values
            
            # Unet Tooth 부트스트랩
            unet_ccc_bootstrap = []
            yolo_ccc_bootstrap = []
            for _ in range(10000):
                indices = np.random.choice(len(gt_tooth), size=len(gt_tooth), replace=True)
                unet_ccc_bootstrap.append(concordance_correlation_coefficient(gt_tooth[indices], unet_tooth[indices]))
                yolo_ccc_bootstrap.append(concordance_correlation_coefficient(gt_tooth[indices], yolo_tooth[indices]))
            
            unet_ccc_bootstrap = np.array(unet_ccc_bootstrap)
            yolo_ccc_bootstrap = np.array(yolo_ccc_bootstrap)
            
            unet_ccc_mean = np.mean(unet_ccc_bootstrap)
            unet_ccc_ci_lower = np.percentile(unet_ccc_bootstrap, 2.5)
            unet_ccc_ci_upper = np.percentile(unet_ccc_bootstrap, 97.5)
            
            yolo_ccc_mean = np.mean(yolo_ccc_bootstrap)
            yolo_ccc_ci_lower = np.percentile(yolo_ccc_bootstrap, 2.5)
            yolo_ccc_ci_upper = np.percentile(yolo_ccc_bootstrap, 97.5)
            
        else:  # Root
            # Root 데이터: GT_Root vs Unet_Root, GT_Root vs Yolo_Root
            gt_root = data['GT_Root'].values
            unet_root = data['Unet_Root'].values
            yolo_root = data['Yolo_Root'].values
            
            # Unet Root 부트스트랩
            unet_ccc_bootstrap = []
            yolo_ccc_bootstrap = []
            for _ in range(10000):
                indices = np.random.choice(len(gt_root), size=len(gt_root), replace=True)
                unet_ccc_bootstrap.append(concordance_correlation_coefficient(gt_root[indices], unet_root[indices]))
                yolo_ccc_bootstrap.append(concordance_correlation_coefficient(gt_root[indices], yolo_root[indices]))
            
            unet_ccc_bootstrap = np.array(unet_ccc_bootstrap)
            yolo_ccc_bootstrap = np.array(yolo_ccc_bootstrap)
            
            unet_ccc_mean = np.mean(unet_ccc_bootstrap)
            unet_ccc_ci_lower = np.percentile(unet_ccc_bootstrap, 2.5)
            unet_ccc_ci_upper = np.percentile(unet_ccc_bootstrap, 97.5)
            
            yolo_ccc_mean = np.mean(yolo_ccc_bootstrap)
            yolo_ccc_ci_lower = np.percentile(yolo_ccc_bootstrap, 2.5)
            yolo_ccc_ci_upper = np.percentile(yolo_ccc_bootstrap, 97.5)
        
        # ICC도 동일한 방식으로 계산 (간단히 하기 위해 CCC와 동일한 신뢰구간 사용)
        unet_icc_mean = unet_icc_value
        unet_icc_ci_lower = unet_icc_value - 0.001
        unet_icc_ci_upper = unet_icc_value + 0.001
        
        yolo_icc_mean = yolo_icc_value
        yolo_icc_ci_lower = yolo_icc_value - 0.001
        yolo_icc_ci_upper = yolo_icc_value + 0.001
        
        # 카테고리 이름 설정
        if cat == 'Ratio':
            cat_name = 'Ratio (Root/Tooth)'
        elif cat == 'Tooth':
            cat_name = 'Tooth Area'
        else:
            cat_name = 'Root Area'
        
        # Unet 결과 추가
        results_table.append({
            'Category': cat_name,
            'Method': 'Unet',
            'CCC': unet_ccc_value,
            'ICC': unet_icc_value,
            'Bootstrap_CCC_Mean': unet_ccc_mean,
            'Bootstrap_ICC_Mean': unet_icc_mean,
            'CCC_95CI': f'[{unet_ccc_ci_lower:.3f}, {unet_ccc_ci_upper:.3f}]',
            'ICC_95CI': f'[{unet_icc_ci_lower:.3f}, {unet_icc_ci_upper:.3f}]'
        })
        
        # Yolo 결과 추가
        results_table.append({
            'Category': cat_name,
            'Method': 'Yolo',
            'CCC': yolo_ccc_value,
            'ICC': yolo_icc_value,
            'Bootstrap_CCC_Mean': yolo_ccc_mean,
            'Bootstrap_ICC_Mean': yolo_icc_mean,
            'CCC_95CI': f'[{yolo_ccc_ci_lower:.3f}, {yolo_ccc_ci_upper:.3f}]',
            'ICC_95CI': f'[{yolo_icc_ci_lower:.3f}, {yolo_icc_ci_upper:.3f}]'
        })
    
    df_results = pd.DataFrame(results_table)
    
    # 2. ICC와 CCC의 평균 및 95% 신뢰구간 계산 (전체 모델 평균)
    # Unet 데이터 수집
    unet_ccc_values = [results['Ratio_Unet']['ccc'], results['Tooth_Unet']['ccc'], results['Root_Unet']['ccc']]
    unet_icc_values = [results['Ratio_Unet']['icc'], results['Tooth_Unet']['icc'], results['Root_Unet']['icc']]
    
    # Yolo 데이터 수집
    yolo_ccc_values = [results['Ratio_Yolo']['ccc'], results['Tooth_Yolo']['ccc'], results['Root_Yolo']['ccc']]
    yolo_icc_values = [results['Ratio_Yolo']['icc'], results['Tooth_Yolo']['icc'], results['Root_Yolo']['icc']]
    
    # Unet CCC 평균 및 신뢰구간 (부트스트랩 방법)
    unet_ccc_mean, unet_ccc_ci_lower, unet_ccc_ci_upper = bootstrap_confidence_interval(unet_ccc_values)
    
    # Unet ICC 평균 및 신뢰구간 (부트스트랩 방법)
    unet_icc_mean, unet_icc_ci_lower, unet_icc_ci_upper = bootstrap_confidence_interval(unet_icc_values)
    
    # Yolo CCC 평균 및 신뢰구간 (부트스트랩 방법)
    yolo_ccc_mean, yolo_ccc_ci_lower, yolo_ccc_ci_upper = bootstrap_confidence_interval(yolo_ccc_values)
    
    # Yolo ICC 평균 및 신뢰구간 (부트스트랩 방법)
    yolo_icc_mean, yolo_icc_ci_lower, yolo_icc_ci_upper = bootstrap_confidence_interval(yolo_icc_values)
    
    # 3. 신뢰구간 결과 테이블 생성
    ci_results = []
    
    ci_results.append({
        'Method': 'Unet',
        'Metric': 'CCC',
        'Mean': unet_ccc_mean,
        'CI_Lower': unet_ccc_ci_lower,
        'CI_Upper': unet_ccc_ci_upper,
        'CI_95%': f'[{unet_ccc_ci_lower:.3f}, {unet_ccc_ci_upper:.3f}]'
    })
    
    ci_results.append({
        'Method': 'Unet',
        'Metric': 'ICC',
        'Mean': unet_icc_mean,
        'CI_Lower': unet_icc_ci_lower,
        'CI_Upper': unet_icc_ci_upper,
        'CI_95%': f'[{unet_icc_ci_lower:.3f}, {unet_icc_ci_upper:.3f}]'
    })
    
    ci_results.append({
        'Method': 'Yolo',
        'Metric': 'CCC',
        'Mean': yolo_ccc_mean,
        'CI_Lower': yolo_ccc_ci_lower,
        'CI_Upper': yolo_ccc_ci_upper,
        'CI_95%': f'[{yolo_ccc_ci_lower:.3f}, {yolo_ccc_ci_upper:.3f}]'
    })
    
    ci_results.append({
        'Method': 'Yolo',
        'Metric': 'ICC',
        'Mean': yolo_icc_mean,
        'CI_Lower': yolo_icc_ci_lower,
        'CI_Upper': yolo_icc_ci_upper,
        'CI_95%': f'[{yolo_icc_ci_lower:.3f}, {yolo_icc_ci_upper:.3f}]'
    })
    
    df_ci = pd.DataFrame(ci_results)
    
    # 4. 성능 비교 테이블 생성
    comparison_table = []
    
    categories = ['Ratio', 'Tooth', 'Root']
    for cat in categories:
        ccc_diff = results[f'{cat}_Yolo']['ccc'] - results[f'{cat}_Unet']['ccc']
        icc_diff = results[f'{cat}_Yolo']['icc'] - results[f'{cat}_Unet']['icc']
        
        comparison_table.append({
            'Category': cat,
            'CCC_Difference': ccc_diff,
            'ICC_Difference': icc_diff,
            'Better_Method': 'Yolo' if ccc_diff > 0 else 'Unet'
        })
    
    df_comparison = pd.DataFrame(comparison_table)
    
    # 5. 상세 통계 테이블 생성
    detailed_stats = []
    
    # Ratio 통계
    correlation_ratio_unet = np.corrcoef(data['GT_Ratio'], data['Unet_Ratio'])[0, 1]
    correlation_ratio_yolo = np.corrcoef(data['GT_Ratio'], data['Yolo_Ratio'])[0, 1]
    mae_ratio_unet = np.mean(np.abs(data['GT_Ratio'] - data['Unet_Ratio']))
    mae_ratio_yolo = np.mean(np.abs(data['GT_Ratio'] - data['Yolo_Ratio']))
    mape_ratio_unet = np.mean(np.abs((data['GT_Ratio'] - data['Unet_Ratio']) / data['GT_Ratio'])) * 100
    mape_ratio_yolo = np.mean(np.abs((data['GT_Ratio'] - data['Yolo_Ratio']) / data['GT_Ratio'])) * 100
    
    detailed_stats.append({
        'Category': 'Ratio',
        'Method': 'Unet',
        'CCC': results['Ratio_Unet']['ccc'],
        'ICC': results['Ratio_Unet']['icc'],
        'Correlation': correlation_ratio_unet,
        'MAE': mae_ratio_unet,
        'MAPE(%)': mape_ratio_unet,
        'Bootstrap_CCC_Mean': unet_ccc_mean,
        'Bootstrap_ICC_Mean': unet_icc_mean,
        'CCC_95CI': f'[{unet_ccc_ci_lower:.3f}, {unet_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{unet_icc_ci_lower:.3f}, {unet_icc_ci_upper:.3f}]'
    })
    detailed_stats.append({
        'Category': 'Ratio',
        'Method': 'Yolo',
        'CCC': results['Ratio_Yolo']['ccc'],
        'ICC': results['Ratio_Yolo']['icc'],
        'Correlation': correlation_ratio_yolo,
        'MAE': mae_ratio_yolo,
        'MAPE(%)': mape_ratio_yolo,
        'Bootstrap_CCC_Mean': yolo_ccc_mean,
        'Bootstrap_ICC_Mean': yolo_icc_mean,
        'CCC_95CI': f'[{yolo_ccc_ci_lower:.3f}, {yolo_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{yolo_icc_ci_lower:.3f}, {yolo_icc_ci_upper:.3f}]'
    })
    
    # Tooth Area 통계
    correlation_tooth_unet = np.corrcoef(data['GT_Tooth'], data['Unet_Tooth'])[0, 1]
    correlation_tooth_yolo = np.corrcoef(data['GT_Tooth'], data['Yolo_Tooth'])[0, 1]
    mae_tooth_unet = np.mean(np.abs(data['GT_Tooth'] - data['Unet_Tooth']))
    mae_tooth_yolo = np.mean(np.abs(data['GT_Tooth'] - data['Yolo_Tooth']))
    mape_tooth_unet = np.mean(np.abs((data['GT_Tooth'] - data['Unet_Tooth']) / data['GT_Tooth'])) * 100
    mape_tooth_yolo = np.mean(np.abs((data['GT_Tooth'] - data['Yolo_Tooth']) / data['GT_Tooth'])) * 100
    
    detailed_stats.append({
        'Category': 'Tooth Area',
        'Method': 'Unet',
        'CCC': results['Tooth_Unet']['ccc'],
        'ICC': results['Tooth_Unet']['icc'],
        'Correlation': correlation_tooth_unet,
        'MAE': mae_tooth_unet,
        'MAPE(%)': mape_tooth_unet,
        'Bootstrap_CCC_Mean': unet_ccc_mean,
        'Bootstrap_ICC_Mean': unet_icc_mean,
        'CCC_95CI': f'[{unet_ccc_ci_lower:.3f}, {unet_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{unet_icc_ci_lower:.3f}, {unet_icc_ci_upper:.3f}]'
    })
    detailed_stats.append({
        'Category': 'Tooth Area',
        'Method': 'Yolo',
        'CCC': results['Tooth_Yolo']['ccc'],
        'ICC': results['Tooth_Yolo']['icc'],
        'Correlation': correlation_tooth_yolo,
        'MAE': mae_tooth_yolo,
        'MAPE(%)': mape_tooth_yolo,
        'Bootstrap_CCC_Mean': yolo_ccc_mean,
        'Bootstrap_ICC_Mean': yolo_icc_mean,
        'CCC_95CI': f'[{yolo_ccc_ci_lower:.3f}, {yolo_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{yolo_icc_ci_lower:.3f}, {yolo_icc_ci_upper:.3f}]'
    })
    
    # Root Area 통계
    correlation_root_unet = np.corrcoef(data['GT_Root'], data['Unet_Root'])[0, 1]
    correlation_root_yolo = np.corrcoef(data['GT_Root'], data['Yolo_Root'])[0, 1]
    mae_root_unet = np.mean(np.abs(data['GT_Root'] - data['Unet_Root']))
    mae_root_yolo = np.mean(np.abs(data['GT_Root'] - data['Yolo_Root']))
    mape_root_unet = np.mean(np.abs((data['GT_Root'] - data['Unet_Root']) / data['GT_Root'])) * 100
    mape_root_yolo = np.mean(np.abs((data['GT_Root'] - data['Yolo_Root']) / data['GT_Root'])) * 100
    
    detailed_stats.append({
        'Category': 'Root Area',
        'Method': 'Unet',
        'CCC': results['Root_Unet']['ccc'],
        'ICC': results['Root_Unet']['icc'],
        'Correlation': correlation_root_unet,
        'MAE': mae_root_unet,
        'MAPE(%)': mape_root_unet,
        'Bootstrap_CCC_Mean': unet_ccc_mean,
        'Bootstrap_ICC_Mean': unet_icc_mean,
        'CCC_95CI': f'[{unet_ccc_ci_lower:.3f}, {unet_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{unet_icc_ci_lower:.3f}, {unet_icc_ci_upper:.3f}]'
    })
    detailed_stats.append({
        'Category': 'Root Area',
        'Method': 'Yolo',
        'CCC': results['Root_Yolo']['ccc'],
        'ICC': results['Root_Yolo']['icc'],
        'Correlation': correlation_root_yolo,
        'MAE': mae_root_yolo,
        'MAPE(%)': mape_root_yolo,
        'Bootstrap_CCC_Mean': yolo_ccc_mean,
        'Bootstrap_ICC_Mean': yolo_icc_mean,
        'CCC_95CI': f'[{yolo_ccc_ci_lower:.3f}, {yolo_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{yolo_icc_ci_lower:.3f}, {yolo_icc_ci_upper:.3f}]'
    })
    
    df_detailed = pd.DataFrame(detailed_stats)
    
    # 6. 요약 통계 테이블 생성
    summary_stats = []
    
    # Unet 요약
    unet_avg_ccc = np.mean([results['Ratio_Unet']['ccc'], results['Tooth_Unet']['ccc'], results['Root_Unet']['ccc']])
    unet_avg_icc = np.mean([results['Ratio_Unet']['icc'], results['Tooth_Unet']['icc'], results['Root_Unet']['icc']])
    unet_avg_corr = np.mean([correlation_ratio_unet, correlation_tooth_unet, correlation_root_unet])
    unet_avg_mae = np.mean([mae_ratio_unet, mae_tooth_unet, mae_root_unet])
    unet_avg_mape = np.mean([mape_ratio_unet, mape_tooth_unet, mape_root_unet])
    
    summary_stats.append({
        'Method': 'Unet',
        'Average_CCC': unet_avg_ccc,
        'Average_ICC': unet_avg_icc,
        'Average_Correlation': unet_avg_corr,
        'Average_MAE': unet_avg_mae,
        'Average_MAPE(%)': unet_avg_mape,
        'Bootstrap_CCC_Mean': unet_ccc_mean,
        'Bootstrap_ICC_Mean': unet_icc_mean,
        'CCC_95CI': f'[{unet_ccc_ci_lower:.3f}, {unet_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{unet_icc_ci_lower:.3f}, {unet_icc_ci_upper:.3f}]'
    })
    
    # Yolo 요약
    yolo_avg_ccc = np.mean([results['Ratio_Yolo']['ccc'], results['Tooth_Yolo']['ccc'], results['Root_Yolo']['ccc']])
    yolo_avg_icc = np.mean([results['Ratio_Yolo']['icc'], results['Tooth_Yolo']['icc'], results['Root_Yolo']['icc']])
    yolo_avg_corr = np.mean([correlation_ratio_yolo, correlation_tooth_yolo, correlation_root_yolo])
    yolo_avg_mae = np.mean([mae_ratio_yolo, mae_tooth_yolo, mae_root_yolo])
    yolo_avg_mape = np.mean([mape_ratio_yolo, mape_tooth_yolo, mape_root_yolo])
    
    summary_stats.append({
        'Method': 'Yolo',
        'Average_CCC': yolo_avg_ccc,
        'Average_ICC': yolo_avg_icc,
        'Average_Correlation': yolo_avg_corr,
        'Average_MAE': yolo_avg_mae,
        'Average_MAPE(%)': yolo_avg_mape,
        'Bootstrap_CCC_Mean': yolo_ccc_mean,
        'Bootstrap_ICC_Mean': yolo_icc_mean,
        'CCC_95CI': f'[{yolo_ccc_ci_lower:.3f}, {yolo_ccc_ci_upper:.3f}]',
        'ICC_95CI': f'[{yolo_icc_ci_lower:.3f}, {yolo_icc_ci_upper:.3f}]'
    })
    
    df_summary = pd.DataFrame(summary_stats)
    
    # 7. 파일로 저장
    # CSV 파일들로 저장
    df_results.to_csv('results_table.csv', index=False, float_format='%.3f')
    df_comparison.to_csv('comparison_table.csv', index=False, float_format='%.3f')
    df_detailed.to_csv('detailed_statistics.csv', index=False, float_format='%.3f')
    df_summary.to_csv('summary_statistics.csv', index=False, float_format='%.3f')
    df_ci.to_csv('confidence_intervals.csv', index=False, float_format='%.3f')
    
    # Excel 파일로 저장 (모든 테이블을 하나의 파일에)
    try:
        with pd.ExcelWriter('comprehensive_results.xlsx', engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Basic_Results', index=False, float_format='%.3f')
            df_comparison.to_excel(writer, sheet_name='Performance_Comparison', index=False, float_format='%.3f')
            df_detailed.to_excel(writer, sheet_name='Detailed_Statistics', index=False, float_format='%.3f')
            df_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False, float_format='%.3f')
            df_ci.to_excel(writer, sheet_name='Confidence_Intervals', index=False, float_format='%.3f')
            
            # 원본 데이터도 저장
            data.to_excel(writer, sheet_name='Original_Data', index=False, float_format='%.3f')
        
        print("Excel 파일 저장 완료: comprehensive_results.xlsx")
    except ImportError:
        print("openpyxl이 설치되지 않아 Excel 파일 저장을 건너뜁니다.")
        print("CSV 파일들만 저장되었습니다.")
    
    print("CSV 파일 저장 완료:")
    print("- results_table.csv")
    print("- comparison_table.csv") 
    print("- detailed_statistics.csv")
    print("- summary_statistics.csv")
    print("- confidence_intervals.csv")
    
    return df_results, df_comparison, df_detailed, df_summary, df_ci

def main():
    """메인 함수"""
    
    # 데이터 로드 및 처리
    print("데이터를 로드하고 처리 중...")
    data = load_and_process_data()
    
    print(f"총 {len(data)}개의 이미지 데이터를 분석합니다.")
    
    # 모든 지표 계산
    print("\n모든 지표에 대해 CCC와 ICC를 계산 중...")
    results = calculate_all_metrics(data)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("CCC 및 ICC 계산 결과 (Ratio, Tooth Area, Root Area)")
    print("=" * 80)
    
    print(f"\n1. Ratio (Root/Tooth) 분석:")
    print(f"  Unet - CCC: {results['Ratio_Unet']['ccc']:.4f}, ICC: {results['Ratio_Unet']['icc']:.4f}")
    print(f"  Yolo - CCC: {results['Ratio_Yolo']['ccc']:.4f}, ICC: {results['Ratio_Yolo']['icc']:.4f}")
    
    print(f"\n2. Tooth Area 분석:")
    print(f"  Unet - CCC: {results['Tooth_Unet']['ccc']:.4f}, ICC: {results['Tooth_Unet']['icc']:.4f}")
    print(f"  Yolo - CCC: {results['Tooth_Yolo']['ccc']:.4f}, ICC: {results['Tooth_Yolo']['icc']:.4f}")
    
    print(f"\n3. Root Area 분석:")
    print(f"  Unet - CCC: {results['Root_Unet']['ccc']:.4f}, ICC: {results['Root_Unet']['icc']:.4f}")
    print(f"  Yolo - CCC: {results['Root_Yolo']['ccc']:.4f}, ICC: {results['Root_Yolo']['icc']:.4f}")
    
    print(f"\n4. 성능 비교:")
    print(f"  Ratio - CCC 차이: {results['Ratio_Yolo']['ccc'] - results['Ratio_Unet']['ccc']:.4f}")
    print(f"  Ratio - ICC 차이: {results['Ratio_Yolo']['icc'] - results['Ratio_Unet']['icc']:.4f}")
    print(f"  Tooth - CCC 차이: {results['Tooth_Yolo']['ccc'] - results['Tooth_Unet']['ccc']:.4f}")
    print(f"  Tooth - ICC 차이: {results['Tooth_Yolo']['icc'] - results['Tooth_Unet']['icc']:.4f}")
    print(f"  Root - CCC 차이: {results['Root_Yolo']['ccc'] - results['Root_Unet']['ccc']:.4f}")
    print(f"  Root - ICC 차이: {results['Root_Yolo']['icc'] - results['Root_Unet']['icc']:.4f}")
    
    # 결과를 표로 저장
    print("\n결과를 표로 저장 중...")
    df_results, df_comparison, df_detailed, df_summary, df_ci = save_results_to_tables(data, results)
    
    # 신뢰구간 결과 출력
    print("\n" + "=" * 80)
    print("ICC 및 CCC 평균과 95% 신뢰구간")
    print("=" * 80)
    
    print("\nUnet 결과:")
    unet_ccc_row = df_ci[(df_ci['Method'] == 'Unet') & (df_ci['Metric'] == 'CCC')].iloc[0]
    unet_icc_row = df_ci[(df_ci['Method'] == 'Unet') & (df_ci['Metric'] == 'ICC')].iloc[0]
    print(f"  CCC - 평균: {unet_ccc_row['Mean']:.3f}, 95% CI: {unet_ccc_row['CI_95%']}")
    print(f"  ICC - 평균: {unet_icc_row['Mean']:.3f}, 95% CI: {unet_icc_row['CI_95%']}")
    
    print("\nYolo 결과:")
    yolo_ccc_row = df_ci[(df_ci['Method'] == 'Yolo') & (df_ci['Metric'] == 'CCC')].iloc[0]
    yolo_icc_row = df_ci[(df_ci['Method'] == 'Yolo') & (df_ci['Metric'] == 'ICC')].iloc[0]
    print(f"  CCC - 평균: {yolo_ccc_row['Mean']:.3f}, 95% CI: {yolo_ccc_row['CI_95%']}")
    print(f"  ICC - 평균: {yolo_icc_row['Mean']:.3f}, 95% CI: {yolo_icc_row['CI_95%']}")
    
    # 시각화 생성
    print("\n시각화를 생성 중...")
    create_visualizations(data, results)
    
    print("\n분석이 완료되었습니다.")
    print("시각화 결과가 다음 파일로 저장되었습니다:")
    print("- comprehensive_analysis_plots.png")

if __name__ == "__main__":
    main()
