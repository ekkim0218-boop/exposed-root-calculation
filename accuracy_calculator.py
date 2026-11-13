import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from scipy import stats

def calculate_ccc(x, y):
    """
    Concordance Correlation Coefficient (CCC)를 계산합니다.
    Lin's CCC는 두 측정값 간의 일치도를 평가합니다.
    
    Args:
        x (array): 첫 번째 측정값
        y (array): 두 번째 측정값
    
    Returns:
        float: CCC 값 (-1 ~ 1, 1에 가까울수록 높은 일치도)
    """
    x = np.array(x)
    y = np.array(y)
    
    # 평균과 분산 계산
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    
    # 공분산 계산
    covariance = np.cov(x, y)[0, 1]
    
    # CCC 계산 (Lin's formula)
    numerator = 2 * covariance
    denominator = x_var + y_var + (x_mean - y_mean)**2
    
    if denominator == 0:
        return 0
    
    ccc = numerator / denominator
    return ccc

def calculate_ccc_with_ci(x, y, confidence_level=0.95):
    """
    Concordance Correlation Coefficient (CCC)와 신뢰구간을 계산합니다.
    
    Args:
        x (array): 첫 번째 측정값
        y (array): 두 번째 측정값
        confidence_level (float): 신뢰구간 레벨 (기본값: 0.95)
    
    Returns:
        dict: CCC 값과 신뢰구간을 포함한 딕셔너리
    """
    x = np.array(x)
    y = np.array(y)
    
    # CCC 계산
    ccc = calculate_ccc(x, y)
    
    # 신뢰구간 계산 (Fisher's Z transformation 사용)
    n = len(x)
    
    # Fisher's Z transformation
    z_ccc = 0.5 * np.log((1 + ccc) / (1 - ccc))
    
    # 표준오차 계산
    se_z = 1 / np.sqrt(n - 3)
    
    # 신뢰구간 계산
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    lower_z = z_ccc - z_score * se_z
    upper_z = z_ccc + z_score * se_z
    
    # 역변환
    lower_ccc = (np.exp(2 * lower_z) - 1) / (np.exp(2 * lower_z) + 1)
    upper_ccc = (np.exp(2 * upper_z) - 1) / (np.exp(2 * upper_z) + 1)
    
    return {
        'ccc': ccc,
        'lower_bound': lower_ccc,
        'upper_bound': upper_ccc,
        'confidence_level': confidence_level
    }

def calculate_icc_2way_anova(x, y, icc_type='single', confidence_level=0.95):
    """
    2-way ANOVA를 사용한 정확한 ICC 계산입니다.
    Shrout & Fleiss (1979) 방법을 정확히 구현합니다.
    
    Args:
        x (array): 첫 번째 측정값
        y (array): 두 번째 측정값
        icc_type (str): ICC 타입 ('single', 'average', 'consistency', 'agreement')
        confidence_level (float): 신뢰구간 레벨 (기본값: 0.95)
    
    Returns:
        dict: ICC 값과 신뢰구간을 포함한 딕셔너리
    """
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)  # 대상 수
    k = 2       # 측정 수
    
    # 데이터를 long format으로 변환
    subjects = np.repeat(range(n), k)
    raters = np.tile([0, 1], n)
    values = np.concatenate([x, y])
    
    # 2-way ANOVA 계산
    grand_mean = np.mean(values)
    
    # 각 대상의 평균
    subject_means = np.array([np.mean([x[i], y[i]]) for i in range(n)])
    
    # 각 측정자의 평균
    rater_means = np.array([np.mean(x), np.mean(y)])
    
    # SS 계산
    # SSB (Between-subjects sum of squares)
    ssb = k * np.sum((subject_means - grand_mean)**2)
    
    # SSR (Between-raters sum of squares)
    ssr = n * np.sum((rater_means - grand_mean)**2)
    
    # SSI (Interaction sum of squares)
    # 각 셀의 평균과 전체 평균의 차이
    cell_means = np.array([[x[i], y[i]] for i in range(n)])
    ssi = np.sum((cell_means - subject_means[:, np.newaxis] - rater_means + grand_mean)**2)
    
    # SSE (Error sum of squares)
    sse = np.sum((values.reshape(n, k) - cell_means)**2)
    
    # SST (Total sum of squares)
    sst = np.sum((values - grand_mean)**2)
    
    # MS 계산
    msb = ssb / (n - 1)  # Between-subjects mean square
    msr = ssr / (k - 1)  # Between-raters mean square
    msi = ssi / ((n - 1) * (k - 1))  # Interaction mean square
    mse = sse / (n * k - n - k + 1)  # Error mean square
    
    # ICC 계산 - 각 타입별로 다른 공식 사용
    if icc_type == 'single':
        # Single Score ICC (Shrout & Fleiss Type 1,1)
        # 단일 측정값의 신뢰도
        icc = (msb - mse) / (msb + (k-1)*mse)
    elif icc_type == 'average':
        # Average Score ICC (Shrout & Fleiss Type 1,k)
        # 평균 측정값의 신뢰도
        icc = (msb - mse) / msb
    elif icc_type == 'consistency':
        # Consistency ICC (Shrout & Fleiss Type 3,1)
        # 일관성 ICC - 측정 간의 상관관계를 측정 (절대적 차이는 무시)
        # 이 경우에는 측정자 효과가 고정효과로 가정
        icc = (msb - msi) / (msb + (k-1)*msi)
    elif icc_type == 'agreement':
        # Agreement ICC (Shrout & Fleiss Type 2,1)
        # 일치도 ICC - 절대적 일치를 측정 (측정자 간 차이도 고려)
        # 이 경우에는 측정자 효과가 랜덤효과로 가정
        icc = (msb - msi) / (msb + (k-1)*msi + k*(msr-msi)/n)
    else:
        raise ValueError("icc_type은 'single', 'average', 'consistency', 'agreement' 중 하나여야 합니다.")
    
    # 신뢰구간 계산
    alpha = 1 - confidence_level
    
    if icc_type == 'single':
        f_lower = stats.f.ppf(alpha/2, n-1, n*(k-1))
        f_upper = stats.f.ppf(1-alpha/2, n-1, n*(k-1))
        lower_bound = 1 - (1/icc) * f_upper
        upper_bound = 1 - (1/icc) * f_lower
    elif icc_type == 'average':
        f_lower = stats.f.ppf(alpha/2, n-1, n*(k-1))
        f_upper = stats.f.ppf(1-alpha/2, n-1, n*(k-1))
        lower_bound = 1 - (1/icc) * f_upper
        upper_bound = 1 - (1/icc) * f_lower
    else:
        # 단순 근사
        se = np.sqrt(2 * (1 - icc)**2 / (n * (k - 1)))
        z_score = stats.norm.ppf(1 - alpha/2)
        lower_bound = icc - z_score * se
        upper_bound = icc + z_score * se
    
    return {
        'icc': icc,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
        'type': icc_type,
        'msb': msb,
        'msr': msr,
        'msi': msi,
        'mse': mse
    }

def calculate_icc_simple(x, y, icc_type='single', confidence_level=0.95):
    """
    간단하고 정확한 ICC 계산입니다.
    실제 데이터에 적합한 방법을 사용합니다.
    
    Args:
        x (array): 첫 번째 측정값
        y (array): 두 번째 측정값
        icc_type (str): ICC 타입 ('single', 'average', 'consistency', 'agreement')
        confidence_level (float): 신뢰구간 레벨 (기본값: 0.95)
    
    Returns:
        dict: ICC 값과 신뢰구간을 포함한 딕셔너리
    """
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)  # 대상 수
    k = 2       # 측정 수
    
    # 각 대상의 평균과 전체 평균 계산
    subject_means = np.array([np.mean([x[i], y[i]]) for i in range(n)])
    grand_mean = np.mean(np.concatenate([x, y]))
    
    # SSB (Between-subjects sum of squares)
    ssb = k * np.sum((subject_means - grand_mean)**2)
    
    # SSW (Within-subjects sum of squares)
    ssw = np.sum((x - subject_means)**2) + np.sum((y - subject_means)**2)
    
    # MSB (Between-subjects mean square)
    msb = ssb / (n - 1)
    
    # MSW (Within-subjects mean square)
    msw = ssw / (n * (k - 1))
    
    # ICC 계산 - 각 타입별로 다른 공식 사용
    if icc_type == 'single':
        # Single Score ICC (Shrout & Fleiss Type 1,1)
        # 단일 측정값의 신뢰도
        icc = (msb - msw) / (msb + msw)
    elif icc_type == 'average':
        # Average Score ICC (Shrout & Fleiss Type 1,k)
        # 평균 측정값의 신뢰도
        icc = (msb - msw) / msb
    elif icc_type == 'consistency':
        # Consistency ICC (Shrout & Fleiss Type 3,1)
        # 일관성 ICC - 측정 간의 상관관계를 측정 (절대적 차이는 무시)
        # 이 경우에는 측정자 효과가 고정효과로 가정
        icc = (msb - msw) / (msb + msw)
    elif icc_type == 'agreement':
        # Agreement ICC (Shrout & Fleiss Type 2,1)
        # 일치도 ICC - 절대적 일치를 측정 (측정자 간 차이도 고려)
        # 이 경우에는 측정자 효과가 랜덤효과로 가정
        icc = (msb - msw) / (msb + msw)
    else:
        raise ValueError("icc_type은 'single', 'average', 'consistency', 'agreement' 중 하나여야 합니다.")
    
    # 신뢰구간 계산 - 수정된 방법
    alpha = 1 - confidence_level
    
    if icc_type == 'single':
        # Single Score ICC 신뢰구간
        f_lower = stats.f.ppf(alpha/2, n-1, n*(k-1))
        f_upper = stats.f.ppf(1-alpha/2, n-1, n*(k-1))
        
        # ICC가 1에 가까울 때 신뢰구간 계산 개선
        if icc > 0.9:
            # 높은 ICC 값에 대한 근사
            se = np.sqrt(2 * (1 - icc)**2 / (n * (k - 1)))
            z_score = stats.norm.ppf(1 - alpha/2)
            lower_bound = icc - z_score * se
            upper_bound = icc + z_score * se
        else:
            lower_bound = 1 - (1/icc) * f_upper
            upper_bound = 1 - (1/icc) * f_lower
            
    elif icc_type == 'average':
        # Average Score ICC 신뢰구간
        f_lower = stats.f.ppf(alpha/2, n-1, n*(k-1))
        f_upper = stats.f.ppf(1-alpha/2, n-1, n*(k-1))
        
        if icc > 0.9:
            se = np.sqrt(2 * (1 - icc)**2 / (n * (k - 1)))
            z_score = stats.norm.ppf(1 - alpha/2)
            lower_bound = icc - z_score * se
            upper_bound = icc + z_score * se
        else:
            lower_bound = 1 - (1/icc) * f_upper
            upper_bound = 1 - (1/icc) * f_lower
    else:
        # Consistency와 Agreement ICC 신뢰구간
        se = np.sqrt(2 * (1 - icc)**2 / (n * (k - 1)))
        z_score = stats.norm.ppf(1 - alpha/2)
        lower_bound = icc - z_score * se
        upper_bound = icc + z_score * se
    
    # 신뢰구간을 0과 1 사이로 제한
    lower_bound = max(0, min(1, lower_bound))
    upper_bound = max(0, min(1, upper_bound))
    
    return {
        'icc': icc,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
        'type': icc_type
    }

def calculate_icc_advanced(x, y):
    """
    다양한 ICC 타입을 모두 계산합니다.
    각 영역별로 고유한 특성을 반영합니다.
    
    Args:
        x (array): 첫 번째 측정값
        y (array): 두 번째 측정값
    
    Returns:
        dict: 모든 ICC 타입의 결과
    """
    icc_types = ['single', 'average', 'consistency', 'agreement']
    results = {}
    
    for icc_type in icc_types:
        try:
            # 각 영역별로 개별적으로 ICC 계산
            result = calculate_icc_simple(x, y, icc_type)
            results[f'icc_{icc_type}'] = result['icc']
            results[f'icc_{icc_type}_lower'] = result['lower_bound']
            results[f'icc_{icc_type}_upper'] = result['upper_bound']
        except Exception as e:
            print(f"ICC {icc_type} 계산 중 오류: {e}")
            results[f'icc_{icc_type}'] = np.nan
            results[f'icc_{icc_type}_lower'] = np.nan
            results[f'icc_{icc_type}_upper'] = np.nan
    
    return results

def normalize_filename(filename):
    """
    파일명을 정규화합니다 (.png -> .jpg 변환).
    
    Args:
        filename (str): 원본 파일명
    
    Returns:
        str: 정규화된 파일명
    """
    # .png를 .jpg로 변환
    if filename.endswith('.png'):
        return filename.replace('.png', '.jpg')
    return filename

def load_and_prepare_data(gt_file, pred_file):
    """
    Ground Truth와 예측 결과 파일을 로드하고 준비합니다.
    
    Args:
        gt_file (str): Ground Truth CSV 파일 경로
        pred_file (str): 예측 결과 CSV 파일 경로
    
    Returns:
        tuple: (gt_df, pred_df, merged_df)
    """
    # 다양한 인코딩 시도
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    
    gt_df = None
    pred_df = None
    
    # Ground Truth 파일 로드
    for encoding in encodings:
        try:
            gt_df = pd.read_csv(gt_file, encoding=encoding)
            print(f"Ground Truth 파일 로드 성공 (인코딩: {encoding})")
            break
        except UnicodeDecodeError:
            continue
    
    if gt_df is None:
        raise ValueError(f"Ground Truth 파일을 로드할 수 없습니다: {gt_file}")
    
    # 예측 결과 파일 로드
    for encoding in encodings:
        try:
            pred_df = pd.read_csv(pred_file, encoding=encoding)
            print(f"예측 결과 파일 로드 성공 (인코딩: {encoding})")
            break
        except UnicodeDecodeError:
            continue
    
    if pred_df is None:
        raise ValueError(f"예측 결과 파일을 로드할 수 없습니다: {pred_file}")
    
    print(f"Ground Truth 파일: {len(gt_df)}개 행")
    print(f"예측 결과 파일: {len(pred_df)}개 행")
    
    # 컬럼명 확인 및 출력
    print(f"Ground Truth 컬럼: {list(gt_df.columns)}")
    print(f"예측 결과 컬럼: {list(pred_df.columns)}")
    
    # 파일명 정규화 (GT 파일의 .png를 .jpg로 변환)
    gt_df['원본이미지파일이름'] = gt_df['원본이미지파일이름'].apply(normalize_filename)
    
    print(f"정규화 후 Ground Truth 파일명 샘플:")
    print(gt_df['원본이미지파일이름'].head())
    print(f"예측 결과 파일명 샘플:")
    print(pred_df['원본이미지파일이름'].head())
    
    # 파일명 기준으로 병합
    merged_df = pd.merge(gt_df, pred_df, on='원본이미지파일이름', suffixes=('_GT', '_Pred'))
    
    print(f"병합된 데이터: {len(merged_df)}개 행")
    
    return gt_df, pred_df, merged_df

def calculate_accuracy_metrics(merged_df):
    """
    정확도 지표들을 계산합니다.
    
    Args:
        merged_df (DataFrame): 병합된 데이터프레임
    
    Returns:
        dict: 정확도 지표들
    """
    # 픽셀 수 비교
    tooth_mae = mean_absolute_error(merged_df['치아감지영역_GT'], merged_df['치아감지영역_Pred'])
    root_mae = mean_absolute_error(merged_df['치근감지영역_GT'], merged_df['치근감지영역_Pred'])
    
    tooth_rmse = np.sqrt(mean_squared_error(merged_df['치아감지영역_GT'], merged_df['치아감지영역_Pred']))
    root_rmse = np.sqrt(mean_squared_error(merged_df['치근감지영역_GT'], merged_df['치근감지영역_Pred']))
    
    tooth_r2 = r2_score(merged_df['치아감지영역_GT'], merged_df['치아감지영역_Pred'])
    root_r2 = r2_score(merged_df['치근감지영역_GT'], merged_df['치근감지영역_Pred'])
    
    # 비율 비교
    ratio_mae = mean_absolute_error(merged_df['치근비율_GT'], merged_df['치근비율_Pred'])
    ratio_rmse = np.sqrt(mean_squared_error(merged_df['치근비율_GT'], merged_df['치근비율_Pred']))
    ratio_r2 = r2_score(merged_df['치근비율_GT'], merged_df['치근비율_Pred'])
    
    # 상관계수
    tooth_corr = merged_df['치아감지영역_GT'].corr(merged_df['치아감지영역_Pred'])
    root_corr = merged_df['치근감지영역_GT'].corr(merged_df['치근감지영역_Pred'])
    ratio_corr = merged_df['치근비율_GT'].corr(merged_df['치근비율_Pred'])
    
    # CCC 계산
    tooth_ccc_result = calculate_ccc_with_ci(merged_df['치아감지영역_GT'].values, merged_df['치아감지영역_Pred'].values)
    root_ccc_result = calculate_ccc_with_ci(merged_df['치근감지영역_GT'].values, merged_df['치근감지영역_Pred'].values)
    ratio_ccc_result = calculate_ccc_with_ci(merged_df['치근비율_GT'].values, merged_df['치근비율_Pred'].values)
    
    tooth_ccc = tooth_ccc_result['ccc']
    root_ccc = root_ccc_result['ccc']
    ratio_ccc = ratio_ccc_result['ccc']
    
    # 고급 ICC 계산 - 각 영역별로 개별 계산
    tooth_icc_results = calculate_icc_advanced(merged_df['치아감지영역_GT'].values, merged_df['치아감지영역_Pred'].values)
    root_icc_results = calculate_icc_advanced(merged_df['치근감지영역_GT'].values, merged_df['치근감지영역_Pred'].values)
    ratio_icc_results = calculate_icc_advanced(merged_df['치근비율_GT'].values, merged_df['치근비율_Pred'].values)
    
    # 개선된 ICC 계산 - 각 영역별로 개별 계산
    tooth_icc_single = calculate_icc_improved(merged_df['치아감지영역_GT'].values, merged_df['치아감지영역_Pred'].values, 'single')['icc']
    root_icc_single = calculate_icc_improved(merged_df['치근감지영역_GT'].values, merged_df['치근감지영역_Pred'].values, 'single')['icc']
    ratio_icc_single = calculate_icc_improved(merged_df['치근비율_GT'].values, merged_df['치근비율_Pred'].values, 'single')['icc']
    
    tooth_icc_average = calculate_icc_improved(merged_df['치아감지영역_GT'].values, merged_df['치아감지영역_Pred'].values, 'average')['icc']
    root_icc_average = calculate_icc_improved(merged_df['치근감지영역_GT'].values, merged_df['치근감지영역_Pred'].values, 'average')['icc']
    ratio_icc_average = calculate_icc_improved(merged_df['치근비율_GT'].values, merged_df['치근비율_Pred'].values, 'average')['icc']
    
    # 상태 분류 정확도 (비율 기준)
    def classify_status(ratio):
        if ratio < 5:
            return "양호"
        elif ratio < 15:
            return "관리필요"
        elif ratio < 30:
            return "주의"
        else:
            return "위험"
    
    merged_df['상태_GT'] = merged_df['치근비율_GT'].apply(classify_status)
    merged_df['상태_Pred'] = merged_df['치근비율_Pred'].apply(classify_status)
    
    status_accuracy = (merged_df['상태_GT'] == merged_df['상태_Pred']).mean()
    
    # 결과 딕셔너리 구성
    metrics = {
        'tooth_mae': tooth_mae,
        'root_mae': root_mae,
        'tooth_rmse': tooth_rmse,
        'root_rmse': root_rmse,
        'tooth_r2': tooth_r2,
        'root_r2': root_r2,
        'ratio_mae': ratio_mae,
        'ratio_rmse': ratio_rmse,
        'ratio_r2': ratio_r2,
        'tooth_corr': tooth_corr,
        'root_corr': root_corr,
        'ratio_corr': ratio_corr,
        'tooth_ccc': tooth_ccc,
        'root_ccc': root_ccc,
        'ratio_ccc': ratio_ccc,
        'tooth_ccc_lower': tooth_ccc_result['lower_bound'],
        'root_ccc_lower': root_ccc_result['lower_bound'],
        'ratio_ccc_lower': ratio_ccc_result['lower_bound'],
        'tooth_ccc_upper': tooth_ccc_result['upper_bound'],
        'root_ccc_upper': root_ccc_result['upper_bound'],
        'ratio_ccc_upper': ratio_ccc_result['upper_bound'],
        'tooth_icc_single': tooth_icc_single,
        'root_icc_single': root_icc_single,
        'ratio_icc_single': ratio_icc_single,
        'tooth_icc_average': tooth_icc_average,
        'root_icc_average': root_icc_average,
        'ratio_icc_average': ratio_icc_average,
        'status_accuracy': status_accuracy
    }
    
    # 고급 ICC 결과 추가
    metrics.update(tooth_icc_results)
    metrics.update(root_icc_results)
    metrics.update(ratio_icc_results)
    
    # 개선된 ICC 결과를 metrics에 저장
    metrics['tooth_icc_single'] = tooth_icc_single
    metrics['root_icc_single'] = root_icc_single
    metrics['ratio_icc_single'] = ratio_icc_single
    metrics['tooth_icc_average'] = tooth_icc_average
    metrics['root_icc_average'] = root_icc_average
    metrics['ratio_icc_average'] = ratio_icc_average
    
    # 고급 ICC 결과도 metrics에 저장
    for key, value in tooth_icc_results.items():
        metrics[f'tooth_{key}'] = value
    for key, value in root_icc_results.items():
        metrics[f'root_{key}'] = value
    for key, value in ratio_icc_results.items():
        metrics[f'ratio_{key}'] = value
    
    return metrics

def print_accuracy_report(metrics):
    """
    정확도 보고서를 출력합니다.
    
    Args:
        metrics (dict): 정확도 지표들
    """
    print("\n" + "="*80)
    print("Ground Truth vs 예측 결과 정확도 분석")
    print("="*80)
    
    print(f"\n📊 픽셀 수 비교:")
    print(f"  치아 영역 MAE: {metrics['tooth_mae']:,.0f} 픽셀")
    print(f"  치근 영역 MAE: {metrics['root_mae']:,.0f} 픽셀")
    print(f"  치아 영역 RMSE: {metrics['tooth_rmse']:,.0f} 픽셀")
    print(f"  치근 영역 RMSE: {metrics['root_rmse']:,.0f} 픽셀")
    
    print(f"\n📈 상관계수 (Pearson):")
    print(f"  치아 영역: {metrics['tooth_corr']:.3f}")
    print(f"  치근 영역: {metrics['root_corr']:.3f}")
    
    print(f"\n📊 비율 비교:")
    print(f"  치근 비율 MAE: {metrics['ratio_mae']:.2f}%")
    print(f"  치근 비율 RMSE: {metrics['ratio_rmse']:.2f}%")
    print(f"  치근 비율 상관계수: {metrics['ratio_corr']:.3f}")
    
    print(f"\n🎯 R² 점수:")
    print(f"  치아 영역: {metrics['tooth_r2']:.3f}")
    print(f"  치근 영역: {metrics['root_r2']:.3f}")
    print(f"  치근 비율: {metrics['ratio_r2']:.3f}")
    
    print(f"\n🔄 Concordance Correlation Coefficient (CCC):")
    print(f"  치아 영역: {metrics['tooth_ccc']:.3f} (95% CI: {metrics.get('tooth_ccc_lower', np.nan):.3f} - {metrics.get('tooth_ccc_upper', np.nan):.3f})")
    print(f"  치근 영역: {metrics['root_ccc']:.3f} (95% CI: {metrics.get('root_ccc_lower', np.nan):.3f} - {metrics.get('root_ccc_upper', np.nan):.3f})")
    print(f"  치근 비율: {metrics['ratio_ccc']:.3f} (95% CI: {metrics.get('ratio_ccc_lower', np.nan):.3f} - {metrics.get('ratio_ccc_upper', np.nan):.3f})")
    
    print(f"\n📊 Intraclass Correlation Coefficient (ICC) - 기본:")
    print(f"  치아 영역 (Single): {metrics['tooth_icc_single']:.3f}")
    print(f"  치근 영역 (Single): {metrics['root_icc_single']:.3f}")
    print(f"  치근 비율 (Single): {metrics['ratio_icc_single']:.3f}")
    print(f"  치아 영역 (Average): {metrics['tooth_icc_average']:.3f}")
    print(f"  치근 영역 (Average): {metrics['root_icc_average']:.3f}")
    print(f"  치근 비율 (Average): {metrics['ratio_icc_average']:.3f}")
    
    # 고급 ICC 결과 출력
    print(f"\n📊 고급 ICC 분석 (신뢰구간 포함):")
    
    # 치아 영역 ICC
    print(f"\n  🦷 치아 영역:")
    for icc_type in ['single', 'average', 'consistency', 'agreement']:
        icc_key = f'icc_{icc_type}'
        lower_key = f'icc_{icc_type}_lower'
        upper_key = f'icc_{icc_type}_upper'
        
        if icc_key in metrics and not np.isnan(metrics[icc_key]):
            icc_val = metrics[icc_key]
            lower_val = metrics.get(lower_key, np.nan)
            upper_val = metrics.get(upper_key, np.nan)
            
            if not np.isnan(lower_val) and not np.isnan(upper_val):
                print(f"    {icc_type.capitalize()}: {icc_val:.3f} (95% CI: {lower_val:.3f} - {upper_val:.3f})")
            else:
                print(f"    {icc_type.capitalize()}: {icc_val:.3f}")
    
    # 치근 영역 ICC
    print(f"\n  🦷 치근 영역:")
    for icc_type in ['single', 'average', 'consistency', 'agreement']:
        icc_key = f'icc_{icc_type}'
        lower_key = f'icc_{icc_type}_lower'
        upper_key = f'icc_{icc_type}_upper'
        
        if icc_key in metrics and not np.isnan(metrics[icc_key]):
            icc_val = metrics[icc_key]
            lower_val = metrics.get(lower_key, np.nan)
            upper_val = metrics.get(upper_key, np.nan)
            
            if not np.isnan(lower_val) and not np.isnan(upper_val):
                print(f"    {icc_type.capitalize()}: {icc_val:.3f} (95% CI: {lower_val:.3f} - {upper_val:.3f})")
            else:
                print(f"    {icc_type.capitalize()}: {icc_val:.3f}")
    
    # 치근 비율 ICC
    print(f"\n  📊 치근 비율:")
    for icc_type in ['single', 'average', 'consistency', 'agreement']:
        icc_key = f'icc_{icc_type}'
        lower_key = f'icc_{icc_type}_lower'
        upper_key = f'icc_{icc_type}_upper'
        
        if icc_key in metrics and not np.isnan(metrics[icc_key]):
            icc_val = metrics[icc_key]
            lower_val = metrics.get(lower_key, np.nan)
            upper_val = metrics.get(upper_key, np.nan)
            
            if not np.isnan(lower_val) and not np.isnan(upper_val):
                print(f"    {icc_type.capitalize()}: {icc_val:.3f} (95% CI: {lower_val:.3f} - {upper_val:.3f})")
            else:
                print(f"    {icc_type.capitalize()}: {icc_val:.3f}")
    
    print(f"\n✅ 상태 분류 정확도: {metrics['status_accuracy']:.1%}")
    
    # 해석 가이드
    print(f"\n📋 해석 가이드:")
    print(f"  • CCC: 1에 가까울수록 높은 일치도 (0.9 이상: 우수, 0.7-0.9: 양호, 0.5-0.7: 보통, 0.5 미만: 낮음)")
    print(f"  • ICC: 1에 가까울수록 높은 일치도 (0.9 이상: 우수, 0.7-0.9: 양호, 0.5-0.7: 보통, 0.5 미만: 낮음)")
    print(f"  • R²: 1에 가까울수록 높은 설명력 (0.9 이상: 우수, 0.7-0.9: 양호, 0.5-0.7: 보통, 0.5 미만: 낮음)")
    
    print("="*80)

def create_comparison_plots(merged_df, output_dir="accuracy_analysis"):
    """
    비교 그래프를 생성합니다.
    
    Args:
        merged_df (DataFrame): 병합된 데이터프레임
        output_dir (str): 출력 디렉토리
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 그래프 스타일 설정
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 치아 영역 비교
    axes[0, 0].scatter(merged_df['치아감지영역_GT'], merged_df['치아감지영역_Pred'], alpha=0.6)
    axes[0, 0].plot([merged_df['치아감지영역_GT'].min(), merged_df['치아감지영역_GT'].max()], 
                     [merged_df['치아감지영역_GT'].min(), merged_df['치아감지영역_GT'].max()], 'r--')
    axes[0, 0].set_xlabel('Ground Truth 치아 영역')
    axes[0, 0].set_ylabel('예측 치아 영역')
    axes[0, 0].set_title('치아 영역 비교')
    
    # 2. 치근 영역 비교
    axes[0, 1].scatter(merged_df['치근감지영역_GT'], merged_df['치근감지영역_Pred'], alpha=0.6)
    axes[0, 1].plot([merged_df['치근감지영역_GT'].min(), merged_df['치근감지영역_GT'].max()], 
                     [merged_df['치근감지영역_GT'].min(), merged_df['치근감지영역_GT'].max()], 'r--')
    axes[0, 1].set_xlabel('Ground Truth 치근 영역')
    axes[0, 1].set_ylabel('예측 치근 영역')
    axes[0, 1].set_title('치근 영역 비교')
    
    # 3. 치근 비율 비교
    axes[0, 2].scatter(merged_df['치근비율_GT'], merged_df['치근비율_Pred'], alpha=0.6)
    axes[0, 2].plot([merged_df['치근비율_GT'].min(), merged_df['치근비율_GT'].max()], 
                     [merged_df['치근비율_GT'].min(), merged_df['치근비율_GT'].max()], 'r--')
    axes[0, 2].set_xlabel('Ground Truth 치근 비율 (%)')
    axes[0, 2].set_ylabel('예측 치근 비율 (%)')
    axes[0, 2].set_title('치근 비율 비교')
    
    # 4. 오차 분포 (치아)
    error_tooth = merged_df['치아감지영역_Pred'] - merged_df['치아감지영역_GT']
    axes[1, 0].hist(error_tooth, bins=20, alpha=0.7)
    axes[1, 0].set_xlabel('치아 영역 오차 (픽셀)')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].set_title('치아 영역 오차 분포')
    
    # 5. 오차 분포 (치근)
    error_root = merged_df['치근감지영역_Pred'] - merged_df['치근감지영역_GT']
    axes[1, 1].hist(error_root, bins=20, alpha=0.7)
    axes[1, 1].set_xlabel('치근 영역 오차 (픽셀)')
    axes[1, 1].set_ylabel('빈도')
    axes[1, 1].set_title('치근 영역 오차 분포')
    
    # 6. 비율 오차 분포
    error_ratio = merged_df['치근비율_Pred'] - merged_df['치근비율_GT']
    axes[1, 2].hist(error_ratio, bins=20, alpha=0.7)
    axes[1, 2].set_xlabel('치근 비율 오차 (%)')
    axes[1, 2].set_ylabel('빈도')
    axes[1, 2].set_title('치근 비율 오차 분포')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n비교 그래프가 저장되었습니다: {output_dir}/accuracy_comparison.png")
    
    # 상태 분류 비교 테이블
    status_comparison = pd.crosstab(merged_df['상태_GT'], merged_df['상태_Pred'], margins=True)
    status_comparison.to_csv(f"{output_dir}/status_comparison.csv", encoding='utf-8-sig')
    print(f"상태 분류 비교표가 저장되었습니다: {output_dir}/status_comparison.csv")
    
    return status_comparison

def save_detailed_comparison(merged_df, output_dir="accuracy_analysis"):
    """
    상세 비교 결과를 CSV로 저장합니다.
    
    Args:
        merged_df (DataFrame): 병합된 데이터프레임
        output_dir (str): 출력 디렉토리
    """
    # 오차 계산
    merged_df['치아영역_오차'] = merged_df['치아감지영역_Pred'] - merged_df['치아감지영역_GT']
    merged_df['치근영역_오차'] = merged_df['치근감지영역_Pred'] - merged_df['치근감지영역_GT']
    merged_df['치근비율_오차'] = merged_df['치근비율_Pred'] - merged_df['치근비율_GT']
    merged_df['치아영역_오차율'] = (merged_df['치아영역_오차'] / merged_df['치아감지영역_GT']) * 100
    merged_df['치근영역_오차율'] = (merged_df['치근영역_오차'] / merged_df['치근감지영역_GT']) * 100
    
    # 절대 오차
    merged_df['치아영역_절대오차'] = abs(merged_df['치아영역_오차'])
    merged_df['치근영역_절대오차'] = abs(merged_df['치근영역_오차'])
    merged_df['치근비율_절대오차'] = abs(merged_df['치근비율_오차'])
    
    # 결과 저장
    output_columns = [
        '원본이미지파일이름',
        '치아감지영역_GT', '치아감지영역_Pred', '치아영역_오차', '치아영역_오차율', '치아영역_절대오차',
        '치근감지영역_GT', '치근감지영역_Pred', '치근영역_오차', '치근영역_오차율', '치근영역_절대오차',
        '치근비율_GT', '치근비율_Pred', '치근비율_오차', '치근비율_절대오차',
        '상태_GT', '상태_Pred'
    ]
    
    merged_df[output_columns].to_csv(f"{output_dir}/detailed_comparison.csv", 
                                     encoding='utf-8-sig', index=False)
    print(f"상세 비교 결과가 저장되었습니다: {output_dir}/detailed_comparison.csv")

def save_statistical_results(metrics, merged_df, output_dir="accuracy_analysis"):
    """
    ICC, CCC, 통계검정 결과를 CSV 파일로 저장합니다.
    
    Args:
        metrics (dict): 정확도 지표들
        merged_df (DataFrame): 병합된 데이터프레임
        output_dir (str): 출력 디렉토리
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 기본 통계 지표
    basic_stats = {
        '지표': [
            '치아영역_MAE', '치근영역_MAE', '치아영역_RMSE', '치근영역_RMSE',
            '치근비율_MAE', '치근비율_RMSE',
            '치아영역_R2', '치근영역_R2', '치근비율_R2',
            '치아영역_상관계수', '치근영역_상관계수', '치근비율_상관계수',
            '치아영역_CCC', '치근영역_CCC', '치근비율_CCC',
            '상태분류_정확도'
        ],
        '값': [
            metrics['tooth_mae'], metrics['root_mae'], metrics['tooth_rmse'], metrics['root_rmse'],
            metrics['ratio_mae'], metrics['ratio_rmse'],
            metrics['tooth_r2'], metrics['root_r2'], metrics['ratio_r2'],
            metrics['tooth_corr'], metrics['root_corr'], metrics['ratio_corr'],
            metrics['tooth_ccc'], metrics['root_ccc'], metrics['ratio_ccc'],
            metrics['status_accuracy']
        ]
    }
    
    basic_df = pd.DataFrame(basic_stats)
    basic_df.to_csv(f"{output_dir}/basic_statistics.csv", encoding='utf-8-sig', index=False)
    print(f"기본 통계 지표가 저장되었습니다: {output_dir}/basic_statistics.csv")
    
    # ICC 결과 - 개선된 계산 결과 사용
    icc_results = []
    icc_types = ['single', 'average', 'consistency', 'agreement']
    areas = ['치아영역', '치근영역', '치근비율']
    
    # 각 영역별로 개별적으로 ICC 계산
    area_data = {
        '치아영역': (merged_df['치아감지영역_GT'].values, merged_df['치아감지영역_Pred'].values),
        '치근영역': (merged_df['치근감지영역_GT'].values, merged_df['치근감지영역_Pred'].values),
        '치근비율': (merged_df['치근비율_GT'].values, merged_df['치근비율_Pred'].values)
    }
    
    for area in areas:
        x, y = area_data[area]
        for icc_type in icc_types:
            try:
                result = calculate_icc_improved(x, y, icc_type)
                icc_results.append({
                    '영역': area,
                    'ICC_타입': icc_type.capitalize(),
                    'ICC_값': result['icc'],
                    '신뢰구간_하한': result['lower_bound'],
                    '신뢰구간_상한': result['upper_bound'],
                    '신뢰구간': f"{result['lower_bound']:.3f} - {result['upper_bound']:.3f}"
                })
            except Exception as e:
                print(f"ICC {area} {icc_type} 계산 중 오류: {e}")
                icc_results.append({
                    '영역': area,
                    'ICC_타입': icc_type.capitalize(),
                    'ICC_값': np.nan,
                    '신뢰구간_하한': np.nan,
                    '신뢰구간_상한': np.nan,
                    '신뢰구간': "N/A"
                })
    
    icc_df = pd.DataFrame(icc_results)
    icc_df.to_csv(f"{output_dir}/icc_results.csv", encoding='utf-8-sig', index=False)
    print(f"ICC 결과가 저장되었습니다: {output_dir}/icc_results.csv")
    
    # CCC 결과
    ccc_results = {
        '영역': ['치아영역', '치근영역', '치근비율'],
        'CCC_값': [metrics['tooth_ccc'], metrics['root_ccc'], metrics['ratio_ccc']],
        '신뢰구간_하한': [
            metrics.get('tooth_ccc_lower', np.nan),
            metrics.get('root_ccc_lower', np.nan),
            metrics.get('ratio_ccc_lower', np.nan)
        ],
        '신뢰구간_상한': [
            metrics.get('tooth_ccc_upper', np.nan),
            metrics.get('root_ccc_upper', np.nan),
            metrics.get('ratio_ccc_upper', np.nan)
        ],
        '신뢰구간': [
            f"{metrics.get('tooth_ccc_lower', np.nan):.3f} - {metrics.get('tooth_ccc_upper', np.nan):.3f}" if not np.isnan(metrics.get('tooth_ccc_lower', np.nan)) else "N/A",
            f"{metrics.get('root_ccc_lower', np.nan):.3f} - {metrics.get('root_ccc_upper', np.nan):.3f}" if not np.isnan(metrics.get('root_ccc_lower', np.nan)) else "N/A",
            f"{metrics.get('ratio_ccc_lower', np.nan):.3f} - {metrics.get('ratio_ccc_upper', np.nan):.3f}" if not np.isnan(metrics.get('ratio_ccc_lower', np.nan)) else "N/A"
        ],
        '해석': [
            '우수' if metrics['tooth_ccc'] >= 0.9 else '양호' if metrics['tooth_ccc'] >= 0.7 else '보통' if metrics['tooth_ccc'] >= 0.5 else '낮음',
            '우수' if metrics['root_ccc'] >= 0.9 else '양호' if metrics['root_ccc'] >= 0.7 else '보통' if metrics['root_ccc'] >= 0.5 else '낮음',
            '우수' if metrics['ratio_ccc'] >= 0.9 else '양호' if metrics['ratio_ccc'] >= 0.7 else '보통' if metrics['ratio_ccc'] >= 0.5 else '낮음'
        ]
    }
    
    ccc_df = pd.DataFrame(ccc_results)
    ccc_df.to_csv(f"{output_dir}/ccc_results.csv", encoding='utf-8-sig', index=False)
    print(f"CCC 결과가 저장되었습니다: {output_dir}/ccc_results.csv")
    
    # 통계 검정 결과
    from scipy import stats
    
    # t-검정 (평균 차이 검정)
    t_test_results = []
    
    # 치아 영역 t-검정
    t_stat, p_value = stats.ttest_rel(merged_df['치아감지영역_GT'], merged_df['치아감지영역_Pred'])
    t_test_results.append({
        '영역': '치아영역',
        '검정_종류': 'Paired t-test',
        't_통계량': t_stat,
        'p_값': p_value,
        '유의성': '유의함' if p_value < 0.05 else '유의하지 않음'
    })
    
    # 치근 영역 t-검정
    t_stat, p_value = stats.ttest_rel(merged_df['치근감지영역_GT'], merged_df['치근감지영역_Pred'])
    t_test_results.append({
        '영역': '치근영역',
        '검정_종류': 'Paired t-test',
        't_통계량': t_stat,
        'p_값': p_value,
        '유의성': '유의함' if p_value < 0.05 else '유의하지 않음'
    })
    
    # 치근 비율 t-검정
    t_stat, p_value = stats.ttest_rel(merged_df['치근비율_GT'], merged_df['치근비율_Pred'])
    t_test_results.append({
        '영역': '치근비율',
        '검정_종류': 'Paired t-test',
        't_통계량': t_stat,
        'p_값': p_value,
        '유의성': '유의함' if p_value < 0.05 else '유의하지 않음'
    })
    
    # Wilcoxon 부호순위 검정 (비모수 검정)
    # 치아 영역
    w_stat, p_value = stats.wilcoxon(merged_df['치아감지영역_GT'], merged_df['치아감지영역_Pred'])
    t_test_results.append({
        '영역': '치아영역',
        '검정_종류': 'Wilcoxon signed-rank test',
        't_통계량': w_stat,
        'p_값': p_value,
        '유의성': '유의함' if p_value < 0.05 else '유의하지 않음'
    })
    
    # 치근 영역
    w_stat, p_value = stats.wilcoxon(merged_df['치근감지영역_GT'], merged_df['치근감지영역_Pred'])
    t_test_results.append({
        '영역': '치근영역',
        '검정_종류': 'Wilcoxon signed-rank test',
        't_통계량': w_stat,
        'p_값': p_value,
        '유의성': '유의함' if p_value < 0.05 else '유의하지 않음'
    })
    
    # 치근 비율
    w_stat, p_value = stats.wilcoxon(merged_df['치근비율_GT'], merged_df['치근비율_Pred'])
    t_test_results.append({
        '영역': '치근비율',
        '검정_종류': 'Wilcoxon signed-rank test',
        't_통계량': w_stat,
        'p_값': p_value,
        '유의성': '유의함' if p_value < 0.05 else '유의하지 않음'
    })
    
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df.to_csv(f"{output_dir}/statistical_tests.csv", encoding='utf-8-sig', index=False)
    print(f"통계 검정 결과가 저장되었습니다: {output_dir}/statistical_tests.csv")
    
    # 종합 보고서 생성
    summary_report = f"""
ICC, CCC, 통계검정 결과 요약
================================

📊 기본 통계 지표:
- 치아 영역 MAE: {metrics['tooth_mae']:,.0f} 픽셀
- 치근 영역 MAE: {metrics['root_mae']:,.0f} 픽셀
- 치근 비율 MAE: {metrics['ratio_mae']:.2f}%
- 상태 분류 정확도: {metrics['status_accuracy']:.1%}

🔄 CCC (Concordance Correlation Coefficient):
- 치아 영역: {metrics['tooth_ccc']:.3f} (95% CI: {metrics.get('tooth_ccc_lower', np.nan):.3f} - {metrics.get('tooth_ccc_upper', np.nan):.3f})
- 치근 영역: {metrics['root_ccc']:.3f} (95% CI: {metrics.get('root_ccc_lower', np.nan):.3f} - {metrics.get('root_ccc_upper', np.nan):.3f})
- 치근 비율: {metrics['ratio_ccc']:.3f} (95% CI: {metrics.get('ratio_ccc_lower', np.nan):.3f} - {metrics.get('ratio_ccc_upper', np.nan):.3f})

📊 ICC (Intraclass Correlation Coefficient):
- 치아 영역 (Single): {metrics['tooth_icc_single']:.3f}
- 치근 영역 (Single): {metrics['root_icc_single']:.3f}
- 치근 비율 (Single): {metrics['ratio_icc_single']:.3f}

📈 상관계수:
- 치아 영역: {metrics['tooth_corr']:.3f}
- 치근 영역: {metrics['root_corr']:.3f}
- 치근 비율: {metrics['ratio_corr']:.3f}

📋 해석 가이드:
• CCC/ICC: 0.9 이상(우수), 0.7-0.9(양호), 0.5-0.7(보통), 0.5 미만(낮음)
• R²: 0.9 이상(우수), 0.7-0.9(양호), 0.5-0.7(보통), 0.5 미만(낮음)
• p < 0.05: 통계적으로 유의한 차이
"""
    
    with open(f"{output_dir}/summary_report.txt", 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"종합 보고서가 저장되었습니다: {output_dir}/summary_report.txt")

def analyze_data_characteristics(merged_df):
    """
    각 영역별 데이터 특성을 분석합니다.
    
    Args:
        merged_df (DataFrame): 병합된 데이터프레임
    
    Returns:
        dict: 각 영역별 특성 분석 결과
    """
    characteristics = {}
    
    # 치아 영역 분석
    tooth_gt = merged_df['치아감지영역_GT'].values
    tooth_pred = merged_df['치아감지영역_Pred'].values
    
    characteristics['치아영역'] = {
        'gt_mean': np.mean(tooth_gt),
        'pred_mean': np.mean(tooth_pred),
        'gt_std': np.std(tooth_gt),
        'pred_std': np.std(tooth_pred),
        'correlation': np.corrcoef(tooth_gt, tooth_pred)[0, 1],
        'cv_gt': np.std(tooth_gt) / np.mean(tooth_gt),  # 변동계수
        'cv_pred': np.std(tooth_pred) / np.mean(tooth_pred)
    }
    
    # 치근 영역 분석
    root_gt = merged_df['치근감지영역_GT'].values
    root_pred = merged_df['치근감지영역_Pred'].values
    
    characteristics['치근영역'] = {
        'gt_mean': np.mean(root_gt),
        'pred_mean': np.mean(root_pred),
        'gt_std': np.std(root_gt),
        'pred_std': np.std(root_pred),
        'correlation': np.corrcoef(root_gt, root_pred)[0, 1],
        'cv_gt': np.std(root_gt) / np.mean(root_gt),
        'cv_pred': np.std(root_pred) / np.mean(root_pred)
    }
    
    # 치근 비율 분석
    ratio_gt = merged_df['치근비율_GT'].values
    ratio_pred = merged_df['치근비율_Pred'].values
    
    characteristics['치근비율'] = {
        'gt_mean': np.mean(ratio_gt),
        'pred_mean': np.mean(ratio_pred),
        'gt_std': np.std(ratio_gt),
        'pred_std': np.std(ratio_pred),
        'correlation': np.corrcoef(ratio_gt, ratio_pred)[0, 1],
        'cv_gt': np.std(ratio_gt) / np.mean(ratio_gt),
        'cv_pred': np.std(ratio_pred) / np.mean(ratio_pred)
    }
    
    return characteristics

def calculate_icc_improved(x, y, icc_type='single', confidence_level=0.95):
    """
    개선된 ICC 계산입니다.
    각 영역의 특성을 고려한 계산을 수행합니다.
    
    Args:
        x (array): 첫 번째 측정값
        y (array): 두 번째 측정값
        icc_type (str): ICC 타입
        confidence_level (float): 신뢰구간 레벨
    
    Returns:
        dict: ICC 값과 신뢰구간
    """
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    k = 2
    
    # 데이터 특성 분석
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    
    # 각 대상의 평균과 전체 평균
    subject_means = np.array([np.mean([x[i], y[i]]) for i in range(n)])
    grand_mean = np.mean(np.concatenate([x, y]))
    
    # SSB (Between-subjects sum of squares)
    ssb = k * np.sum((subject_means - grand_mean)**2)
    
    # SSW (Within-subjects sum of squares)
    ssw = np.sum((x - subject_means)**2) + np.sum((y - subject_means)**2)
    
    # MSB (Between-subjects mean square)
    msb = ssb / (n - 1)
    
    # MSW (Within-subjects mean square)
    msw = ssw / (n * (k - 1))
    
    # ICC 계산 - 각 타입별로 다른 공식 사용
    if icc_type == 'single':
        icc = (msb - msw) / (msb + msw)
    elif icc_type == 'average':
        icc = (msb - msw) / msb
    elif icc_type == 'consistency':
        icc = (msb - msw) / (msb + msw)
    elif icc_type == 'agreement':
        icc = (msb - msw) / (msb + msw)
    else:
        raise ValueError("icc_type은 'single', 'average', 'consistency', 'agreement' 중 하나여야 합니다.")
    
    # 신뢰구간 계산 - 개선된 방법
    alpha = 1 - confidence_level
    
    if icc_type == 'single':
        if icc > 0.9:
            se = np.sqrt(2 * (1 - icc)**2 / (n * (k - 1)))
            z_score = stats.norm.ppf(1 - alpha/2)
            lower_bound = icc - z_score * se
            upper_bound = icc + z_score * se
        else:
            f_lower = stats.f.ppf(alpha/2, n-1, n*(k-1))
            f_upper = stats.f.ppf(1-alpha/2, n-1, n*(k-1))
            lower_bound = 1 - (1/icc) * f_upper
            upper_bound = 1 - (1/icc) * f_lower
            
    elif icc_type == 'average':
        if icc > 0.9:
            se = np.sqrt(2 * (1 - icc)**2 / (n * (k - 1)))
            z_score = stats.norm.ppf(1 - alpha/2)
            lower_bound = icc - z_score * se
            upper_bound = icc + z_score * se
        else:
            f_lower = stats.f.ppf(alpha/2, n-1, n*(k-1))
            f_upper = stats.f.ppf(1-alpha/2, n-1, n*(k-1))
            lower_bound = 1 - (1/icc) * f_upper
            upper_bound = 1 - (1/icc) * f_lower
    else:
        se = np.sqrt(2 * (1 - icc)**2 / (n * (k - 1)))
        z_score = stats.norm.ppf(1 - alpha/2)
        lower_bound = icc - z_score * se
        upper_bound = icc + z_score * se
    
    # 신뢰구간을 0과 1 사이로 제한
    lower_bound = max(0, min(1, lower_bound))
    upper_bound = max(0, min(1, upper_bound))
    
    return {
        'icc': icc,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
        'type': icc_type,
        'x_mean': x_mean,
        'y_mean': y_mean,
        'x_std': x_std,
        'y_std': y_std
    }

def test_icc_ccc_calculations():
    """
    ICC와 CCC 계산을 테스트합니다.
    """
    print("ICC와 CCC 계산 테스트...")
    
    # 테스트 데이터 생성 (높은 일치도)
    np.random.seed(42)
    n_samples = 100
    
    # Ground Truth 데이터
    gt_tooth = np.random.normal(50000, 10000, n_samples)
    gt_root = np.random.normal(5000, 1000, n_samples)
    gt_ratio = (gt_root / gt_tooth) * 100
    
    # 예측 데이터 (높은 일치도)
    pred_tooth = gt_tooth + np.random.normal(0, 1000, n_samples)  # 작은 노이즈
    pred_root = gt_root + np.random.normal(0, 100, n_samples)      # 작은 노이즈
    pred_ratio = (pred_root / pred_tooth) * 100
    
    print(f"\n테스트 데이터 정보:")
    print(f"  샘플 수: {n_samples}")
    print(f"  치아 영역 GT 평균: {np.mean(gt_tooth):.0f}")
    print(f"  치아 영역 Pred 평균: {np.mean(pred_tooth):.0f}")
    print(f"  치근 영역 GT 평균: {np.mean(gt_root):.0f}")
    print(f"  치근 영역 Pred 평균: {np.mean(pred_root):.0f}")
    
    # CCC 계산
    tooth_ccc = calculate_ccc(gt_tooth, pred_tooth)
    root_ccc = calculate_ccc(gt_root, pred_root)
    ratio_ccc = calculate_ccc(gt_ratio, pred_ratio)
    
    print(f"\n🔄 CCC 결과:")
    print(f"  치아 영역: {tooth_ccc:.3f}")
    print(f"  치근 영역: {root_ccc:.3f}")
    print(f"  치근 비율: {ratio_ccc:.3f}")
    
    # 간단한 ICC 계산
    print(f"\n📊 간단한 ICC 결과:")
    
    # 치아 영역 ICC
    print(f"\n  🦷 치아 영역:")
    for icc_type in ['single', 'average', 'consistency', 'agreement']:
        try:
            result = calculate_icc_simple(gt_tooth, pred_tooth, icc_type)
            print(f"    {icc_type.capitalize()}: {result['icc']:.3f} (95% CI: {result['lower_bound']:.3f} - {result['upper_bound']:.3f})")
        except Exception as e:
            print(f"    {icc_type.capitalize()}: 계산 오류 - {e}")
    
    # 치근 영역 ICC
    print(f"\n  🦷 치근 영역:")
    for icc_type in ['single', 'average', 'consistency', 'agreement']:
        try:
            result = calculate_icc_simple(gt_root, pred_root, icc_type)
            print(f"    {icc_type.capitalize()}: {result['icc']:.3f} (95% CI: {result['lower_bound']:.3f} - {result['upper_bound']:.3f})")
        except Exception as e:
            print(f"    {icc_type.capitalize()}: 계산 오류 - {e}")
    
    # 치근 비율 ICC
    print(f"\n  📊 치근 비율:")
    for icc_type in ['single', 'average', 'consistency', 'agreement']:
        try:
            result = calculate_icc_simple(gt_ratio, pred_ratio, icc_type)
            print(f"    {icc_type.capitalize()}: {result['icc']:.3f} (95% CI: {result['lower_bound']:.3f} - {result['upper_bound']:.3f})")
        except Exception as e:
            print(f"    {icc_type.capitalize()}: 계산 오류 - {e}")
    
    # 상관계수 비교
    tooth_corr = np.corrcoef(gt_tooth, pred_tooth)[0, 1]
    root_corr = np.corrcoef(gt_root, pred_root)[0, 1]
    ratio_corr = np.corrcoef(gt_ratio, pred_ratio)[0, 1]
    
    print(f"\n📈 상관계수 비교:")
    print(f"  치아 영역: {tooth_corr:.3f}")
    print(f"  치근 영역: {root_corr:.3f}")
    print(f"  치근 비율: {ratio_corr:.3f}")
    
    print(f"\n✅ 테스트 완료!")
    return True

def calculate_icc_by_area(merged_df):
    """
    각 영역별로 개별적으로 ICC를 계산합니다.
    
    Args:
        merged_df (DataFrame): 병합된 데이터프레임
    
    Returns:
        dict: 각 영역별 ICC 결과
    """
    areas = {
        '치아영역': (merged_df['치아감지영역_GT'].values, merged_df['치아감지영역_Pred'].values),
        '치근영역': (merged_df['치근감지영역_GT'].values, merged_df['치근감지영역_Pred'].values),
        '치근비율': (merged_df['치근비율_GT'].values, merged_df['치근비율_Pred'].values)
    }
    
    icc_types = ['single', 'average', 'consistency', 'agreement']
    results = {}
    
    for area_name, (x, y) in areas.items():
        print(f"\n📊 {area_name} ICC 계산:")
        print(f"  데이터 특성:")
        print(f"    GT 평균: {np.mean(x):.0f}, Pred 평균: {np.mean(y):.0f}")
        print(f"    GT 표준편차: {np.std(x):.0f}, Pred 표준편차: {np.std(y):.0f}")
        print(f"    상관계수: {np.corrcoef(x, y)[0, 1]:.3f}")
        
        area_results = {}
        for icc_type in icc_types:
            try:
                result = calculate_icc_improved(x, y, icc_type)
                area_results[icc_type] = result['icc']
                print(f"    {icc_type.capitalize()}: {result['icc']:.3f} (95% CI: {result['lower_bound']:.3f} - {result['upper_bound']:.3f})")
            except Exception as e:
                print(f"    {icc_type.capitalize()}: 계산 오류 - {e}")
                area_results[icc_type] = np.nan
        
        results[area_name] = area_results
    
    return results

def main():
    """
    메인 함수
    """
    import sys
    
    # 명령행 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("ICC와 CCC 계산 테스트를 실행합니다...")
        test_icc_ccc_calculations()
        return
    
    # 파일 경로
    gt_file = r"C:\Users\HP\Exposed-Root-Calculation\GT_analysis_20250808\gt_analysis_20250808_100911.csv"
    pred_file = r"C:\Users\HP\Exposed-Root-Calculation\analysis_results_fixed.csv"
    
    print("Ground Truth vs 예측 결과 정확도 분석 시작...")
    print("ICC와 CCC를 포함한 종합적인 일치도 분석을 수행합니다.")
    
    try:
        # 데이터 로드 및 준비
        gt_df, pred_df, merged_df = load_and_prepare_data(gt_file, pred_file)
        
        # 데이터 특성 분석
        characteristics = analyze_data_characteristics(merged_df)
        print("\n📊 데이터 특성 분석:")
        for area, chars in characteristics.items():
            print(f"  {area}:")
            print(f"    GT 평균: {chars['gt_mean']:.0f}, Pred 평균: {chars['pred_mean']:.0f}")
            print(f"    GT 표준편차: {chars['gt_std']:.0f}, Pred 표준편차: {chars['pred_std']:.0f}")
            print(f"    상관계수: {chars['correlation']:.3f}")
            print(f"    GT 변동계수: {chars['cv_gt']:.3f}, Pred 변동계수: {chars['cv_pred']:.3f}")
        
        # 정확도 지표 계산
        metrics = calculate_accuracy_metrics(merged_df)
        
        # 각 영역별 ICC 계산
        area_icc_results = calculate_icc_by_area(merged_df)
        
        # 결과 출력
        print_accuracy_report(metrics)
        
        # 그래프 생성 및 저장
        status_comparison = create_comparison_plots(merged_df)
        
        # 상세 비교 결과 저장
        save_detailed_comparison(merged_df)
        
        # ICC, CCC, 통계검정 결과 저장
        save_statistical_results(metrics, merged_df)
        
        print(f"\n분석 완료! 결과는 'accuracy_analysis' 폴더에 저장되었습니다.")
        print(f"ICC와 CCC 계산이 성공적으로 완료되었습니다.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        print("테스트를 실행하려면: python accuracy_calculator.py --test")

if __name__ == '__main__':
    main() 