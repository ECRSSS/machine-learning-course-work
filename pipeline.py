import dill
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearnex import patch_sklearn

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from xgboost import XGBClassifier

from datetime import datetime
import re

import pandas as pd

import warnings

patch_sklearn()
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TARGET_EVENTS = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                 'sub_open_dialog_click', 'sub_custom_question_submit_click',
                 'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                 'sub_car_request_submit_click']
HOUR_PATTERN = re.compile('(\d\d):\d\d:\d\d')

DATE_FORMAT = '%Y-%m-%d'


def extract_width(res_str):
    wharr = res_str.split('x')
    if len(wharr) == 2:
        return wharr[0]
    else:
        return None


def extract_height(res_str):
    wharr = res_str.split('x')
    if len(wharr) == 2:
        return wharr[1]
    else:
        return None


# объединяем датафреймы
def get_merged_data():  # -> return merged dataframe and list of hits columns
    sessions_df = pd.read_csv('data/ga_sessions.csv')
    hits_df = pd.read_csv('data/ga_hits.csv')
    merged_df = sessions_df.merge(hits_df, how='inner', on='session_id')
    return merged_df


def make_target_column(merged_df):
    merged_df['event_value'] = merged_df.apply(lambda row: row['event_action'] in TARGET_EVENTS, axis=1)
    merged_df['event_value'] = merged_df['event_value'].astype('bool')
    merged_df['event_value'] = merged_df['event_value'].replace({True: 1, False: 0})
    return merged_df


def balance_data(merged_df):
    hits_df_columns = ['session_id', 'hit_date', 'hit_time', 'hit_number', 'hit_type', 'hit_referer', 'hit_page_path',
                       'event_category', 'event_action', 'event_label']

    def __get_part_by_action(event_action):
        part = merged_df[merged_df['event_action'] == event_action]

        if action in TARGET_EVENTS:
            print(f'{action} (True) = {len(part)} rows')
            return part
        else:
            print(f'{action} (False) = 1200 rows')
            return part.head(1200)

    sample = merged_df.head(1)
    list_actions = merged_df['event_action'].unique().tolist()

    for action in list_actions:
        sample = pd.concat([sample, __get_part_by_action(action)], axis=0)

    sample = sample.drop(hits_df_columns, axis=1)
    sample = sample.drop('client_id', axis=1)
    return sample.drop_duplicates()


def clear_utm_columns(merged_df):
    print('start clear_utm_columns')
    merged_df.loc[merged_df['utm_source'].isna(), 'utm_source'] = 'unknown'
    merged_df.loc[merged_df['utm_campaign'].isna(), 'utm_campaign'] = 'unknown'
    merged_df.loc[merged_df['utm_adcontent'].isna(), 'utm_adcontent'] = 'unknown'
    merged_df = merged_df.drop(['utm_keyword'], axis=1)
    merged_df.loc[merged_df['utm_medium'].str.contains('\(no'), 'utm_medium'] = 'none'
    return merged_df


def clear_brand_and_os(merged_df):
    print('start clear_brand_and_os')
    merged_df.loc[merged_df['device_brand'].isna(), 'device_brand'] = 'unknown'
    merged_df = merged_df.drop(['device_model'], axis=1)
    merged_df = merged_df.drop(['device_os'], axis=1)
    return merged_df


def clear_datetime_columns(merged_df):
    print('start clear_datetime_columns')

    def __extract_hour(timestr):
        groups = HOUR_PATTERN.findall(timestr)
        if len(groups) > 0:
            hour = int(groups[0])
        else:
            hour = None
        return hour

    def __extract_year(date_str):
        date = datetime.strptime(date_str, DATE_FORMAT)
        return date.year

    def __extract_month(date_str):
        date = datetime.strptime(date_str, DATE_FORMAT)
        return date.month

    def __extract_day(date_str):
        date = datetime.strptime(date_str, DATE_FORMAT)
        return date.day

    def __extract_day_of_week(date_str):
        date = datetime.strptime(date_str, DATE_FORMAT)
        return date.weekday()

    merged_df['visit_hour'] = merged_df.apply(lambda row: __extract_hour(row['visit_time']), axis=1)
    merged_df['visit_year'] = merged_df.apply(lambda row: __extract_year(row['visit_date']), axis=1)
    merged_df['visit_month'] = merged_df.apply(lambda row: __extract_month(row['visit_date']), axis=1)
    merged_df['visit_day'] = merged_df.apply(lambda row: __extract_day(row['visit_date']), axis=1)
    merged_df['visit_day_of_week'] = merged_df.apply(lambda row: __extract_day_of_week(row['visit_date']), axis=1)

    # удаление необработанных столбцов
    merged_df = merged_df.drop(['visit_date', 'visit_time'], axis=1)
    # удаление года тк для всех значений он одинаков
    merged_df = merged_df.drop(['visit_year'], axis=1)

    return merged_df


def clear_geo(merged_df):
    print('start clear_geo')
    # geo
    merged_df['place'] = merged_df.apply(
        lambda row: row['geo_country'].lower().replace(' ', '_') + '_' + row['geo_city'].lower().replace(' ', '_'),
        axis=1)
    merged_df = merged_df.drop(['geo_country', 'geo_city'], axis=1)
    return merged_df


def clear_resolution(merged_df):
    print('start clear_resolution')
    merged_df['screen_width'] = merged_df.apply(lambda row: extract_width(row['device_screen_resolution']), axis=1)
    merged_df['screen_height'] = merged_df.apply(lambda row: extract_height(row['device_screen_resolution']), axis=1)
    merged_df = merged_df.drop(['device_screen_resolution'], axis=1)
    merged_df['screen_width'] = pd.to_numeric(merged_df['screen_width'], downcast='integer')
    merged_df['screen_height'] = pd.to_numeric(merged_df['screen_height'], downcast='integer')

    return merged_df


def obj_columns_to_categories(merged_df):
    print('start obj_columns_to_categories')
    category_columns = list(merged_df.select_dtypes(include=['object']).columns)
    for col in category_columns:
        merged_df[col] = merged_df[col].astype("category")
    return merged_df


def main():
    print('Sber Autopodpiska marketing target event prediction pipeline')
    merged_df = get_merged_data()
    merged_df = make_target_column(merged_df)
    merged_df = balance_data(merged_df)

    # merged_df = pd.read_csv('prepared_data_samples/balanced_10_06_2023-20_17_48.csv')
    # merged_df.drop(columns=merged_df.columns[0], axis=1, inplace=True)

    # merged_df.to_csv(f'prepared_data_samples/balanced_{datetime.now().strftime("%m_%d_%Y-%H_%M_%S")}.csv')

    y = merged_df['event_value']
    X = merged_df.drop(['event_value'], axis=1)

    data_clear = Pipeline(steps=[
        ('clear_utm_columns', FunctionTransformer(clear_utm_columns)),
        ('clear_brand_and_os', FunctionTransformer(clear_brand_and_os)),
        ('clear_datetime_columns', FunctionTransformer(clear_datetime_columns)),
        ('clear_geo', FunctionTransformer(clear_geo)),
        ('clear_resolution', FunctionTransformer(clear_resolution)),
        ('obj_columns_to_categories', FunctionTransformer(obj_columns_to_categories))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    column_transform = ColumnTransformer(transformers=[
        ('categorical_transformation', categorical_transform, make_column_selector(dtype_include=[object, 'category'])),
        ('numerical_transformation', numerical_transformer,
         make_column_selector(dtype_include=['int16', 'int64', 'float64']))
    ])

    xgb_params = {'objective': 'binary:logistic', 'base_score': None, 'booster': 'gbtree', 'callbacks': None,
                  'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': 0.17, 'device': None,
                  'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None,
                  'feature_types': None, 'gamma': 0.05, 'grow_policy': None, 'importance_type': None,
                  'interaction_constraints': None, 'learning_rate': 0.35000000000000003, 'max_bin': None,
                  'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 7,
                  'max_leaves': None, 'min_child_weight': 3.0, 'monotone_constraints': None, 'multi_strategy': None,
                  'n_estimators': 70, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': None,
                  'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None,
                  'subsample': 0.9500000000000001, 'tree_method': None, 'validate_parameters': None, 'verbosity': None}

    model = XGBClassifier(**xgb_params)

    pipe = Pipeline(steps=[
        ('data_clean', data_clear),
        ('column_transform', column_transform),
        ('classifier', model)
    ])

    score = cross_val_score(pipe, X, y, cv=20, scoring='roc_auc', n_jobs=-1)

    print(f'model: {type(model).__name__}, ROC_AUC: {score.mean():.4f}')

    pipe.fit(X, y)

    with open('model/pipe.pickle', mode='wb') as f:
        dill.dump({
            'model': pipe,
            'metadata': {
                'author': 'Nick Glebanov',
                'version': 1,
                'date': datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                'accuracy': score
            }
        }, f)


if __name__ == '__main__':
    main()
