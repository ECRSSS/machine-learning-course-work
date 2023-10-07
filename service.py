import json

import dill
import pandas
import re
from datetime import datetime
import __main__

TARGET_EVENTS = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                 'sub_open_dialog_click', 'sub_custom_question_submit_click',
                 'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                 'sub_car_request_submit_click']

__main__.pandas = pandas
__main__.pd = pandas
__main__.re = re
__main__.datetime = datetime
__main__.TARGET_EVENTS = TARGET_EVENTS

from fastapi import FastAPI
from pydantic import BaseModel
import requests


class SessionForm(BaseModel):
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    event_value: int


app = FastAPI()
with open('model/pipe.pickle', mode='rb') as f:
    model = dill.load(f)


@app.get('/healthcheck')
def status():
    return "OK"


@app.get('/version')
def version():
    metadata = model['metadata']
    metadata['date'] = str(metadata['date'])
    metadata['type'] = str(metadata['type'])
    content = json.dumps(metadata)
    return content


@app.post('/predict', response_model=Prediction)
def predict(form: SessionForm):
    modeldump = form.dict()
    modeldump['id'] = 0
    dataframe = pandas.DataFrame(modeldump, index=['id'])
    y = model['model'].predict(dataframe)
    print(y)
    print(f'PREDICT: {y[0]}')
    return {
        'event_value': y[0]
    }


def test_predict():
    df = pandas.read_csv('prepared_data_samples/balanced_10_06_2023-20_17_48.csv')
    positive = df[df['event_value'] == 1].head(100)
    negative = df[df['event_value'] == 0].head(100)

    positive_x_dicts = positive.drop(columns=positive.columns[0], axis=1).drop(columns=['event_value'], axis=1).to_dict(
        orient='records')
    negative_x_dicts = negative.drop(columns=negative.columns[0], axis=1).drop(columns=['event_value'], axis=1).to_dict(
        orient='records')

    errors_when_request = 0

    positive_correct_answers = 0
    for data in positive_x_dicts:
        data['device_model'] = 'stub'
        data['device_os'] = 'stub'
        resp = requests.post(url='http://127.0.0.1:8000/predict', data=json.dumps(data))
        if resp.status_code == 200:
            if Prediction.model_validate_json(resp.text).event_value == 1:
                positive_correct_answers += 1
        else:
            errors_when_request += 1

    negative_correct_answers = 0
    for data in negative_x_dicts:
        data['device_model'] = 'stub'
        data['device_os'] = 'stub'
        resp = requests.post(url='http://127.0.0.1:8000/predict', data=json.dumps(data))
        if resp.status_code == 200:
            if Prediction.model_validate_json(resp.text).event_value == 0:
                negative_correct_answers += 1
        else:
            errors_when_request += 1

    print(f'demo positive requests: верно {positive_correct_answers}/100\n')
    print(f'demo negative requests: верно {negative_correct_answers}/100\n')
    print(f'errors: {errors_when_request}\n')
    assert errors_when_request == 0
