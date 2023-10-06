import json

import dill
import pandas
import __main__

__main__.pandas = pandas

from fastapi import FastAPI
from pydantic import BaseModel


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
    id: float
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
    modeldump = form.model_dump()
    print(modeldump)
    dataframe = pandas.DataFrame(modeldump, index=['id'])
    y = model['model'].predict(dataframe)
    print(y)
    print(f'PREDICT: {y[0]}')
    return {
        'id': form.id,
        'event_value': y[0]
    }
