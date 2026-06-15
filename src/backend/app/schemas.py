from pydantic import BaseModel

class PredictionRequest(BaseModel):

    EXT_SOURCE_3: float
    EXT_SOURCE_2: float
    DAYS_BIRTH: float
    AMT_CREDIT: float
    DAYS_EMPLOYED: float
    AMT_ANNUITY: float
    AMT_INCOME_TOTAL: float
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: float
    DAYS_LAST_PHONE_CHANGE: float