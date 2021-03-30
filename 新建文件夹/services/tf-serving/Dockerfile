FROM tensorflow/serving

COPY ./saved_models/aki/ /models/aki/
COPY ./saved_models/sepsis/ /models/sepsis/
COPY ./saved_models/mi/ /models/mi/
COPY ./saved_models/vancomycin/ /models/vancomycin/

COPY ./model.config /models/