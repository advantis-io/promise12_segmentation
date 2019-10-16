ARG FROM_IMAGE=tensorflow/tensorflow:latest-gpu-py3

FROM $FROM_IMAGE

RUN apt-get update && \
    apt-get install -yq libsm6 libxext6 libxrender-dev

RUN pip install ipython

RUN pip install scikit-image scikit-learn opencv-python \
    SimpleITK tensorflow-gpu keras
