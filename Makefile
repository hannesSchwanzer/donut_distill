CUDA_TARGET=6
PYTHON_TARGET=python

train:
	CUDA_VISIBLE_DEVICES=$(CUDA_TARGET) $(PYTHON_TARGET) train.py

preprocess:
	$(PYTHON_TARGET) preprocess_donut.py
