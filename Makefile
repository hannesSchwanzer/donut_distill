CUDA_TARGET=6
PYTHON_TARGET=python

train:
	CUDA_VISIBLE_DEVICES=$(CUDA_TARGET) $(PYTHON_TARGET) donut_distill/train_teacher.py

preprocess:
	$(PYTHON_TARGET) donut_distill/preprocess_donut.py

watch:
	watch nvidia-smi
