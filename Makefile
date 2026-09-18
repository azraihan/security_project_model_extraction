# Convenience targets. Override vars, e.g.  make victim ARCH=resnet18 EPOCHS=60
ARCH ?= resnet18
EPOCHS ?= 60
SUB_ARCH ?= smallcnn
VICTIM ?= checkpoints/victim.pt

.PHONY: install smoke victim main defense all clean

install:
	pip install -r requirements.txt

smoke:
	python run_sweep.py --experiment main --smoke
	python run_sweep.py --experiment defense --smoke

victim:
	python victim_train.py --arch $(ARCH) --epochs $(EPOCHS) --out $(VICTIM)

main:
	python run_sweep.py --experiment main --victim $(VICTIM) --sub-arch $(SUB_ARCH)

defense:
	python run_sweep.py --experiment defense --victim $(VICTIM) --sub-arch $(SUB_ARCH) --transfer cifar100

all: victim main defense

clean:
	rm -rf cache results checkpoints __pycache__
