#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python main_cls.py data=modelnet40

#CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python main_cls.py data=scanobjectnn
