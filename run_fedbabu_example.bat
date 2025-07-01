@echo off
echo ==========================================
echo Executando FedBABU com Fine-tuning da Cabeca
echo ==========================================

echo.
echo 1. Apenas congelar cabeca (sem fine-tuning):
echo python "train_CWT _BABU.py" --freeze_head --dataset cifar10 --net_name ViT-tiny --FL_platform ViT-CWT
echo.

echo 2. Processo completo FedBABU (congelar + fine-tuning):
python "train_CWT _BABU.py" --freeze_head --finetune_head --finetune_epochs 10 --finetune_lr 1e-4 --dataset cifar10 --net_name ViT-tiny --FL_platform ViT-CWT --save_model_flag

echo.
echo 3. Com configuracao personalizada:
echo python "train_CWT _BABU.py" --freeze_head --finetune_head --finetune_epochs 15 --finetune_lr 5e-5 --dataset cifar10 --net_name Swin-tiny --FL_platform Swin-CWT --max_communication_rounds 50 --save_model_flag

echo.
echo Execucao concluida!
pause
