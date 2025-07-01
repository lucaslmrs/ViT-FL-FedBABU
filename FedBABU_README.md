# FedBABU Implementation - Vision Transformers

## Overview
Esta implementação adiciona o método FedBABU (Federated Learning with Body Aggregation and Body Updates) ao projeto ViT-FL, permitindo treinar o corpo do modelo de forma federada e depois fazer fine-tuning personalizado da cabeça para cada cliente.

## Fluxo de Execução

### FASE 1: Treinamento Federado do Corpo
```
┌─────────────────────────────────────────────┐
│  Modelo ViT/Swin                           │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │     CORPO       │  │     CABEÇA      │  │
│  │   (Encoder)     │  │ (Classifier)    │  │
│  │   ✅ Treinável   │  │  ❌ Congelada   │  │
│  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────┘
```

### FASE 2: Fine-tuning Individual da Cabeça
```
┌─────────────────────────────────────────────┐
│  Modelo Treinado                           │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │     CORPO       │  │     CABEÇA      │  │
│  │   (Encoder)     │  │ (Classifier)    │  │
│  │  ❌ Congelado   │  │   ✅ Treinável   │  │
│  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────┘
```

## Novos Parâmetros

### Parâmetros FedBABU
- `--freeze_head`: Congela a cabeça durante o treinamento federado
- `--finetune_head`: Realiza fine-tuning da cabeça após o treinamento principal
- `--finetune_epochs`: Número de épocas para fine-tuning (padrão: 5)
- `--finetune_lr`: Learning rate para fine-tuning (padrão: 1e-4)

## Comandos de Execução

### 1. Apenas Congelar Cabeça
```bash
python "train_CWT _BABU.py" --freeze_head --dataset cifar10
```

### 2. FedBABU Completo
```bash
python "train_CWT _BABU.py" --freeze_head --finetune_head --finetune_epochs 10 --finetune_lr 1e-4 --save_model_flag
```

### 3. Configuração Personalizada
```bash
python "train_CWT _BABU.py" \
    --freeze_head \
    --finetune_head \
    --finetune_epochs 15 \
    --finetune_lr 5e-5 \
    --dataset cifar10 \
    --net_name ViT-tiny \
    --FL_platform ViT-CWT \
    --max_communication_rounds 50 \
    --save_model_flag
```

## Arquivos Gerados

Após a execução, os seguintes arquivos são criados no diretório de saída:

```
output_dir/
├── logs/                           # Logs da Fase 1 (treinamento federado)
├── finetune_logs/                  # Logs da Fase 2 (fine-tuning)
├── finetune_head_results.csv       # Resultados consolidados por cliente
├── client_0_finetuned_head.pth     # Cabeça personalizada do Cliente 0
├── client_1_finetuned_head.pth     # Cabeça personalizada do Cliente 1
└── ...
```

## Funcionalidades Implementadas

### 1. Congelamento Inteligente
- Detecta automaticamente se o modelo tem `head` (Swin) ou `classifier` (ViT)
- Congela apenas a camada de classificação, mantendo o encoder treinável

### 2. Fine-tuning Personalizado
- Fine-tuning individual para cada cliente após o treinamento federado
- Validação e early stopping durante o fine-tuning
- Salvamento automático do melhor estado da cabeça

### 3. Logging Detalhado
- Logs separados para cada fase do treinamento
- Métricas específicas por cliente durante o fine-tuning
- Resultados consolidados em CSV

### 4. Compatibilidade
- Funciona com modelos ViT e Swin Transformer
- Compatível com datasets CIFAR-10, CelebA e Retina
- Suporta diferentes tipos de otimizadores

## Benefícios do FedBABU

1. **Personalização**: Cada cliente recebe uma cabeça especializada
2. **Privacidade**: Apenas o corpo do modelo é compartilhado
3. **Eficiência**: Reduz a comunicação ao compartilhar apenas parte do modelo
4. **Performance**: Melhora a accuracy individual de cada cliente

## Exemplo de Uso Completo

```bash
# 1. Executar treinamento FedBABU completo
python "train_CWT _BABU.py" \
    --freeze_head \
    --finetune_head \
    --finetune_epochs 10 \
    --finetune_lr 1e-4 \
    --dataset cifar10 \
    --net_name ViT-tiny \
    --FL_platform ViT-CWT \
    --max_communication_rounds 100 \
    --E_epoch 1 \
    --batch_size 32 \
    --learning_rate 3e-3 \
    --save_model_flag

# 2. Os resultados serão salvos automaticamente em:
# - output/finetune_head_results.csv
# - output/client_*_finetuned_head.pth
```

## Monitoramento

Use o TensorBoard para monitorar o treinamento:

```bash
# Para logs do treinamento federado
tensorboard --logdir output/logs

# Para logs do fine-tuning
tensorboard --logdir output/finetune_logs
```

## Troubleshooting

### Erro de Memória
- Reduza `--batch_size`
- Reduza `--finetune_epochs`

### Baixa Performance
- Aumente `--finetune_lr`
- Aumente `--finetune_epochs`
- Verifique se `--freeze_head` está ativado

### Problemas de Compatibilidade
- Certifique-se de que o modelo tem `head` ou `classifier`
- Verifique se os datasets estão no formato correto
