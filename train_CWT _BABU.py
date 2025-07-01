# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import numpy as np
import pandas as pd
from math import ceil

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.scheduler import setup_scheduler
from utils.data_utils import DatasetFLViT, create_dataset_and_evalmetrix
from utils.util import valid
from utils.start_config import initization_configure

def finetune_head_per_client(args, model, writer):
    """Fine-tune apenas a cabeça do modelo para cada cliente individualmente"""
    print("=============== Iniciando Fine-tuning da Cabeça por Cliente ===============")
    
    # Primeiro, descongelar apenas a cabeça
    for param in model.parameters():
        param.requires_grad = False
    
    # Descongelar apenas a cabeça/classifier
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
        print("Head do modelo descongelada para fine-tuning")
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("Classifier do modelo descongelado para fine-tuning")
    
    # Configurar otimizador apenas para parâmetros da cabeça
    head_params = [p for p in model.parameters() if p.requires_grad]
    head_optimizer = torch.optim.AdamW(head_params, lr=args.finetune_lr, weight_decay=0.01)
    
    # Configurar loss function
    if args.num_classes == 1:
        loss_fct = torch.nn.MSELoss()
    else:
        loss_fct = torch.nn.CrossEntropyLoss()
    
    # Fine-tuning para cada cliente
    client_results = {}
    
    for client_idx, single_client in enumerate(args.dis_cvs_files):
        print(f"\n--- Fine-tuning para Cliente {single_client} ---")
        args.single_client = single_client
        
        # Preparar dados do cliente
        trainset = DatasetFLViT(args, phase='train')
        train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), 
                                batch_size=args.batch_size, num_workers=args.num_workers)
        
        if args.dataset == 'CelebA':
            valset = DatasetFLViT(args, phase='val')
            val_loader = DataLoader(valset, sampler=SequentialSampler(valset), 
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            valset = DatasetFLViT(args, phase='val')
            val_loader = DataLoader(valset, sampler=SequentialSampler(valset), 
                                  batch_size=args.batch_size, num_workers=8)
        
        testset = DatasetFLViT(args, phase='test')
        test_loader = DataLoader(testset, sampler=SequentialSampler(testset), 
                               batch_size=args.batch_size, num_workers=8)
        
        # Salvar estado inicial da cabeça (opcional - para comparação)
        initial_head_state = {}
        if hasattr(model, 'head'):
            initial_head_state = model.head.state_dict().copy()
        elif hasattr(model, 'classifier'):
            initial_head_state = model.classifier.state_dict().copy()
        
        model.train()
        best_acc = 0.0
        best_head_state = None
        
        # Fine-tuning epochs
        for ft_epoch in range(args.finetune_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_loader):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                if args.num_classes == 1:
                    y = y.float()
                
                head_optimizer.zero_grad()
                predict = model(x)
                loss = loss_fct(predict.view(-1, args.num_classes), y.view(-1))
                
                loss.backward()
                head_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Log a cada 10 steps
                if (step + 1) % 10 == 0:
                    print(f"Cliente {single_client} FT Epoch {ft_epoch}, Step {step+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / num_batches
            
            # Validação após cada época de fine-tuning
            model.eval()
            with torch.no_grad():
                val_acc = valid(args, model, val_loader, test_loader, TestFlag=False)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    if hasattr(model, 'head'):
                        best_head_state = model.head.state_dict().copy()
                    elif hasattr(model, 'classifier'):
                        best_head_state = model.classifier.state_dict().copy()
            
            model.train()
            
            # Logging
            writer.add_scalar(f"finetune/{single_client}/loss", avg_loss, ft_epoch)
            writer.add_scalar(f"finetune/{single_client}/val_acc", val_acc, ft_epoch)
            
            print(f"Cliente {single_client} FT Epoch {ft_epoch}: Loss {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Carregar melhor estado da cabeça
        if best_head_state is not None:
            if hasattr(model, 'head'):
                model.head.load_state_dict(best_head_state)
            elif hasattr(model, 'classifier'):
                model.classifier.load_state_dict(best_head_state)
        
        # Avaliação final do cliente
        model.eval()
        with torch.no_grad():
            final_acc = valid(args, model, val_loader, test_loader, TestFlag=True)
        
        client_results[single_client] = {
            'best_val_acc': best_acc,
            'final_test_acc': final_acc
        }
        
        # Salvar modelo específico do cliente (opcional)
        if args.save_model_flag:
            client_model_path = os.path.join(args.output_dir, f'client_{single_client}_finetuned_head.pth')
            if hasattr(model, 'head'):
                torch.save(model.head.state_dict(), client_model_path)
            elif hasattr(model, 'classifier'):
                torch.save(model.classifier.state_dict(), client_model_path)
            print(f"Cabeça fine-tuned do cliente {single_client} salva em {client_model_path}")
        
        print(f"Cliente {single_client} - Melhor Val Acc: {best_acc:.4f}, Test Acc Final: {final_acc:.4f}")
    
    # Salvar resultados finais
    results_df = pd.DataFrame.from_dict(client_results, orient='index')
    results_df.to_csv(os.path.join(args.output_dir, 'finetune_head_results.csv'))
    
    print("\n=============== Fine-tuning da Cabeça Concluído ===============")
    print("Resultados por cliente:")
    for client, results in client_results.items():
        print(f"Cliente {client}: Val Acc {results['best_val_acc']:.4f}, Test Acc {results['final_test_acc']:.4f}")
    
    return client_results




def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Prepare dataset
    create_dataset_and_evalmetrix(args)

    # FASE 1: Congelar cabeça se especificado
    if args.freeze_head:
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = False
            print("Head do modelo Swin congelada para treinamento CWT_BABU")
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = False
            print("Classifier do modelo ViT congelado para treinamento CWT_BABU")
        
        # Verificar quais parâmetros estão treináveis
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parâmetros treináveis: {trainable_params:,} / {total_params:,}")

    testset = DatasetFLViT(args, phase = 'test' )
    test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.batch_size, num_workers=8)

    # if not CelebA then get the union val dataset,
    if not args.dataset in ['CelebA']:
        valset = DatasetFLViT(args, phase = 'val' )
        val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=8)

    # Prepare optimizer, scheduler
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

        print("===============Not implemented optimization type, we used default adamw optimizer ===============")

    tot_step_per_round = [ceil(value / args.batch_size) for value in args.clients_with_len.values()]
    args.t_total = sum(tot_step_per_round) * args.max_communication_rounds * args.E_epoch

    scheduler = setup_scheduler(args, optimizer, t_total=args.t_total)
    if args.num_classes == 1:
        # loss_fct = torch.nn.L1Loss()
        loss_fct = torch.nn.MSELoss()
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    # print('For debugging usage, t_total', args.t_total)

    # Train!
    print("=============== Running training ===============")

    model.zero_grad()
    args.global_step = 0

    for epoch in range(args.max_communication_rounds):

        model.train()
        if args.decay_type == 'step':
            scheduler.step()

        ## iterative each client


        for single_client in args.dis_cvs_files:
            print('Train the client', single_client, 'of communication round', epoch)
            args.single_client = single_client

            trainset = DatasetFLViT(args, phase='train')
            train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.batch_size, num_workers=args.num_workers)
            if args.dataset == 'CelebA':
                valset = DatasetFLViT(args, phase='val')
                val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=args.num_workers)

            for inner_epoch in range(args.E_epoch):
                for step, batch in enumerate(train_loader):
                    args.global_step += 1
                    batch = tuple(t.to(args.device) for t in batch)
                    x, y = batch
                    if args.num_classes == 1:
                        y = y.float()

                    predict = model(x)
                    loss = loss_fct(predict.view(-1, args.num_classes), y.view(-1))

                    loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if not args.decay_type == 'step':
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()

                    writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=args.global_step)
                    writer.add_scalar("train/lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=args.global_step)
                    args.learning_rate_record.append(optimizer.param_groups[0]['lr'])

                    if (step +1) % 10 == 0:
                        message = "Client: %s inner epoch: %d step: %d (%d), round: %d (%d) loss: %2.2f lr:  %f" % ( single_client,
                            inner_epoch, step, len(train_loader), epoch,    args.max_communication_rounds,   loss.item()    , optimizer.param_groups[0]['lr']
                        )
                        print(message)

                valid(args, model, val_loader, test_loader, TestFlag=True)
                model.train()

        np.save(args.output_dir + '/learning_rate.npy', args.learning_rate_record)
        args.record_val_acc = args.record_val_acc.append(args.current_acc, ignore_index=True)
        args.record_val_acc.to_csv(os.path.join(args.output_dir, 'val_acc.csv'))
        args.record_test_acc = args.record_test_acc.append(args.current_test_acc, ignore_index=True)
        args.record_test_acc.to_csv(os.path.join(args.output_dir, 'test_acc.csv'))

    writer.close()
    print("================End training! ================ ")

    # FASE 2: Fine-tuning da cabeça após treinamento principal
    if args.finetune_head:
        print("================Iniciando Fine-tuning da Cabeça ================ ")
        # Reabrir writer para fine-tuning
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "finetune_logs"))
        
        # Realizar fine-tuning da cabeça por cliente
        finetune_results = finetune_head_per_client(args, model, writer)
        
        writer.close()
        
        return finetune_results
    
    return None





def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type = str, default="Swin-tiny",  help="Basic Name of this run with detailed network-architecture selection")
    parser.add_argument("--FL_platform", type = str, default="Swin-CWT", choices=["Swin-CWT", 'ViT-CWT',"ResNet-CWT", "Swin-CWT", "EfficientNet-CWT"],  help="Choose of different FL platform.")
    parser.add_argument("--dataset", choices=["cifar10", "Retina","CelebA"], default="cifar10", help="Which dataset.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Where is dataset located.")

    parser.add_argument("--save_model_flag",  action='store_true', default=False,  help="Save the best model for each client.")
    parser.add_argument("--cfg",  type=str, default="configs/swin_tiny_patch4_window7_224.yaml", metavar="FILE", help='path to args file for Swin-FL',)

    # Parâmetros para FedBABU
    parser.add_argument('--freeze_head', action='store_true', default=False, 
                       help="Congelar apenas a cabeça do modelo para CWT_BABU")
    parser.add_argument('--finetune_head', action='store_true', default=False,
                       help="Realizar fine-tuning da cabeça após treinamento")
    parser.add_argument('--finetune_epochs', type=int, default=5,
                       help="Número de épocas para fine-tuning da cabeça")
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                       help="Learning rate para fine-tuning da cabeça")

    parser.add_argument('--Pretrained', action='store_true', default=True, help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/swin_tiny_patch4_window7_224.pth", help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="whether gradient clip to 1 or not")

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=32, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default= 3e-3, type=float, help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")

    ## FL related parameters
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int, help="Total communication rounds.")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "real" ,"central"], default="split_2", help="Which data partitions to use")


    args = parser.parse_args()

    # Initialization

    model = initization_configure(args)

    # Training, Validating, and Testing
    finetune_results = train(args, model)

    # Show final performance
    message = '\n \n ==============Start showing final performance ================= \n'
    message += 'Final union test accuracy is: %2.5f with std: %2.5f \n' %  \
                   (np.asarray(list(args.current_test_acc.values())).mean(),  np.asarray(list(args.current_test_acc.values())).std())
    
    # Adicionar resultados do fine-tuning se disponível
    if finetune_results is not None:
        finetune_test_accs = [results['final_test_acc'] for results in finetune_results.values()]
        message += f'Fine-tuned test accuracy - Mean: {np.mean(finetune_test_accs):.5f}, Std: {np.std(finetune_test_accs):.5f} \n'
    
    message += "================ End ================ \n"

    with open(args.file_name, 'a+') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)

if __name__ == "__main__":
    main()
