import os, sys, logging, argparse, yaml, easydict, json
import numpy as np
import torch

from transformers import (
    TrainingArguments,
    Trainer,
)
from transformers.trainer import Trainer
from peft import (
    LoraConfig,
    get_peft_model,
)
from accelerate import Accelerator
from torchdrug.utils import comm, pretty

from llm import *
from collector import *
from preprocess import *
from baseline_checkpoint import BaselineCheckpointCallback, save_baseline_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str,
                        default='config/fb15k237.yaml')
    parser.add_argument("--version", "-v", type=str,
                        default='')
    parser.add_argument("--seed", "-s", type=int,
                        default=42)
    parser.add_argument("--eval-split", choices=("valid", "test"), default="test")
    parser.add_argument("--checkpoint-steps", type=int, default=500,
                        help="Save compact frozen-model weights every N optimizer steps; 0 disables step saves.")
    parser.add_argument("--save-mkgl-checkpoint", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint_steps < 0:
        parser.error("--checkpoint-steps must be >= 0")
    
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
        if args.version:
            cfg.dataset.version = args.version
    torch.manual_seed(args.seed + comm.get_rank())

    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version'):
        config_name += '_' + cfg.dataset.version
    args.config_name = config_name
    cfg.trainer.output_dir += config_name
    
    
    if comm.get_rank() == 0:
        print("Config file: %s" % args.config)
        print(pretty.format(cfg))
    

    saved_dir = 'data/preprocessed/'
    file_path = saved_dir+args.config_name+'.pkl'
    if 'ind' in args.config_name:
        dataset = InductiveKGCDataset.load(file_path)
    else:
        dataset = KGCDataset.load(file_path)
    tokenizer = dataset.tokenizer
    cfg.context_retriever.kg_encoder.base_layer.num_relation = int(
        dataset.kgdata.num_relation)
    cfg.score_retriever.kg_encoder.base_layer.num_relation = int(
        dataset.kgdata.num_relation)
    
    torch.nn.Module = torch.nn._Module
    config = MKGLConfig.from_pretrained(**cfg.mkglconfig)
    model = MKGL.from_pretrained(
        **cfg.mkgl, device_map={"": Accelerator().process_index}, config=config)

    lora_config = LoraConfig(**cfg.loraconfig)
    model = get_peft_model(model, lora_config)

    kgl2token = torch.tensor(np.stack(dataset.vocab_df.text_token_ids)[:, :cfg.kgl_token_length])     
    model.init_kg_specs(kgl2token, tokenizer.vocab_size, cfg) 

    checkpoint_path = args.save_mkgl_checkpoint or os.path.join(
        cfg.trainer.output_dir, "mkgl_checkpoint.pt")
    checkpoint_metadata = {
        "config_name": config_name,
        "config": json.loads(json.dumps(cfg)),
        "seed": args.seed,
        "eval_split": args.eval_split,
        "base_model": cfg.mkgl.pretrained_model_name_or_path,
        "preprocessed_dataset": file_path,
        "orig_vocab_size": int(tokenizer.vocab_size),
        "kgl2token": kgl2token.cpu(),
        "kgl_token_indices": [int(i) for i in dataset.vocab_df.index],
        "kgl_raw_names": dataset.vocab_df.raw_name.tolist(),
        "relation_vocab": list(dataset.kgdata.relation_vocab),
    }
    for vocab_name in ("entity_vocab", "transductive_vocab", "inductive_vocab"):
        vocab = getattr(dataset.kgdata, vocab_name, None)
        if vocab is not None:
            checkpoint_metadata[vocab_name] = list(vocab)
    if comm.get_rank() == 0:
        tokenizer.save_pretrained(os.path.join(cfg.trainer.output_dir, "tokenizer"))
    checkpoint_callback = BaselineCheckpointCallback(
        model, checkpoint_path, checkpoint_metadata, args.checkpoint_steps)
    
    if comm.get_rank() == 0:
        print(model.print_trainable_parameters())
        print(model)

    
    if 'ind' in args.config:
        task = KGL4IndKGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    else:
        task = KGL4KGC(cfg.mkgl4kgc, llmodel=model, dataset=dataset)
    

    data_loader = MKGLDataCollector(dataset)
    
    training_args = TrainingArguments(**cfg.trainer)
    if comm.get_rank() == 0:
        print(training_args)


    def compute_metrics(predictions):
        ranking = predictions[0].astype(float)
        metric = ("mr", "mrr", "hits@1", "hits@3", "hits@10")
        results = {}
        for _metric in metric:
            if _metric == "mr":
                score = ranking.mean()
            elif _metric == "mrr":
                score = (1 / ranking).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            results[_metric] = score
        if comm.get_rank() == 0:
            print(results)
        return results

    removed_columns = ['h_raw', 't_raw', 'r_raw', 'h_fine', 't_fine', 'r_fine', 'inv_r_fine']

    trainer = Trainer(
        model=task,
        args=training_args,
        eval_dataset=(dataset.valid_data if args.eval_split == "valid" else dataset.test_data).remove_columns(
            removed_columns),  
        train_dataset=dataset.train_data.remove_columns(removed_columns),
        data_collator=data_loader,
        compute_metrics=compute_metrics,
        callbacks=[checkpoint_callback],
    )
    trainer.evaluate()
    trainer.train()
    if trainer.is_world_process_zero():
        save_baseline_checkpoint(
            model, checkpoint_path, checkpoint_metadata,
            step=trainer.state.global_step, epoch=trainer.state.epoch,
            training_completed=True,
        )
        trainer.save_state()

