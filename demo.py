import tempfile

from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

# weather_infomation_generation 为示例数据集，用户也可以使用自己的数据集进行训练
dataset_dict = MsDataset.load('weather_infomation_generation', namespace='zhaozh')

# 训练数据的输入出均为文本，需要将数据集预处理为输入为 src_txt，输出为 tgt_txt 的格式：
train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})

# 用户自己数据集构造
# train_dataset_dict = {"src_txt": ["text1", "text2"], "tgt_txt": ["text1", "text2"]}
# eval_dataset_dict = {"src_txt": ["text1", "text2"], "tgt_txt": ["text1", "text2"]}
# train_dataset = MsDataset(Dataset.from_dict(train_dataset_dict))
# eval_dataset = MsDataset(Dataset.from_dict(eval_dataset_dict))

num_warmup_steps = 500
def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

# 可以在代码修改 configuration 的配置
def cfg_modify_fn(cfg):
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {
        "type": "AdamW",
        "lr": 1e-3,
        "options": {}
    }
    cfg.train.max_epochs = 20
    cfg.train.dataloader = {
        "batch_size_per_gpu": 8,
        "workers_per_gpu": 1
    }
    return cfg

kwargs = dict(
    model='damo/nlp_palm2.0_pretrained_chinese-base',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir='/media/geoai/Data/nlp',
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(
    name=Trainers.text_generation_trainer, default_args=kwargs)
trainer.train()
