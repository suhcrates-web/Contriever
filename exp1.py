# from src.options import Options
# options = Options()
# opt = options.parse()

# print(opt)
# """
# Namespace(output_dir='./checkpoint/my_experiments', train_data=[], eval_data=[], eval_datasets=[], eval_datasets_dir='./', model_path='none', continue_training=False, num_workers=5, chunk_length=256, loading_mode='split', lower_case=False, sampling_coefficient=0.0, augmentation='none', prob_augmentation=0.0, dropout=0.1, rho=0.05, contrastive_mode='moco', queue_size=65536, 
# temperature=1.0, momentum=0.999, moco_train_mode_encoder_k=False, eval_normalize_text=False, norm_query=False, norm_doc=False, projection_size=768, ratio_min=0.1, ratio_max=0.5, score_function='dot', retriever_model_id='bert-base-uncased', pooling='average', random_init=False, per_gpu_batch_size=64, per_gpu_eval_batch_size=256, total_steps=1000, warmup_steps=-1, local_rank=-1, main_port=10001, seed=0, optim='adamw', scheduler='linear', lr=0.0001, lr_min_ratio=0.0, weight_decay=0.01, beta1=0.9, beta2=0.98, eps=1e-06, log_freq=100, eval_freq=500, save_freq=50000, maxload=None, label_smoothing=0.0, negative_ctxs=1, negative_hard_min_idx=0, negative_hard_ratio=0.0)
# """
import glob
import os
data_path ='./src'


files = glob.glob(os.path.join(data_path, "*.*"))
print(files)