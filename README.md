# STCInterLLM: Causal Intervention is Large Language Models Need for Large-scale Spatio-temporal Forecasting

Shijie Li, He Li, Xiaojing Li, Yong Xu,  Zhenhong Lin, Huaiguang Jiang*  (*Correspondence)<br />

[School of Future Technology](https://www2.scut.edu.cn/ft/main.htm), @[South China University of Technology](https://www.scut.edu.cn/en/)

-----

## Abstract

<p style="text-align: justify">
Spatio-temporal forecasting plays a crucial role in the dynamic perception of smart cities, such as traffic flow prediction, renewable energy forecasting, and load prediction. It aims to understand the trends of spatio-temporal changes under the interaction of various factors. Sufficient high-quality data and powerful models form the foundation for accurate spatio-temporal forecasting. Unfortunately, in reality, data is often sparse. In such cases, while adaptive graphs and Large Language Models (LLMs) can maintain performance, they face issues of spatial spurious associations and hallucinations, respectively. These issues further affect the ability of the model to learn and infer cross spatio-temporal and cross-scale features. To address this, we propose a novel model named Spatio-Temporal Causal Intervention Large Language Model (STCInterLLM). This model employs a newly designed causal intervention encoder to update spatial spurious correlations in the spatio-temporal adaptive graph. Subsequently, the novel Chain-of-Action instruction text is utilized to enforce the decomposition of the prediction process, thereby enhancing the causal representation of features while mitigating hallucinations in LLMs. Finally, a lightweight marker alignment module ensures the consistency of the encoder, instruction text, and LLM, enabling accurate forecasting of distinct scale spatio-temporal evolution patterns. Extensive experiments are conducted on power distribution systems and transportation systems, and the proposed STCInterLLM consistently achieves state-of-the-art performance across significantly different scenarios.
</p>

![image](https://github.com/lishijie15/STInterLLM/blob/main/pictures/Algorithm.png)

## Introduction of Our Model

* To better adapt to various scenarios lacking high-quality data, we propose the STInterLLM, which integrates a novel encoder and instruction-tuning to achieve efficient and accurate large-scale spatio-temporal prediction while minimizing double hallucinations.
* To precisely eliminate spurious spatio-temporal associations caused by adaptive graphs, we specifically design the lightweight causal inference and intervention mechanism named causal intervention encoder. This is integrated with multi-scale temporal feature extraction modules to help the encoder enable the LLM to understand and learn complex spatio-temporal evolution patterns.
* To effectively mitigate hallucinations in the LLM, we construct a novel Chains-of-Action instruction-tuning and alignment module. By enforcing step-by-step decomposition and examples of predictive actions, ensuring that the LLM aligns with and comprehends the complex information provided by the proposed encoder.
* To fully evaluate the performance of the model, we conduct extensive experiments based on the Power Distribution System(PDS) (Scenario 1) and NYC-taxi (Scenario 2) datasets. The proposed STInterLLM achieves SOTA performance across various spatio-temporal scenarios.

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:

* <a href='#Environment'>1. Environment </a>
* <a href='#Training STCInterLLM'>2. Training STCInterLLM </a>
  * <a href='#Prepare Pre-trained Checkpoint'>2.1. Prepare Pre-trained Checkpoint</a>
  * <a href='#Instruction Tuning'>2.2. Instruction Tuning</a>
* <a href='#Evaluating STCInterLLM'>3. Evaluating STCInterLLM</a>
  * <a href='#Preparing Checkpoints and Data'>3.1. Preparing Checkpoints and Data</a>
  * <a href='#Running Evaluation'>3.2. Running Evaluation</a>
  * <a href='#Evaluation Metric Calculation'>3.3. Evaluation Metric Calculation</a>

****

<span id='Environment'/>

### 1. Environment

Please first clone the repo and install the required environment, which can be done by running the following commands:

```shell
conda create -n STCInterLLM python=3.9.13

conda activate STCInterLLM

# Torch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# To support vicuna base model
pip3 install "fschat[model_worker,webui]"

# To install pyg and pyg-relevant packages
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Clone our STCInterLLM or download it
git clone https://github.com/lishijie15/STCInterLLM.git
cd STCInterLLM

# Install required libraries
# (The recommendation is to install separately using the following method)
pip install deepspeed
pip install ray
pip install einops
pip install wandb
pip install flash-attn==2.3.5
pip install transformers==4.34.0

# （or you can install according to the requirements file.）
pip install -r requirements.txt
```

- All neural networks are implemented using PyTorch and trained on 8 NVIDIA A800 80GB GPUs / 8 NVIDIA H100 80GB GPUs.

<span id='Training STCInterLLM'/>

### 2. Training STCInterLLM

<span id='Prepare Pre-trained Checkpoint'/>

#### 2.1. Preparing Pre-trained Checkpoint

STCInterLLM is trained based on following excellent existing models.
Please follow the instructions to prepare the checkpoints.

- `Vicuna`:
  Prepare our base model Vicuna, which is an instruction-tuned chatbot and base model in our implementation. Please download its weights [here](https://github.com/lm-sys/FastChat#model-weights). We generally utilize v1.5 and v1.5-16k model with 7B parameters. You should update the 'config.json' of vicuna, for example, the 'config.json' in v1.5-16k can be found in [config.json](https://huggingface.co/datasets/bjdwh/checkpoints/blob/main/train_config/config.json)

- `Causal Encoder`:
  
  we specifically design the lightweight causal inference and intervention mechanism named causal intervention encoder. This is integrated with multi-scale temporal feature extraction modules to help the encoder enable the LLM to understand and learn complex spatio-temporal evolution patterns. The weights of [causal_encoder](./checkpoints/causal_encoder/pretrain_causalencoder.pth) are pre-trained through a typical multi-step spatio-temporal prediction task.
  
- `Spatio-temporal Train Data`:

  To evaluate the effectiveness of the proposed model in predicting spatio-temporal patterns across different scenarios, we have constructed two distinct scale scenarios. Scenario 1 involves net load forecasting for large-scale PDSs, considering the integration of substantial Renewable Energy Sources(RES). Scenario 2 focuses on traffic flow prediction, taking into account factors such as crime rates. These data are organized in [train_data](./STCInterLLM/ST_data_STCInterLLM/train_data). Please download them and put them at ./STCInterLLM/ST_data_STCInterLLM/train_data

<span id='Instruction Tuning'/>

#### 2.2. Instruction Tuning 

* **Start tuning:** After the aforementioned steps, you could start the instruction tuning by filling blanks at [STCInterLLM_train.sh](./STCInterLLM_train.sh). There is an example as below: 

```shell
# to fill in the following path to run our STCInterLLM!
model_path=./checkpoints/vicuna-7b-v1.5-16k
instruct_ds=./ST_data_urbangpt/train_10pv/train_10pv_only.json
st_data_path=./ST_data_urbangpt/train_10pv/train_pv10_only.pkl
pretra_ste=Causal_Encoder
output_model=./checkpoints/Causal_Encoder_7b_pv10

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=20001 \
    urbangpt/train/train_power.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --st_content ./TAXI.json \
    --st_data_path ${st_data_path} \
    --st_tower ${pretra_ste} \
    --tune_st_mlp_adapter True \
    --st_select_layer -2 \
    --use_st_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4800 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
    
```

<span id='Evaluating STCInterLLM'/>

### 3. Evaluating STCInterLLM

<span id='Preparing Checkpoints and Data'/>

#### 3.1. Preparing Checkpoints and Data 

* **Checkpoints:** You could try to evaluate STCInterLLM by using your own model or our released checkpoints.
* **Data:** We split test sets for Scenario 1 and Scenario 2 datasets and make the instruction data for evaluation. Please refer to the [evaluating](./STCInterLLM_eval.sh).

<span id='Running Evaluation'/>

#### 4.2. Running Evaluation

You could start the second stage tuning by filling blanks at [STCInterLLM_eval.sh](./STCInterLLM_eval.sh). There is an example as below: 

```shell
# to fill in the following path to evaluation!
output_model=./checkpoints/Causal_Encoder_7b_pv10
datapath=./ST_data_urbangpt/test_10pv/test_10pv_only.json
st_data_path=./ST_data_urbangpt/test_10pv/test_pv10_only.pkl
res_path=./result_test/Causal_Encoder_7b_pv10_
start_id=0
end_id=593208
num_gpus=8

python ./urbangpt/eval/test_stcinterllm.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}
```

#### 4.3. Evaluation Metric Calculation

<span id='Evaluation Metric Calculation'/>

You can use [result_test.py](./metric_calculation/result_test.py) to calculate the performance metrics of the predicted results. 

