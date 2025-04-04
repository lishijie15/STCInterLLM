# to fill in the following path to evaluation!
output_model=./checkpoints/Causal_Encoder_7b_pv10_noNorm
datapath=./ST_data/test_10pv/test_10pv.json
st_data_path=./ST_data/test_10pv/test_10pv.pkl
res_path=./result_test/Causal_Encoder_7b_pv10_noNorm_
start_id=0
end_id=593208
num_gpus=8

python ./STCILLM/eval/test_STCILLM_power.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}