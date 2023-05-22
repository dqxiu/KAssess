# model_names=('chavinlo/alpaca-13b') 
# model_names=('lmsys/vicuna-7b-delta-v1.1')
model_names=('facebook/opt-66b')
# model_names=('t5-3b')
# model_names=('lmsys/viçcuna-13b-delta-v1.1')
# model_names=('vicuna-7b')
model_names=('THUDM/glm-10b')
# model_names=('databricks/dolly-v2-12b')
model_names=('lambdalabs/llama-13b_alpaca') 
# model_names=('Dogge/alpaca-13b')
for model_name in ${model_names[@]}
do
    echo "Task $model_name Started"
    CUDA_VISIBLE_DEVICES=5 python $YOUR_PROJECT_PATHcode/simple_rr_main_batch_update.py --model_name $model_name
    echo "Task $model_name Finished"
done
