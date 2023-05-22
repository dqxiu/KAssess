for((i=63;i<=63;i++)); do
CUDA_VISIBLE_DEVICES=3 python $YOUR_PROJECT_PATHcode/wikidata_get.py --bash_id $i &
done