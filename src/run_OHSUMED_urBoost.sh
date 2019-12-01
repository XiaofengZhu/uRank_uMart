./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urBoost.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urBoost.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urBoost.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urBoost.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urBoost.txt

echo "\n" >> OHSUMED_urBoost.txt
cat experiments/base_model/params.json >> OHSUMED_urBoost.txt