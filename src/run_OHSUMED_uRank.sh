./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_uRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_uRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_uRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_uRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_uRank.txt

echo "\n" >> OHSUMED_uRank.txt
cat experiments/base_model/params.json >> OHSUMED_uRank.txt