./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/1 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/1 --tfrecords_filename MQ2007.tfrecords >> MQ2007_grank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn grank  --data_dir ${TF_RANK_DATA}/MQ2007/2 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn grank  --data_dir ${TF_RANK_DATA}/MQ2007/2 --tfrecords_filename MQ2007.tfrecords >> MQ2007_grank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/3 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/3 --tfrecords_filename MQ2007.tfrecords >> MQ2007_grank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/4 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/4 --tfrecords_filename MQ2007.tfrecords >> MQ2007_grank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/5 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn grank --data_dir ${TF_RANK_DATA}/MQ2007/5 --tfrecords_filename MQ2007.tfrecords >> MQ2007_grank.txt

echo "\n" >> MQ2007_grank.txt
cat experiments/base_model/params.json >> MQ2007_grank.txt