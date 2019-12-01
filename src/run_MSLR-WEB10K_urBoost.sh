./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/1 --tfrecords_filename MSLR-WEB10K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/1 --tfrecords_filename MSLR-WEB10K.tfrecords >> MSLR-WEB10K_urBoost.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/2 --tfrecords_filename MSLR-WEB10K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/2 --tfrecords_filename MSLR-WEB10K.tfrecords >> MSLR-WEB10K_urBoost.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/3 --tfrecords_filename MSLR-WEB10K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/3 --tfrecords_filename MSLR-WEB10K.tfrecords >> MSLR-WEB10K_urBoost.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/4 --tfrecords_filename MSLR-WEB10K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/4 --tfrecords_filename MSLR-WEB10K.tfrecords >> MSLR-WEB10K_urBoost.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/5 --tfrecords_filename MSLR-WEB10K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ${TF_RANK_DATA}/MSLR-WEB10K/5 --tfrecords_filename MSLR-WEB10K.tfrecords >> MSLR-WEB10K_urBoost.txt

echo "\n" >> MSLR-WEB10K_urBoost.txt
cat experiments/base_model/params.json >> MSLR-WEB10K_urBoost.txt