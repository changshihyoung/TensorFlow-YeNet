#train
#python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/train/cover /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/train/stego /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/valid/cover /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/valid/stego --log-path="/home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/log" --use-batch-norm --lr 4e-1 --max-epochs=200 --log-interval=20 --gpu="2,3"

#test
python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/test/cover /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/test/stego --log-path="/home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2/log" --gpu="2,3"

#data_split
#python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Data/SUNI_0.4_15000 /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2 --train-percent=0.6 --valid-percent=0.2 --test-percent=0.2 --gpu=" "
