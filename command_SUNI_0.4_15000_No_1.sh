#train
python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_1/train/cover /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_1/train/stego /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_1/valid/cover /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_1/valid/stego --log-path="/home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_1/log" --use-batch-norm --lr 4e-1 --max-epochs=200 --log-interval=32 --gpu="0,1"

#test
#python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/test/cover /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/test/stego --log-path="/home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/log" --gpu="0,1,2,3"

#data_split
#python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Data/SUNI_0.4_15000 /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2 --train-percent=0.6 --valid-percent=0.2 --test-percent=0.2 --gpu=" "
