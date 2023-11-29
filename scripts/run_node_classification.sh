#!/bin/bash

for a in Cora CiteSeer PubMed CoraFull Amazon-Computers Amazon-Photo CitationFull-DBLP Flickr Coauthor-CS Coauthor-Physics
do

    for seed in 1 10 100
    do
        for rank in 1 2 4 8 16 32 64 128 256 512
        do
            for embedding_dim in 1 2 4 8 16 32 64 128 256 512
            do
                for feature_dim in 0.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/1_1_deepwalk_classification_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/1_1_deepwalk_classification.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

                    pid=$!
                    wait $pid

                done

            done

        done

    done

done

   
for a in Cora CiteSeer PubMed CoraFull Amazon-Computers Amazon-Photo CitationFull-DBLP Flickr Coauthor-CS Coauthor-Physics
do

    for seed in 1 10 100
    do
        for rank in 1 2 4 8 16 32 64 128 256 512
        do
            for embedding_dim in 1 2 4 8 16 32 64 128 256 512
            do
                for feature_dim in 1.0 0.01 0.0001
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/1_2_sage_classification_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/1_2_sage_classification.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

                    pid=$!
                    wait $pid

                done

            done

        done

    done

done


for a in Cora CiteSeer PubMed CoraFull Amazon-Computers Amazon-Photo CitationFull-DBLP Flickr Coauthor-CS Coauthor-Physics
do

    for seed in 1 10 100
    do
        for rank in 1 2 4 8 16 32 64 128 256 512
        do
            for embedding_dim in 1 2 4 8 16 32 64 128 256 512
            do
                for feature_dim in 1.0 0.01 0.0001
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/1_3_gcn_classification_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/1_3_gcn_classification.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

                    pid=$!
                    wait $pid

                done

            done

        done

    done

done