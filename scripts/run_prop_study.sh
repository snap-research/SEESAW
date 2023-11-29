#!/bin/bash

for a in Cora CiteSeer PubMed CoraFull Amazon-Computers Amazon-Photo CitationFull-DBLP Flickr Coauthor-CS Coauthor-Physics
do

    for seed in 1 10 100
    do
        for rank in 16 32 64 128 256 512
        do
            for embedding_dim in 16 32 64 128 256 512
            do
                for feature_dim in 0.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/2_1_deepwalk_link_prediction_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/2_1_deepwalk_link_prediction.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

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
        for rank in 16 32 64 128 256 512
        do
            for embedding_dim in 16 32 64 128 256 512
            do
                for feature_dim in 0.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/2_4_prop_deepwalk_node_classification_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/2_4_prop_deepwalk_node_classification.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

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
        for rank in 16 32 64 128 256 512
        do
            for embedding_dim in 16 32 64 128 256 512
            do
                for feature_dim in 0.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/2_5_prop_deepwalk_link_prediction_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/2_5_prop_deepwalk_link_prediction.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

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
        for rank in 16 32 64 128 256 512
        do
            for embedding_dim in 16 32 64 128 256 512
            do
                for feature_dim in 1.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/2_2_sage_link_prediction_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/2_2_sage_link_prediction.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

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
        for rank in 16 32 64 128 256 512
        do
            for embedding_dim in 16 32 64 128 256 512
            do
                for feature_dim in 1.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/2_3_gcn_link_prediction_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/2_3_gcn_link_prediction.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

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
        for rank in 16 32 64 128 256 512
        do
            for embedding_dim in 16 32 64 128 256 512
            do
                for feature_dim in 1.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/2_6_prop_non_gcn_node_classification_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/2_6_prop_non_gcn_node_classification.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

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
        for rank in 16 32 64 128 256 512
        do
            for embedding_dim in 16 32 64 128 256 512
            do
                for feature_dim in 1.0
                do
                    c="Dataset = ${a} rank = ${rank} embedding_dim = ${embedding_dim} feature_dim = ${feature_dim}"

                    echo "Start ${c}"
                    output_file="../exp_logs/2_7_prop_non_gcn_link_prediction_${a}_verity_${rank}_${embedding_dim}_${seed}_${feature_dim}.txt"
                    most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
                    
                    CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python ../src/2_7_prop_non_gcn_link_prediction.py --dataset $a --rank $rank --embedding_dim $embedding_dim --seed $seed --feature_dim $feature_dim --cluster 0 > $output_file 2>&1 &

                    pid=$!
                    wait $pid

                done

            done

        done

    done

done

