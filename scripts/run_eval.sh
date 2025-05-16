model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ
for task in "vrag"; do # Done
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn --no_bf16
done

model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ
for task in "vh"; do # Done
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn --no_bf16
done


model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ
for task in "mm_niah_text"; do # Done
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn --no_bf16
done

model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ
for task in "mm_niah_image"; do # Done
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn --no_bf16
done

model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ
for task in "icl"; do # Done
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn --no_bf16
done


model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ # Done
for task in "summ"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn --no_bf16
done


model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ # Done
for task in "docqa"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn --no_bf16
done


model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ # TODO
for task in "vrag" "vh"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 64 --use_yarn --no_bf16
done

model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ # Done
for task in "mm_niah_image"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 64 --use_yarn --no_bf16
done

model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ # Done
for task in "icl"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 64 --use_yarn --no_bf16
done

model_name=Qwen/Qwen2.5-VL-72B-Instruct-AWQ # Done
for task in "summ" "docqa"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-72B-Instruct-AWQ_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 64 --use_yarn --no_bf16
done

model_name=Qwen/Qwen2.5-VL-32B-Instruct
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-32B-Instruct_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 64 --use_yarn
done


model_name=Qwen/Qwen2.5-VL-32B-Instruct
for task in "summ" "docqa"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-32B-Instruct_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn
done


model_name=Qwen/Qwen2.5-VL-32B-Instruct
for task in "docqa"; do # "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/Qwen2.5-VL-32B-Instruct_yarn \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --use_yarn
done


#--------------------------------large scale inference part--------------------------------
# Qwen
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --gpu_list 0,1,2,3
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --gpu_list 0,1 \
--task_list mm_niah_text --length_list 128 --gpu_group_size 2

python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --gpu_list 0,1,2,3,4,5,6,7

python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_list 0,1,2,3 \
--length_list 8,16,32 --gpu_group_size 1; \
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_list 0,1,2,3 \
--length_list 8,16,32 --gpu_group_size 2
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_list 4,5,6,7 \
--length_list 128 --gpu_group_size 4

python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-72B-Instruct-AWQ --gpu_list 0,1,2,3 \
--length_list 8,16,32 --gpu_group_size 1 --no_bf16; \
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-72B-Instruct-AWQ --gpu_list 0,1,2,3 \
--length_list 8,16,32,64 --gpu_group_size 2 --no_bf16
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-72B-Instruct-AWQ --gpu_list 4,5,6,7 \
--length_list 128 --gpu_group_size 4 --no_bf16


#---------------------using yarn---------------------
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --gpu_list 0,1 --use_yarn; \
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --gpu_list 0,1 \
--task_list mm_niah_text --length_list 128 --gpu_group_size 2 --use_yarn

python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --gpu_list 2,3 --use_yarn

python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_list 4,5,6,7 \
--length_list 8,16,32 --gpu_group_size 1 --use_yarn; \
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_list 4,5,6,7 \
--length_list 8,16,32 --gpu_group_size 2 --use_yarn; \
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --gpu_list 4,5,6,7 \
--length_list 64,128 --gpu_group_size 4 --use_yarn

python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-72B-Instruct-AWQ --gpu_list 0,1,2,3 \
--length_list 8,16,32 --gpu_group_size 1 --no_bf16 --use_yarn; \
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-72B-Instruct-AWQ --gpu_list 0,1,2,3 \
--length_list 8,16,32,64 --gpu_group_size 2 --no_bf16 --use_yarn
python scripts/eval_task_manager.py --model_name Qwen/Qwen2.5-VL-72B-Instruct-AWQ --gpu_list 4,5,6,7 \
--length_list 128 --gpu_group_size 4 --no_bf16 --use_yarn

#---------------------end using yarn---------------------

python scripts/eval_task_manager.py --model_name Qwen/Qwen2-VL-2B-Instruct --gpu_list 0,1,2,3

python scripts/eval_task_manager.py --model_name Qwen/Qwen2-VL-7B-Instruct --gpu_list 0,1,2,3
python scripts/eval_task_manager.py --model_name Qwen/Qwen2-VL-7B-Instruct --gpu_list 0,1,2,3 \
--task_list "mm_niah_image" --length_list 128 --gpu_group_size 4

python scripts/eval_task_manager.py --model_name Qwen/Qwen2-VL-72B-Instruct-AWQ --gpu_list 4,5,6,7 \
--length_list 8,16,32 --gpu_group_size 1 --no_bf16; \
python scripts/eval_task_manager.py --model_name Qwen/Qwen2-VL-72B-Instruct-AWQ --gpu_list 4,5,6,7 \
--length_list 8,16,32 --gpu_group_size 2 --no_bf16
python scripts/eval_task_manager.py --model_name Qwen/Qwen2-VL-72B-Instruct-AWQ --gpu_list 4,5,6,7 \
--length_list 64,128 --gpu_group_size 4 --no_bf16

python scripts/eval_task_manager.py --model_name kosbu/QVQ-72B-Preview-AWQ --gpu_list 0,1,2,3 \
--length_list 8,16,32 --gpu_group_size 1 --no_bf16; \
python scripts/eval_task_manager.py --model_name kosbu/QVQ-72B-Preview-AWQ --gpu_list 0,1,2,3 \
--length_list 8,16,32,64 --gpu_group_size 2 --no_bf16
python scripts/eval_task_manager.py --model_name kosbu/QVQ-72B-Preview-AWQ --gpu_list 4,5,6,7 \
--length_list 128 --gpu_group_size 4 --no_bf16

# idefics 8B, we turn off image_splitting at longer context for non-text-rich images.
# Otherwise, there are a few times more tokens than our standardized length
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b --gpu_list 0,1,2,3
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b --gpu_list 0,1,2,3 \
--length_list 32 --gpu_group_size 4
# --do_image_splitting False
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b --gpu_list 0,1,2,3 \
--task_list "vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --do_image_splitting False
# summ, docqa
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b --gpu_list 4,5,6,7 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4

# idefics 8B chatty
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b-chatty --gpu_list 0,1,2,3
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b-chatty --gpu_list 4,5,6,7 \
--length_list 32 --gpu_group_size 4
# --do_image_splitting False
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b-chatty --gpu_list 4,5,6,7 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --do_image_splitting False
# summ, docqa
python scripts/eval_task_manager.py --model_name HuggingFaceM4/idefics2-8b-chatty --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2

# mantis, the default of --do_image_splitting is False
python scripts/eval_task_manager.py --model_name TIGER-Lab/Mantis-8B-Idefics2 --gpu_list 4,5,6,7
python scripts/eval_task_manager.py --model_name TIGER-Lab/Mantis-8B-Idefics2 --gpu_list 0,1,2,3 \
--task_list "docqa,vh,icl" --length_list 128 --gpu_group_size 4

# idefics 3 8B
python scripts/eval_task_manager.py --model_name HuggingFaceM4/Idefics3-8B-Llama3 --length_list 8,16,32 --gpu_list 0,1,2,3; \
python scripts/eval_task_manager.py --model_name HuggingFaceM4/Idefics3-8B-Llama3 --gpu_list 0,1,2,3 \
--task_list "vh,icl" --length_list 32 --gpu_group_size 4; \
# --do_image_splitting False
python scripts/eval_task_manager.py --model_name HuggingFaceM4/Idefics3-8B-Llama3 --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 2 --do_image_splitting False; \
python scripts/eval_task_manager.py --model_name HuggingFaceM4/Idefics3-8B-Llama3 --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --do_image_splitting False
# summ, docqa
# 1. When testing natural images with 64K and 128K length, we want the num_crops=1, so do_image_splitting=False
# 2. When testing text-rich images with 64K and 128K length, we want the num_crops=4 (while the default for Idefics3 is 16).
# However, the num_crops is hard to adjust for Idefics3, so we use image_resize=0.5 instead.
python scripts/eval_task_manager.py --model_name HuggingFaceM4/Idefics3-8B-Llama3 --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2 --image_resize 0.5

# Phi 3.5
python scripts/eval_task_manager.py --model_name microsoft/Phi-3.5-vision-instruct --gpu_list 0,1,2,3
python scripts/eval_task_manager.py --model_name microsoft/Phi-3.5-vision-instruct --gpu_list 0,1,2,3 \
--task_list "icl" --length_list 32 --gpu_group_size 4; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name microsoft/Phi-3.5-vision-instruct --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name microsoft/Phi-3.5-vision-instruct --gpu_list 6,7 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2

# Phi 3
python scripts/eval_task_manager.py --model_name microsoft/Phi-3-vision-128k-instruct --length_list 8,16,32 --gpu_list 0,1,2,3
python scripts/eval_task_manager.py --model_name microsoft/Phi-3-vision-128k-instruct --gpu_list 0,1,2,3 \
--task_list "icl" --length_list 32 --gpu_group_size 4; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name microsoft/Phi-3-vision-128k-instruct --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name microsoft/Phi-3-vision-128k-instruct --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4

# phi 4 only supprots 2-card parallelism
python scripts/eval_task_manager.py --model_name microsoft/Phi-4-multimodal-instruct --length_list 8,16,32 --gpu_list 4,5,6,7
python scripts/eval_task_manager.py --model_name microsoft/Phi-4-multimodal-instruct --gpu_list 4,5,6,7 \
--task_list "icl" --length_list 32 --gpu_group_size 2; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name microsoft/Phi-4-multimodal-instruct --gpu_list 4,5,6,7 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 2 --image_resize 0.5 --max_image_num 300
# summ, docqa
python scripts/eval_task_manager.py --model_name microsoft/Phi-4-multimodal-instruct --gpu_list 6,7 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2
# slidevqa for InternVL3-38B needs further truncation

# InternVl 3
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-1B --gpu_list 4,5 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-2B --gpu_list 6,7 --length_list 8,16,32 --do_prefill
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-2B --gpu_list 6,7 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 1 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-2B --gpu_list 6,7 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 1 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-8B --gpu_list 2,3 --length_list 8,16,32 --do_prefill
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-8B --gpu_list 0,1 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 1 --image_resize 0.5 --max_image_num 400 --do_prefill; \
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-8B --gpu_list 0,1 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 1 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-14B --gpu_list 4,5,6,7 --length_list 8,16,32 --do_prefill
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-14B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 1 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-14B --gpu_list 4,5,6,7 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-38B --gpu_list 0,1,2,3 --length_list 8,16,32 --gpu_group_size 2 --do_prefill
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-38B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 2 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL3-38B --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4 --do_prefill --vision_batch_size 8 # we need lower vision_batch_size for text-rich images
# slidevqa for InternVL3-38B needs 8 cards

# internvl 2.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-1B --gpu_list 0,1,2,3 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-2B --gpu_list 0,1,2,3 --length_list 8,16,32 --do_prefill
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-2B --gpu_list 0,1,2,3 \
--length_list 32 --gpu_group_size 4 --do_prefill
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-2B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-2B --gpu_list 4,5 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-4B --gpu_list 0,1,2,3

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-8B --gpu_list 4,5,6,7 --length_list 8,16,32 --do_prefill
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-8B --gpu_list 4,5,6,7 \
--length_list 32 --gpu_group_size 4 --do_prefill
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-8B --gpu_list 4,5,6,7 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-8B --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-26B --gpu_list 4,5,6,7 --length_list 8,16,32 --do_prefill
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-26B --gpu_list 4,5,6,7 \
--length_list 8,16,32 --gpu_group_size 4 --do_prefill
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-26B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2_5-26B --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4 --do_prefill

# V2PE
model_name=/workspace/zw/V2PE-256K
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/V2PE-256K_16 \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --do_prefill --v2pe_step 16
done

model_name=/workspace/zw/V2PE-256K
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/V2PE-256K_64 \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --do_prefill
done

model_name=/workspace/zw/V2PE-256K
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/V2PE-256K_256 \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 128 --do_prefill --v2pe_step 256
done


# internvl 2
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-1B --gpu_list 0,1,2,3

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-2B --gpu_list 0,1,2,3 --length_list 8,16,32 --do_prefill
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-2B --gpu_list 0,1,2,3 \
--task_list "icl" --length_list 32 --gpu_group_size 4 --do_prefill; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-2B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-2B --gpu_list 0,1 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-4B --gpu_list 0,1,2,3 --length_list 8,16,32 --do_prefill
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-4B --gpu_list 0,1,2,3 \
--task_list "icl,docqa" --length_list 32 --gpu_group_size 4 --do_prefill; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-4B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-4B --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4 --do_prefill

python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-8B --gpu_list 0,1,2,3 --length_list 8,16,32 --do_prefill
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-8B --gpu_list 0,1,2,3 \
--task_list "icl,docqa" --length_list 32 --gpu_group_size 4 --do_prefill; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-8B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400 --do_prefill
# summ, docqa
python scripts/eval_task_manager.py --model_name OpenGVLab/InternVL2-8B --gpu_list 4,5,6,7 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4 --do_prefill

# pixtral 12b
python scripts/eval_task_manager.py --model_name mistral-community/pixtral-12b --gpu_list 4,5,6,7 \
--length_list 8,16,32 --gpu_group_size 4 --vision_batch_size 6
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name mistral-community/pixtral-12b --gpu_list 4,5,6,7 \
--length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --vision_batch_size 16
#  larger images need smaller vision_batch_size
python scripts/eval_task_manager.py --model_name mistral-community/pixtral-12b --gpu_list 0,1,2,3 \
--length_list 64,128 --task_list "summ,docqa" --gpu_group_size 4 --image_resize 0.5 --vision_batch_size 1

# ovis2
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-1B --gpu_list 0,1,2,3 --length_list 8,16,32; \
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-1B --gpu_list 0,1,2,3 \
--task_list "vh,icl,docqa" --length_list 32 --gpu_group_size 2; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-1B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-1B --gpu_list 0,1 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2

python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-2B --gpu_list 0,1,2,3 --length_list 8,16,32; \
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-2B --gpu_list 0,1,2,3 \
--task_list "vh,icl,docqa" --length_list 32 --gpu_group_size 2; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-2B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-2B --gpu_list 2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2


python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-4B --gpu_list 0,1,2,3 --length_list 8,16,32; \
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-4B --gpu_list 0,1,2,3 \
--task_list "vh,icl,docqa" --length_list 32 --gpu_group_size 2; \
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-4B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-4B --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4

python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-8B --gpu_list 0,1,2,3 --length_list 8,16,32
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-8B --gpu_list 0,1,2,3 \
--task_list "vh,icl,docqa" --length_list 32 --gpu_group_size 2
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-8B --gpu_list 0,1,2,3 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-8B --gpu_list 2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 2

python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-16B --gpu_list 4,5,6,7 --length_list 8,16,32; \
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-16B --gpu_list 4,5,6,7 \
--length_list 8,16,32 --gpu_group_size 2
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-16B --gpu_list 4,5,6,7 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-16B --gpu_list 4,5,6,7 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4


python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-34B --gpu_list 0,1,2,3 --length_list 8; \
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-34B --gpu_list 0,1,2,3 \
--length_list 8,16,32 --gpu_group_size 4
# --image_resize 0.5
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-34B --gpu_list 4,5,6,7 \
--task_list "vrag,vh,mm_niah_text,mm_niah_image,icl" --length_list 64,128 --gpu_group_size 4 --image_resize 0.5 --max_image_num 400
# summ, docqa
python scripts/eval_task_manager.py --model_name AIDC-AI/Ovis2-34B --gpu_list 0,1,2,3 \
--task_list "summ,docqa" --length_list 64,128 --gpu_group_size 4

# nvila resize to 448x448, then 3x3 compression, the image tokens are not passing 128k
python scripts/eval_task_manager.py --model_name Efficient-Large-Model/NVILA-Lite-2B-hf-preview --gpu_list 0,1,2,3 \
--num_workers 0

python scripts/eval_task_manager.py --model_name Efficient-Large-Model/NVILA-Lite-8B-hf-preview --gpu_list 0,1,2,3 \
--num_workers 0
python scripts/eval_task_manager.py --model_name Efficient-Large-Model/NVILA-Lite-8B-hf-preview --gpu_list 4,5,6,7 \
--task_list "icl" --num_workers 0 --gpu_group_size 2

# gemma 3 image crop is disabled, this crop splits images into pre-defined corp number and then resize to 896x896
# all image then resize to 896x896, there is a 4x4 compression
# image tokens passing 128K by 25%
python scripts/eval_task_manager.py --model_name google/gemma-3-4b-it --gpu_list 4,5,6,7
python scripts/eval_task_manager.py --model_name google/gemma-3-4b-it --gpu_list 6,7 \
--task_list "vh,icl" --length_list 128 --gpu_group_size 2 --max_image_num 400

python scripts/eval_task_manager.py --model_name google/gemma-3-12b-it --gpu_list 0,1,2,3
python scripts/eval_task_manager.py --model_name google/gemma-3-12b-it --gpu_list 0,1,2,3 \
--task_list "vh,icl" --length_list 128 --gpu_group_size 2 --max_image_num 400

python scripts/eval_task_manager.py --model_name google/gemma-3-27b-it --length_list 8,16,32,64 --gpu_list 4,5,6,7; \
python scripts/eval_task_manager.py --model_name google/gemma-3-27b-it --length_list 8,16,32,64,128 --gpu_list 4,5,6,7 --gpu_group_size 4
python scripts/eval_task_manager.py --model_name google/gemma-3-27b-it --gpu_list 0,1,2,3 \
--task_list "vh,icl" --length_list 128 --gpu_group_size 4 --max_image_num 400
