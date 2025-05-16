# TODO remember to change --test_file_root, --image_file_root, and --output_dir to your own folder
# TODO you can replace eval_api.py with eval_api_batch.py for faster inference
# gemini-2.0-flash-001
model_name=gemini-2.0-flash-001
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32 --api_sleep 3 --preprocessing_num_workers 1
done

# we use max image size 384 for non-text-rich images for 64K and 128K. Otherwise, there are much more tokens as Gemini use image tiling.
# gemini tile images when any side is longer than 384
# https://ai.google.dev/gemini-api/docs/image-understanding#technical-details-image
model_name=gemini-2.0-flash-001
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" ; do
  python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
  --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
  --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
  --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
  --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1 --max_image_num 400 --max_image_size 384
done

model_name=gemini-2.0-flash-001
for task in "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1
done


# gemini-2.0-flash-thinking-exp-01-21
model_name=gemini-2.0-flash-thinking-exp-01-21
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32 --api_sleep 3
done

model_name=gemini-2.0-flash-thinking-exp-01-21
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl"; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1 --max_image_num 400 --max_image_size 384
done

model_name=gemini-2.0-flash-thinking-exp-01-21
for task in "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1
done


# gemini-2.5-flash-preview-04-17
model_name=gemini-2.5-flash-preview-04-17
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32 --api_sleep 10 --preprocessing_num_workers 1
done

model_name=gemini-2.5-flash-preview-04-17
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl"; do
  python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
  --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
  --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
  --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
  --num_workers 16 --test_length 64,128 --api_sleep 10 --preprocessing_num_workers 1 --max_image_num 400 --max_image_size 384
done

model_name=gemini-2.5-flash-preview-04-17
for task in "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 64,128 --api_sleep 10 --preprocessing_num_workers 1
done


# gemini-2.5-pro
model_name=gemini-2.5-pro-preview-03-25
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32 --api_sleep 3 --preprocessing_num_workers 1
done

model_name=gemini-2.5-pro-preview-03-25
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1 --max_image_num 400 --max_image_size 384
done

model_name=gemini-2.5-pro-preview-03-25
for task in "summ" "docqa" ; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1
done

# TODO GPT-4o and Claude-3.7 support using web url,
# TODO you can use https://huggingface.co/datasets/ZhaoweiWang/image_collection/resolve/main to replace
# TODO the local image path for less pressure for you upload bandwidth
# ------------------------gpt-4o-2024-11-20------------------------
model_name=gpt-4o-2024-11-20
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl" "summ" "docqa"; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root https://huggingface.co/datasets/ZhaoweiWang/image_collection/resolve/main \
    --num_workers 16 --test_length 8,16,32 --api_sleep 3 --preprocessing_num_workers 1
done

model_name=gpt-4o-2024-11-20
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "icl"; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root https://huggingface.co/datasets/ZhaoweiWang/image_collection/resolve/main \
    --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1 --image_detail low
done

model_name=gpt-4o-2024-11-20
for task in "summ" "docqa"; do
    python eval_api_batch.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root https://huggingface.co/datasets/ZhaoweiWang/image_collection/resolve/main \
    --num_workers 16 --test_length 64,128 --api_sleep 3 --preprocessing_num_workers 1
done


# ------------------------claude-3-7-sonnet-20250219------------------------
# 8,16
model_name=claude-3-7-sonnet-20250219
for task in "vrag" "vh" "icl" "mm_niah_text" "mm_niah_image" "summ" "docqa"; do
    python eval_api_batch.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root /home/zhaowei.wang/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16 --api_sleep 3 --preprocessing_num_workers 1
done

# 32, cannot test ICL at 32K tokens
model_name=claude-3-7-sonnet-20250219
for task in "vrag" "vh" "mm_niah_text" "mm_niah_image" "summ" "docqa"; do
    python eval_api.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root https://huggingface.co/datasets/ZhaoweiWang/image_collection/resolve/main \
    --num_workers 16 --test_length 32 --api_sleep 3 --preprocessing_num_workers 1
done


# 64 cannot test vh, mm-niah-text, mm-niah-image, and icl at 64K tokens
model_name=claude-3-7-sonnet-20250219
for task in "vrag" "summ" "docqa"; do
    python eval_api_batch.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root https://huggingface.co/datasets/ZhaoweiWang/image_collection/resolve/main \
    --num_workers 16 --test_length 64 --api_sleep 3 --preprocessing_num_workers 1
done

# 128 cannot test vh, mm-niah-text, mm-niah-image, icl, and docqa at 64K tokens
model_name=claude-3-7-sonnet-20250219
for task in "vrag" "summ"; do
    python eval_api_batch.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /home/zhaowei.wang/data_dir/mmlb_result/${model_name} \
    --test_file_root /home/zhaowei.wang/data_dir/mmlb_data \
    --image_file_root https://huggingface.co/datasets/ZhaoweiWang/image_collection/resolve/main \
    --num_workers 16 --test_length 128 --api_sleep 3 --preprocessing_num_workers 1
done
