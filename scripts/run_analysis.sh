# TODO remember to change --test_file_root, --image_file_root, and --output_dir to your own folder
#--------------------------------OCR extracted DocVQA--------------------------------
model_name=Qwen/Qwen2.5-VL-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-VL-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=1 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=2 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-VL-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=0,1 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-4b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-12b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-27b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_docqa"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


#--------------------------------VRAG with images replaced by its entity names--------------------------------
model_name=Qwen/Qwen2.5-VL-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=2 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-3B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-VL-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=4 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=Qwen/Qwen2.5-7B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=5 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-VL-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=6,7 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=Qwen/Qwen2.5-32B-Instruct
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=0,1 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done

model_name=google/gemma-3-4b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-12b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=2,3 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done


model_name=google/gemma-3-27b-it
dir_name=$(echo $model_name | rev | cut -d'/' -f1 | rev)
for task in "text_rag"; do
    CUDA_VISIBLE_DEVICES=0,1 python eval.py --config configs/${task}_all.yaml --model_name_or_path ${model_name} \
    --output_dir /workspace/zw/data_dir/mmlb_result/${dir_name} \
    --test_file_root /workspace/zw/data_dir/mmlb_data \
    --image_file_root /workspace/zw/data_dir/mmlb_image \
    --num_workers 16 --test_length 8,16,32,64,128
done
