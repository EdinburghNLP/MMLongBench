# download images
for file in 1_vrag_image.tar.gz 2_vh_image.tar.gz 2_mm-niah_image.tar.gz 3_icl_image.tar.gz 4_summ_image.tar.gz 5_docqa_image.tar.gz; do
  wget -c https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/resolve/main/$file
done
# or
#for file in 1_vrag_image.tar.gz 2_vh_image.tar.gz 2_mm-niah_image.tar.gz 3_icl_image.tar.gz 4_summ_image.tar.gz 5_docqa_image.tar.gz; do
#  huggingface-cli download ZhaoweiWang/MMLongBench $file --local-dir ./ --repo-type dataset
#done

# decompress images
for file in 1_vrag_image.tar.gz 2_vh_image.tar.gz 2_mm-niah_image.tar.gz 3_icl_image.tar.gz 4_summ_image.tar.gz 5_docqa_image.tar.gz; do
  tar -xzvf "$file"
done