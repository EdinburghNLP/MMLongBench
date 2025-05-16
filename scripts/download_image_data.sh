for file in vrag.tar.gz vh.tar.gz mm-niah.tar.gz icl.tar.gz summ.tar.gz docqa.tar.gz; do
  wget -c https://huggingface.co/datasets/ZhaoweiWang/MMLongBench/resolve/main/$file
  tar -xzvf $file
done