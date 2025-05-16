for i in {0..3}; do python scripts/eval_gpt4_summ.py --num_shards 4 --shard_idx $i & done
