import yaml

# cannot be shared ones: use_chat_template

lengths_mapping = {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}
master_mapping = {
    "infoseek": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"vrag/infoseek_K{k[:-1]}_dep3.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "viquae": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"vrag/viquae_K{k[:-1]}_dep6.jsonl"
        } for k, v in lengths_mapping.items()
    },
    # visual haystack
    "vh_single": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/vh_single_test_1000_K{k[:-1]}_dep6.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "vh_multi": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/vh_multi_test_1000_K{k[:-1]}_dep3.jsonl"
        } for k, v in lengths_mapping.items()
    },
    # MM-NIAH
    "mm_niah_retrieval-text": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/retrieval-text_test_K{k[:-1]}_dep6.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "mm_niah_counting-text": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/counting-text_test_K{k[:-1]}_dep3.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "mm_niah_reasoning-text": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/reasoning-text_test_K{k[:-1]}_dep3.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "mm_niah_retrieval-image": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/retrieval-image_test_K{k[:-1]}_dep6.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "mm_niah_counting-image": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/counting-image_test_K{k[:-1]}_dep3.jsonl"
        } for k, v in lengths_mapping.items()
    },
    "mm_niah_reasoning-image": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"NIAH/reasoning-image_test_K{k[:-1]}_dep6.jsonl"
        } for k, v in lengths_mapping.items()
    },
    # cars196
    "cars196": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"ICL/cars196_K{k[:-1]}.json"
        } for k, v in lengths_mapping.items()
    },
    # food101
    "food101": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"ICL/food101_K{k[:-1]}.json"
        } for k, v in lengths_mapping.items()
    },
    # inat2021
    "inat2021": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"ICL/inat2021_K{k[:-1]}.json"
        } for k, v in lengths_mapping.items()
    },
    # sun397
    "sun397": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"ICL/sun397_K{k[:-1]}.json"
        } for k, v in lengths_mapping.items()
    },
    # gov-report
    "gov-report": {
        k: {
            "input_length": v, "generation_max_length": 384, "test_files": f"summ/gov_K{k[:-1]}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    # multi-lexsum
    "multi-lexsum": {
        k: {
            "input_length": v, "generation_max_length": 384, "test_files": f"summ/lexsum_K{k[:-1]}.jsonl"
        } for k, v in lengths_mapping.items()
    },
    # longdocurl
    "longdocurl": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"documentQA/longdocurl_K{k[:-1]}.jsonl",
        } for k, v in lengths_mapping.items()
    }, 
    # mmlongdoc
    "mmlongdoc": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"documentQA/mmlongdoc_K{k[:-1]}.jsonl",
        } for k, v in lengths_mapping.items()
    },
    # slidevqa
    "slidevqa": {
        k: {
            "input_length": v, "generation_max_length": 128, "test_files": f"documentQA/slidevqa_K{k[:-1]}.jsonl",
        } for k, v in lengths_mapping.items()
    }
}


def process_configs(config_name, datasets, input_lengths, **kwargs):
    configs = []
    for i, d in enumerate(datasets):
        con = master_mapping[d]
        print(d)
        for l in input_lengths:
            c = con[l]
            print(c)
            configs.append({
                "input_max_length": c['input_length'],
                "datasets": d,
                "generation_max_length": c['generation_max_length'],
                "test_files": c.get("test_files", ""),
            })
    out_config = {k: ",".join([str(c[k]) for c in configs]) for k in configs[0]}
    out_config.update({**kwargs})
    with open(config_name, "w") as f:
        yaml.dump(out_config, f, sort_keys=False)


def mmlb_configs(input_lengths = ["128k"], fname_postfix = ""):
    vrag = ['infoseek', 'viquae']
    process_configs(
        f"configs/vrag{fname_postfix}.yaml", vrag, input_lengths,
        use_chat_template = True, max_test_samples = 100,
    )

    vh = ["vh_single", "vh_multi"]
    process_configs(
        f"configs/vh{fname_postfix}.yaml", vh, input_lengths,
        use_chat_template = True, max_test_samples = 100,
    )

    mm_niah_text = ["mm_niah_retrieval-text", "mm_niah_counting-text", "mm_niah_reasoning-text"]
    process_configs(
        f"configs/mm_niah_text{fname_postfix}.yaml", mm_niah_text, input_lengths,
        use_chat_template = True, max_test_samples = 50,
    )

    mm_niah_image = ["mm_niah_retrieval-image", "mm_niah_counting-image", "mm_niah_reasoning-image"]
    process_configs(
        f"configs/mm_niah_image{fname_postfix}.yaml", mm_niah_image, input_lengths,
        use_chat_template = True, max_test_samples = 50,
    )

    icl = ["cars196", "food101", "inat2021", "sun397"]
    process_configs(
        f"configs/icl{fname_postfix}.yaml", icl, input_lengths,
        use_chat_template = True, max_test_samples = 100,
    )

    summ = ['gov-report', 'multi-lexsum']
    process_configs(
        f"configs/summ{fname_postfix}.yaml", summ, input_lengths,
        use_chat_template = True, max_test_samples = 100,
    )

    docqa = ["longdocurl", "mmlongdoc", "slidevqa"]
    process_configs(
        f"configs/docqa{fname_postfix}.yaml", docqa, input_lengths,
        use_chat_template = True, max_test_samples = 100,
    )


if __name__ == "__main__":
    mmlb_configs(input_lengths=["8k", "16k", "32k", "64k", "128k"], fname_postfix="_test")
