import datasets
dataset = datasets.load_dataset("CLUTRR/v1", "gen_train234_test2to10")
dataset.save_to_disk('.')