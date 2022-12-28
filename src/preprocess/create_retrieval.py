import sys
# sys.path.append("../data/long-range-arena-main/lra_benchmarks/matching/")
import lra_matching_input_pipeline as input_pipeline
import tensorflow as tf
import tfrecord

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_matching_datasets(
    n_devices = 1, task_name = None, data_dir = "../data/lra_release/lra_release/tsv_data/",
    batch_size = 1, fixed_vocab = None, max_length = 4000, tokenizer = "char",
    vocab_file_path = None)

def concat_serialize(inst):
    return {"input_ids_0": tf.concat([inst["inputs1"][0], tf.zeros(96, dtype=tf.int32)], 0),
            "input_ids_1": tf.concat([inst["inputs2"][0], tf.zeros(96, dtype=tf.int32)], 0),
            "label": inst["targets"][0]}

mapping = {"train":train_ds, "dev": eval_ds, "test":test_ds}
for split, ds in mapping.items():
    ds = ds.map(concat_serialize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    writer = tfrecord.TFRecordWriter(f"../data/lra_processed/lra-retrieval.{split}.tfrecord")
    for idx, inst in enumerate(iter(ds)):
        writer.write({k: (v.numpy(), "int") for k, v in inst.items()})
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")
    writer.close()
