import sys
# sys.path.append("../data/long-range-arena-main/lra_benchmarks/listops/")
import lra_listops_input_pipeline as input_pipeline
import tensorflow as tf
import tfrecord

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_datasets(
    n_devices = 1, task_name = "basic", data_dir = "../data/lra_release/lra_release/listops-1000/",
    batch_size = 1, max_length = 2000)

def concat_serialize(inst):
    return {"input_ids_0": tf.concat([inst["inputs"][0], tf.zeros(48, dtype=tf.int32)], 0),
            "label": inst["targets"][0]}

mapping = {"train":train_ds, "dev": eval_ds, "test":test_ds}
for split, ds in mapping.items():
    ds = ds.map(concat_serialize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    writer = tfrecord.TFRecordWriter(f"../data/lra_processed/lra-listops.{split}.tfrecord")
    for idx, inst in enumerate(iter(ds)):
        writer.write({k: (v.numpy(), "int") for k, v in inst.items()})
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")
    writer.close()

