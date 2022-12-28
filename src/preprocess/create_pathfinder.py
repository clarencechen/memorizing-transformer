
import numpy as np
import os
import tensorflow as tf
import tfrecord

root_dir = "../data/lra_release/lra_release/"
subdir = "pathfinder32"

def filename_to_image(inst):
    metadata = tf.strings.split(inst)
    return {"input_ids_0": tf.io.read_file(tf.strings.join([data_dir, metadata[0], metadata[1]], separator=os.sep)), 
            "label": tf.strings.to_number(metadata[3], tf.int32)}

def image_to_seq(inst):
    return {"input_ids_0": tf.reshape(tf.cast(tf.image.decode_png(inst["input_ids_0"]), tf.int32), (-1,)),
            "label": inst["label"]}

for diff_level in ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]:
    data_dir = os.path.join(root_dir, subdir, diff_level)
    metadata_list = [
        os.path.join(data_dir, "metadata", file)
        for file in os.listdir(os.path.join(data_dir, "metadata"))
        if file.endswith(".npy")
    ]

    for idx, metadata_file in enumerate(metadata_list):
        print(idx, len(metadata_list), metadata_file, "\t\t", end = "\r")
    ds = tf.data.TextLineDataset(metadata_list, buffer_size=0, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(filename_to_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x: tf.strings.length(x["input_ids_0"]) > 0)
    ds = ds.map(image_to_seq, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    mapping = {k: tfrecord.TFRecordWriter(f"../data/lra_processed/lra-{subdir}-{diff_level}.{k}.tfrecord") for k in ["train", "dev", "test"]}
    for idx, inst in enumerate(iter(ds)):
        writer = mapping["train"]
        if idx % 10 == 8:
            writer = mapping["dev"]
        elif idx % 10 == 9:
            writer = mapping["test"]
        writer.write({k: (v.numpy(), "int") for k, v in inst.items()})
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")

    for writer in mapping.values():
        writer.close()
