from anylabeling_import import import_images_with_anylabeling


if __name__ == "__main__":
    import_images_with_anylabeling(
        dataset_name="ppe_dataset",
        image_dir="/media/images/ppe",
        labels_dir="/media/images/ppe_xany",
        tags=["raw_import"],
    )