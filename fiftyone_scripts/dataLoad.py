import argparse

import fiftyone as fo

from anylabeling_import import create_or_load_dataset, delete_dataset, import_images_with_anylabeling


def main():
    parser = argparse.ArgumentParser(description="Manage FiftyOne dataset and incremental X-AnyLabeling import")
    parser.add_argument("--create-dataset", type=str, help="Create or load a dataset by name")
    parser.add_argument("--delete-dataset", type=str, help="Delete a dataset by name")
    parser.add_argument("--image-dir", type=str, help="Directory containing source images")
    parser.add_argument("--labels-dir", type=str, help="Directory containing X-AnyLabeling JSON labels")
    parser.add_argument("--dataset-name", type=str, default="ppe_dataset", help="Dataset name used for import")
    parser.add_argument("--launch", action="store_true", help="Launch the FiftyOne app after import")
    args = parser.parse_args()

    if args.delete_dataset:
        delete_dataset(args.delete_dataset)
        return

    if args.create_dataset:
        create_or_load_dataset(args.create_dataset)
        return

    if args.image_dir and args.labels_dir:
        import_images_with_anylabeling(
            dataset_name=args.dataset_name,
            image_dir=args.image_dir,
            labels_dir=args.labels_dir,
            tags=["raw_import"],
        )

    if args.launch:
        dataset = fo.load_dataset(args.dataset_name)
        from anylabeling_import import ensure_ground_truth_field
        ensure_ground_truth_field(dataset)
        fo.launch_app(dataset, port=5151)


if __name__ == "__main__":
    main()