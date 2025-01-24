import json
from os import listdir, path
from pathlib import Path
import shutil
from tqdm import tqdm


def preprocess_annotations_links_funsd(annotation_path):
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Extract the form field information
    fields = data.get("form", [])

    # Map ids to field label and text for lookup
    id_to_text = {
        field["id"]: {
            "text": field["text"],
            "label": field["label"],
        }
        for field in fields
    }

    qa_pairs = []
    for field in fields:
        question_text = field["text"]
        links = field.get("linking", [])

        if field["label"] != "question":
            continue

        answers = []
        for link in links:
            if field["id"] in link:
                link.remove(field["id"])

            if id_to_text[link[0]]["label"] != "answer":
                continue

            answers.append(id_to_text[link[0]]["text"])

        if len(answers) == 0:
            qa_pairs.append({"question": question_text, "answer": ""})
        elif len(answers) == 1:
            qa_pairs.append({"question": question_text, "answer": answers[0]})
        else:
            qa_pairs.append({"question": question_text, "answers": answers})

    return {"form": qa_pairs}


def preprocess_annotations_labels_funsd(annotation_path, max_datapoints : int | None = None):
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Extract the form field information
    fields = data.get("form", [])

    result = []
    for field in fields:
        text = field["text"]
        label = field["label"]

        result.append({"text": text, "label": label})

    return {"elements": result}



def preprocess_directory_funsd(directory_path, output_path, process_annotation_fn=preprocess_annotations_links_funsd, max_datapoints : int | None = None):
    # Create new directories and metadata file
    Path(output_path).mkdir(parents=True, exist_ok=True)
    metadata_file = open(path.join(output_path, "metadata.jsonl"), "w")

    image_directory = path.join(directory_path, "images")
    annotation_directory = path.join(directory_path, "annotations")
    file_ids = [
        file.removesuffix(".png")
        for file in listdir(image_directory)
        if path.isfile(path.join(image_directory, file))
    ]

    for i, id in enumerate(file_ids):
        if max_datapoints and i >= max_datapoints:
            break
        image_path = path.join(image_directory, f"{id}.png")
        annotation_path = path.join(annotation_directory, f"{id}.json")

        preprocessed_annotation = process_annotation_fn(annotation_path)
        gt = {"gt_parse": preprocessed_annotation}
        file_metadata = {
            "file_name": f"{id}.png",
            "ground_truth": json.dumps(gt),
        }
        metadata_file.write(json.dumps(file_metadata))
        metadata_file.write("\n")

        shutil.copy(image_path, output_path)

    metadata_file.close()


def preprocess_docvqa(annotations_path, images_path, output_path):

    for annotation_file_name in listdir(annotations_path):
        annotation_file_name = path.join(annotations_path, annotation_file_name)
        if path.isdir(annotation_file_name):
            continue

        with open(annotation_file_name, "r") as annotation_file:
            annotation_file = json.load(annotation_file)
        dataset_split = annotation_file["dataset_split"]
        if dataset_split == "test":
            continue

        data = annotation_file["data"]

        output_directory = path.join(output_path, dataset_split)
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        metadata_file = open(path.join(output_directory, "metadata.jsonl"), "w")

        for datapoint in tqdm(data, desc=dataset_split):
            image_path = path.join(images_path, datapoint["image"])
            image_name = path.basename(images_path)

            question = datapoint["question"]
            answers = datapoint["answers"]

            gt_parses = []
            for answer in answers:
                gt_parses.append({"question": question, "answer": answer})

            file_metadata = {
                "file_name": image_name,
                "ground_truth": json.dumps({"gt_parses": gt_parses}),
            }

            metadata_file.write(json.dumps(file_metadata))
            metadata_file.write("\n")

            shutil.copy(image_path, output_directory)

        metadata_file.close()



if __name__ == "__main__":
    # test_directory = "dataset/testing_data"
    # train_directory = "dataset/training_data"
    # process_annotation_fn=preprocess_annotations_labels_funsd
    # preprocess_directory(test_directory, "preprocessed_dataset/test", process_annotation_fn)
    # preprocess_directory(train_directory, "preprocessed_dataset/train", process_annotation_fn)

    # preprocess_directory(train_directory, "preprocessed_dataset/test", process_annotation_fn, max_datapoints=5)
    # preprocess_directory(train_directory, "preprocessed_dataset/train", process_annotation_fn, max_datapoints=5)

    # preprocess_directory_funsd(train_directory, "preprocessed_dataset/test", process_annotation_fn)
    # preprocess_directory_funsd(train_directory, "preprocessed_dataset/train", process_annotation_fn)

    preprocess_docvqa("docvqa/queries", "docvqa", "preprocessed_dataset_docvqa")

