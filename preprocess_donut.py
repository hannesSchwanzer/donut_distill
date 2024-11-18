from PIL import Image
import json
from os import listdir, path


def preprocess_annotations(annotation_path):
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

        if len(links) == 0:
            continue

        answers = []
        for link in links:
            if field["id"] in link:
                link.remove(field["id"])

            if id_to_text[link[0]]["label"] != "answer":
                # print(f"Link is not of type answer ({annotation_path}):")
                # print("\tquestion:", question_text)
                # print(
                #     "\tlink:", id_to_text[link[0]]["label"], id_to_text[link[0]]["text"]
                # )
                continue

            answers.append(id_to_text[link[0]]["text"])

        if len(answers) == 0:
            qa_pairs.append({"question": question_text, "answer": ""})
        elif len(answers) == 1:
            qa_pairs.append({"question": question_text, "answer": answers[0]})
        else:
            qa_pairs.append({"question": question_text, "answers": answers})

    return {"qa_pairs": qa_pairs}


def preprocess_directory(directory_path):
    metadata_file = open(path.join(directory_path, "metadata.jsonl"), "w")

    image_directory = path.join(directory_path, "images")
    annotation_directory = path.join(directory_path, "annotations")
    file_ids = [
        file.removesuffix(".png")
        for file in listdir(image_directory)
        if path.isfile(path.join(image_directory, file))
    ]

    for id in file_ids:
        image_path = path.join(image_directory, f"{id}.png")
        annotation_path = path.join(annotation_directory, f"{id}.json")

        preprocessed_annotation = preprocess_annotations(annotation_path)
        file_metadata = {
            "file_name": image_path,
            "ground_truth": preprocessed_annotation,
        }
        metadata_file.write(json.dumps(file_metadata))
        metadata_file.write("\n")

    metadata_file.close()


if __name__ == "__main__":
    test_directory = "dataset/testing_data"
    train_directory = "dataset/training_data"
    preprocess_directory(test_directory)
    preprocess_directory(train_directory)
